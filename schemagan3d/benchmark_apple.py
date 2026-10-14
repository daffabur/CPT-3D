"""Benchmark 3D GAN optimization on CPU or Apple MPS.

Set PYTORCH_ENABLE_MPS_FALLBACK=1 before Python starts to benchmark the
existing ConvTranspose3d model with CPU fallback.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import time

import torch

from .experimental_models import Generator3DUpsample
from .models import Discriminator3D, Generator3D
from .training import optimization_step


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


def operator_probe(device: torch.device) -> dict:
    result = {}
    for dtype in (torch.float32, torch.float16):
        key = str(dtype).removeprefix("torch.")
        try:
            layer = torch.nn.ConvTranspose3d(2, 2, 4, 2, 1, device=device, dtype=dtype)
            value = torch.randn(1, 2, 2, 4, 4, device=device, dtype=dtype, requires_grad=True)
            layer(value).sum().backward()
            synchronize(device)
            result[key] = "forward+backward passed"
        except Exception as exc:
            result[key] = f"{type(exc).__name__}: {exc}"
    return result


def peak_memory(device: torch.device) -> dict[str, float]:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_mib = rss / 1024**2 if platform.system() == "Darwin" else rss / 1024
    result = {"process_peak_rss_mib": float(rss_mib)}
    if device.type == "mps":
        result.update(
            mps_current_allocated_mib=torch.mps.current_allocated_memory() / 1024**2,
            mps_driver_allocated_mib=torch.mps.driver_allocated_memory() / 1024**2,
        )
    return result


def benchmark(model_name: str, device: torch.device, steps: int, batch_size: int) -> dict:
    generator_class = Generator3D if model_name == "transpose" else Generator3DUpsample
    torch.manual_seed(14)
    generator = generator_class(16).to(device)
    discriminator = Discriminator3D(16).to(device)
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    target = torch.randn(batch_size, 1, 8, 16, 16, device=device)
    mask_xy = torch.zeros(batch_size, 1, 1, 16, 16, device=device)
    for item in range(batch_size):
        mask_xy[item, :, :, item % 16, (item * 3) % 16] = 1.0
    mask = mask_xy.expand(-1, -1, 8, -1, -1).clone()
    batch = {"target": target, "mask": mask, "sparse": target * mask}
    for _ in range(2):
        optimization_step(generator, discriminator, batch, g_optimizer, d_optimizer)
    synchronize(device)
    durations = []
    losses = None
    for _ in range(steps):
        started = time.perf_counter()
        losses = optimization_step(generator, discriminator, batch, g_optimizer, d_optimizer)
        synchronize(device)
        durations.append(time.perf_counter() - started)
    return {
        "model": model_name,
        "device": str(device),
        "batch_size": batch_size,
        "measured_steps": steps,
        "total_seconds": sum(durations),
        "mean_seconds_per_step": sum(durations) / len(durations),
        "min_seconds_per_step": min(durations),
        "max_seconds_per_step": max(durations),
        "generator_parameters": sum(p.numel() for p in generator.parameters()),
        "discriminator_parameters": sum(p.numel() for p in discriminator.parameters()),
        "last_losses": losses,
        "memory": peak_memory(device),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--model", choices=("transpose", "upsample", "both"), default="both")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    device = torch.device(args.device)
    report = {
        "system": {
            "platform": platform.platform(),
            "macos": platform.mac_ver()[0],
            "machine": platform.machine(),
            "torch": torch.__version__,
            "mps_built": torch.backends.mps.is_built(),
            "mps_available": torch.backends.mps.is_available(),
            "mps_fallback_env": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0"),
        },
        "operator_probe": operator_probe(device),
        "benchmarks": [],
    }
    if device.type == "mps" and not torch.backends.mps.is_available():
        report["error"] = "MPS is not available to this Python process"
    else:
        models = ("transpose", "upsample") if args.model == "both" else (args.model,)
        for model in models:
            try:
                report["benchmarks"].append(benchmark(model, device, args.steps, args.batch_size))
            except Exception as exc:
                report["benchmarks"].append(
                    {"model": model, "device": str(device), "error": f"{type(exc).__name__}: {exc}"}
                )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
