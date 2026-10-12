"""Command-line entry point for CPT Super Learn - Digital Twin 3D."""

from __future__ import annotations

import argparse
from pathlib import Path

from .data_loader import extract_pdf_to_csv
from .environment3d import RealCPTEnvironment3D, SyntheticEnvironment3D
from .visualization import DigitalTwin3D, headless_smoke_test


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "real_cpt"
DEFAULT_PDF = DATA_DIR / "SONDIR SONDIR NGC 2025(1).pdf"
DEFAULT_CSV = DATA_DIR / "sondir_ngc_2025.csv"
DEFAULT_REPORT = DATA_DIR / "extraction_report.json"


def ensure_real_csv(pdf_path: Path, csv_path: Path) -> None:
    if csv_path.exists():
        return
    if not pdf_path.exists():
        alternate = pdf_path.with_name("SONDIR SONDIR NGC 2025.pdf")
        pdf_path = alternate if alternate.exists() else pdf_path
    if not pdf_path.exists():
        raise FileNotFoundError(
            f"Real CPT PDF not found at {pdf_path}. Put it in {DATA_DIR}."
        )
    report = extract_pdf_to_csv(pdf_path, csv_path, DEFAULT_REPORT)
    print(
        f"Extracted {report['measured_rows']} measured rows from "
        f"{report['cpt_count']} CPT points into {csv_path}"
    )


def choose_mode() -> str:
    print("CPT Super Learn - Digital Twin 3D")
    print("1. Synthetic 3D Mode")
    print("2. Real CPT Data Mode")
    choice = input("Choose mode [1/2]: ").strip()
    if choice not in {"1", "2"}:
        raise SystemExit("Invalid mode. Choose 1 or 2.")
    return "synthetic" if choice == "1" else "real"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("synthetic", "real"))
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--extract-only", action="store_true")
    parser.add_argument("--off-screen", action="store_true", help="Render one frame without GUI")
    parser.add_argument("--screenshot", type=Path)
    args = parser.parse_args()

    if args.extract_only:
        extract_pdf_to_csv(args.pdf, args.csv, DEFAULT_REPORT)
        print(f"CSV written to {args.csv}")
        return
    mode = args.mode or choose_mode()
    if mode == "synthetic":
        environment = SyntheticEnvironment3D()
    else:
        ensure_real_csv(args.pdf, args.csv)
        environment = RealCPTEnvironment3D(args.csv)
    if args.off_screen:
        result = headless_smoke_test(environment)
        print(f"Headless PyVista smoke test passed: {result}")
        if args.screenshot:
            print("Screenshot skipped: headless sandbox has no macOS OpenGL context.")
        return
    game = DigitalTwin3D(environment, off_screen=False)
    game.show(args.screenshot)


if __name__ == "__main__":
    main()
