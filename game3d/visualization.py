"""PyVista user interface for the standalone 3D digital twin."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .environment3d import RealCPTEnvironment3D, SyntheticEnvironment3D


def headless_smoke_test(environment) -> dict:
    """Validate PyVista meshes without opening an OpenGL render window."""

    try:
        import pyvista as pv
    except ImportError as exc:
        raise RuntimeError("PyVista is required for visualization") from exc
    values = environment.truth if isinstance(environment, SyntheticEnvironment3D) else environment.prediction
    z = np.asarray(environment.z)
    if z[0] > z[-1]:
        z = z[::-1]
        values = values[:, :, ::-1]
    spacing = (
        float(np.mean(np.diff(environment.x))),
        float(np.mean(np.diff(environment.y))),
        float(np.mean(np.diff(z))),
    )
    grid = pv.ImageData(
        dimensions=values.shape,
        spacing=spacing,
        origin=(float(environment.x[0]), float(environment.y[0]), float(z[0])),
    )
    grid.point_data["value"] = np.nan_to_num(values, nan=0.0).ravel(order="F")
    vehicle = pv.Box()
    profile = pv.Line((0, 0, float(z.min())), (0, 0, float(z.max())))
    return {
        "volume_points": grid.n_points,
        "volume_shape": tuple(values.shape),
        "vehicle_cells": vehicle.n_cells,
        "profile_cells": profile.n_cells,
    }


class DigitalTwin3D:
    """Interactive scene; environment logic remains independent of PyVista."""

    def __init__(self, environment, *, off_screen: bool = False):
        try:
            import pyvista as pv
        except ImportError as exc:
            raise RuntimeError(
                "PyVista is required. Install game3d/requirements-game3d.txt"
            ) from exc
        self.pv = pv
        self.environment = environment
        self.plotter = pv.Plotter(
            shape=(1, 2) if isinstance(environment, SyntheticEnvironment3D) else (1, 1),
            off_screen=off_screen,
            window_size=(1500, 800),
        )
        self.status = "WASD/arrows: move | Space: sample/preview | Mouse: rotate, zoom, pan"
        self._register_controls()
        self.render_scene()

    def _register_controls(self) -> None:
        keys = {
            "w": (0, 1),
            "Up": (0, 1),
            "s": (0, -1),
            "Down": (0, -1),
            "a": (-1, 0),
            "Left": (-1, 0),
            "d": (1, 0),
            "Right": (1, 0),
        }
        for key, movement in keys.items():
            self.plotter.add_key_event(
                key, lambda movement=movement: self._move(*movement)
            )
        self.plotter.add_key_event("space", self._sample)

    def _move(self, dx: int, dy: int) -> None:
        self.environment.move(dx, dy)
        self.status = f"Vehicle moved to {self._vehicle_xy_text()}"
        self.render_scene(reset_camera=False)

    def _sample(self) -> None:
        event = self.environment.sample()
        self.status = event.message
        self.render_scene(reset_camera=False)

    def _vehicle_xy(self) -> tuple[float, float]:
        environment = self.environment
        if isinstance(environment, RealCPTEnvironment3D):
            return environment.vehicle_xy
        return float(environment.vehicle[0]), float(environment.vehicle[1])

    def _vehicle_xy_text(self) -> str:
        x, y = self._vehicle_xy()
        return f"({x:.1f}, {y:.1f})"

    def _volume_grid(self, values: np.ndarray, x, y, z):
        values = np.asarray(values)
        x, y, z = np.asarray(x), np.asarray(y), np.asarray(z)
        if z[0] > z[-1]:
            z = z[::-1]
            values = values[:, :, ::-1]
        spacing = (
            float(np.mean(np.diff(x))) if len(x) > 1 else 1.0,
            float(np.mean(np.diff(y))) if len(y) > 1 else 1.0,
            float(np.mean(np.diff(z))) if len(z) > 1 else 1.0,
        )
        grid = self.pv.ImageData(
            dimensions=values.shape,
            spacing=spacing,
            origin=(float(x[0]), float(y[0]), float(z[0])),
        )
        grid.point_data["value"] = np.nan_to_num(values, nan=0.0).ravel(order="F")
        return grid

    def _add_volume(self, values, title: str, subplot: int) -> None:
        env = self.environment
        self.plotter.subplot(0, subplot)
        grid = self._volume_grid(values, env.x, env.y, env.z)
        finite = np.asarray(values)[np.isfinite(values)]
        clim = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)
        self.plotter.add_volume(
            grid,
            scalars="value",
            cmap="viridis",
            opacity="sigmoid_6",
            clim=clim,
            scalar_bar_args={"title": f"{env.parameter} [{env.unit}]"},
        )
        self.plotter.add_mesh(grid.outline(), color="black", line_width=1)
        self.plotter.add_text(title, position="upper_left", font_size=12)

    def _add_vehicle(self, subplot: int) -> None:
        self.plotter.subplot(0, subplot)
        x, y = self._vehicle_xy()
        top = float(np.max(self.environment.z)) + 0.8
        scale_x = max(float(np.ptp(self.environment.x)) / 18.0, 0.8)
        scale_y = max(float(np.ptp(self.environment.y)) / 18.0, 0.8)
        body = self.pv.Box(
            bounds=(x - scale_x, x + scale_x, y - scale_y, y + scale_y, top, top + 0.7)
        )
        cabin = self.pv.Box(
            bounds=(x - 0.5 * scale_x, x + 0.4 * scale_x, y - 0.7 * scale_y, y + 0.7 * scale_y, top + 0.7, top + 1.25)
        )
        self.plotter.add_mesh(body, color="#f4a261")
        self.plotter.add_mesh(cabin, color="#264653")

    def _add_profile_line(self, x: float, y: float, z_min: float, z_max: float, color: str, subplot: int) -> None:
        self.plotter.subplot(0, subplot)
        self.plotter.add_mesh(
            self.pv.Line((x, y, z_min), (x, y, z_max)),
            color=color,
            line_width=6,
        )

    def _add_synthetic_annotations(self, subplot: int) -> None:
        env = self.environment
        for x, y in env.sample_positions:
            self._add_profile_line(x, y, float(env.z.min()), float(env.z.max()), "red", subplot)
        self._add_vehicle(subplot)

    def _add_real_annotations(self, subplot: int) -> None:
        env = self.environment
        for cpt_id, (x, y), elevation in zip(
            env.dataset.ids, env.dataset.positions_xy, env.dataset.elevations_m
        ):
            self._add_profile_line(
                float(x), float(y), float(elevation - 15.0), float(elevation), "red", subplot
            )
            self.plotter.add_point_labels(
                np.asarray([[x, y, elevation]]),
                [cpt_id],
                point_color="red",
                text_color="black",
                font_size=12,
            )
        if env.last_prediction_event:
            x, y = env.last_prediction_event.position_xy
            self._add_profile_line(x, y, float(env.z.min()), float(env.z.max()), "orange", subplot)
        self._add_vehicle(subplot)

    def render_scene(self, *, reset_camera: bool = True) -> None:
        env = self.environment
        camera_positions = None
        if not reset_camera:
            camera_positions = [renderer.camera_position for renderer in self.plotter.renderers]
        self.plotter.clear()
        if isinstance(env, SyntheticEnvironment3D):
            self._add_volume(env.truth, "Synthetic ground truth (IC)", 0)
            self._add_synthetic_annotations(0)
            self.plotter.subplot(0, 1)
            if env.prediction is None:
                empty = self._volume_grid(np.zeros_like(env.truth), env.x, env.y, env.z)
                self.plotter.add_mesh(empty.outline(), color="black")
                self.plotter.add_text("Sample with Space to reconstruct", position="upper_left")
            else:
                self._add_volume(env.prediction, "Horizontal IDW prediction (IC)", 1)
            self._add_synthetic_annotations(1)
            stats = (
                f"CPT: {env.cpt_count} | Vehicle: {self._vehicle_xy_text()} | "
                f"RMSE: {env.rmse:.4f} | MAE: {env.mae:.4f}"
                if env.cpt_count
                else f"CPT: 0 | Vehicle: {self._vehicle_xy_text()} | RMSE: n/a"
            )
            self.plotter.link_views()
        else:
            self._add_volume(env.prediction, "Real qc - horizontal IDW prediction", 0)
            self._add_real_annotations(0)
            local_uncertainty = float(env.uncertainty[tuple(env.vehicle)].mean())
            stats = (
                f"Measured CPT: 3 (red) | Vehicle: {self._vehicle_xy_text()} | "
                f"prediction uncertainty proxy: {local_uncertainty:.2f}\n"
                "HIGH UNCERTAINTY: only three measured CPTs; orange is prediction only"
            )
        self.plotter.subplot(0, 0)
        self.plotter.add_text(f"{stats}\n{self.status}", position="lower_left", font_size=9)
        if camera_positions:
            for renderer, position in zip(self.plotter.renderers, camera_positions):
                renderer.camera_position = position
        else:
            self.plotter.view_isometric()
        self.plotter.render()

    def show(self, screenshot: Path | None = None) -> None:
        self.plotter.show(screenshot=str(screenshot) if screenshot else None)
