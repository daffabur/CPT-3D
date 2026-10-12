"""Extraction and loading of the real CPT PDF and normalized CSV."""

from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


CSV_COLUMNS = (
    "cpt_id",
    "latitude_deg",
    "longitude_deg",
    "local_x_m",
    "local_y_m",
    "surface_elevation_m",
    "depth_m",
    "vertical_rl_m",
    "qc_kg_cm2",
    "biconus_h_kg_cm2",
    "qf_kg_cm2",
    "hl_kg_cm2",
    "total_hl_kg_cm2",
    "local_friction_kg_cm2",
    "friction_ratio_pct",
    "local_friction_reading_kg_cm2",
    "total_friction_reading_kg_cm2",
)


@dataclass(frozen=True)
class CPTDataset:
    ids: tuple[str, ...]
    positions_xy: np.ndarray
    elevations_m: np.ndarray
    depths_m: np.ndarray
    qc_profiles: np.ndarray
    latitudes: np.ndarray
    longitudes: np.ndarray


def dms_to_decimal(degrees: float, minutes: float, seconds: float, hemisphere: str) -> float:
    value = float(degrees) + float(minutes) / 60.0 + float(seconds) / 3600.0
    return -value if hemisphere.upper() in {"S", "W", "B"} else value


def geographic_to_local(
    latitudes: np.ndarray, longitudes: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Convert geographic coordinates to local east/north metres about S1."""

    latitudes = np.asarray(latitudes, dtype=float)
    longitudes = np.asarray(longitudes, dtype=float)
    radius = 6_371_008.8
    lat0, lon0 = np.deg2rad(latitudes[0]), np.deg2rad(longitudes[0])
    lat, lon = np.deg2rad(latitudes), np.deg2rad(longitudes)
    east = radius * (lon - lon0) * np.cos((lat + lat0) / 2.0)
    north = radius * (lat - lat0)
    return east, north


def _parse_dms(text: str, hemisphere: str) -> float:
    match = re.search(
        rf"(\d+)°\s*(\d+)'\s*([\d.]+)\"\s*{hemisphere}", text, re.I
    )
    if not match:
        raise ValueError(f"coordinate {hemisphere} was not found")
    return dms_to_decimal(*map(float, match.groups()), hemisphere)


def extract_pdf_to_csv(
    pdf_path: Path, csv_path: Path, report_path: Path | None = None
) -> dict:
    """Extract all three tabular pages and write a normalized, audited CSV."""

    try:
        import pdfplumber
    except ImportError as exc:
        raise RuntimeError("PDF extraction requires pdfplumber") from exc

    records: list[dict] = []
    site_meta: list[dict] = []
    with pdfplumber.open(pdf_path) as document:
        if len(document.pages) != 6:
            raise ValueError(f"expected 6 PDF pages, found {len(document.pages)}")
        for page_number in (1, 3, 5):
            page = document.pages[page_number - 1]
            text = page.extract_text() or ""
            id_match = re.search(r"Titik no\.\s*:\s*(S\d+)", text)
            elevation_match = re.search(r"Elevasi\s*:\s*([\d,.]+)", text)
            if not id_match or not elevation_match:
                raise ValueError(f"metadata missing on page {page_number}")
            cpt_id = id_match.group(1)
            latitude = _parse_dms(text, "S")
            longitude = _parse_dms(text, "T")
            elevation = float(elevation_match.group(1).replace(",", "."))
            tables = page.extract_tables()
            if len(tables) != 1 or len(tables[0]) != 77:
                raise ValueError(f"unexpected table layout on page {page_number}")
            header = " ".join(tables[0][0])
            required_header_tokens = ("Depth", "(m)", "Conus", "kg/cm2", "Friction Ratio", "(%)")
            if any(token not in header for token in required_header_tokens):
                raise ValueError(f"expected units were not found on page {page_number}")
            rows = tables[0][1:]
            for expected_number, row in enumerate(rows, start=1):
                if len(row) != 11 or int(row[0]) != expected_number:
                    raise ValueError(f"invalid row {expected_number} for {cpt_id}")
                numbers = [None if value == "-" else float(value) for value in row[1:]]
                depth = numbers[0]
                records.append(
                    {
                        "cpt_id": cpt_id,
                        "latitude_deg": latitude,
                        "longitude_deg": longitude,
                        "surface_elevation_m": elevation,
                        "depth_m": depth,
                        "vertical_rl_m": elevation - depth,
                        "qc_kg_cm2": numbers[1],
                        "biconus_h_kg_cm2": numbers[2],
                        "qf_kg_cm2": numbers[3],
                        "hl_kg_cm2": numbers[4],
                        "total_hl_kg_cm2": numbers[5],
                        "local_friction_kg_cm2": numbers[6],
                        "friction_ratio_pct": numbers[7],
                        "local_friction_reading_kg_cm2": numbers[8],
                        "total_friction_reading_kg_cm2": numbers[9],
                    }
                )
            site_meta.append(
                {
                    "cpt_id": cpt_id,
                    "latitude": latitude,
                    "longitude": longitude,
                    "elevation_m": elevation,
                }
            )

    east, north = geographic_to_local(
        np.asarray([site["latitude"] for site in site_meta]),
        np.asarray([site["longitude"] for site in site_meta]),
    )
    coordinates = {
        site["cpt_id"]: (float(x), float(y))
        for site, x, y in zip(site_meta, east, north)
    }
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for record in records:
            record["local_x_m"], record["local_y_m"] = coordinates[record["cpt_id"]]
            writer.writerow({key: "" if record.get(key) is None else record[key] for key in CSV_COLUMNS})

    measured = [record for record in records if record["qc_kg_cm2"] is not None]
    report = {
        "source_pdf": pdf_path.name,
        "coordinate_reference": "local east/north metres relative to S1 (equirectangular)",
        "vertical_reference": "RL = surface elevation - depth, metres",
        "primary_parameter": "qc_kg_cm2",
        "cpt_count": len(site_meta),
        "rows_total": len(records),
        "measured_rows": len(measured),
        "missing_qc_rows": len(records) - len(measured),
        "depth_interval_m": 0.2,
        "depth_range_m": [0.0, 15.0],
        "verified_units": {
            "depth": "m",
            "qc": "kg/cm2",
            "local_friction": "kg/cm2",
            "friction_ratio": "%",
        },
        "sites": site_meta,
    }
    if report_path:
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def load_cpt_csv(csv_path: Path) -> CPTDataset:
    grouped: dict[str, list[dict[str, str]]] = {}
    with csv_path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            grouped.setdefault(row["cpt_id"], []).append(row)
    ids = tuple(sorted(grouped))
    if ids != ("S1", "S2", "S3"):
        raise ValueError(f"expected S1, S2, S3; found {ids}")
    depths = np.arange(0.2, 15.0 + 0.1, 0.2)
    profiles, positions, elevations, latitudes, longitudes = [], [], [], [], []
    for cpt_id in ids:
        rows = grouped[cpt_id]
        measured = [row for row in rows if row["qc_kg_cm2"]]
        measured_depths = np.asarray([float(row["depth_m"]) for row in measured])
        if len(measured) != 75 or not np.allclose(measured_depths, depths):
            raise ValueError(f"{cpt_id} does not contain 75 qc values at 0.2 m intervals")
        profiles.append([float(row["qc_kg_cm2"]) for row in measured])
        positions.append([float(rows[0]["local_x_m"]), float(rows[0]["local_y_m"])])
        elevations.append(float(rows[0]["surface_elevation_m"]))
        latitudes.append(float(rows[0]["latitude_deg"]))
        longitudes.append(float(rows[0]["longitude_deg"]))
    return CPTDataset(
        ids=ids,
        positions_xy=np.asarray(positions),
        elevations_m=np.asarray(elevations),
        depths_m=depths,
        qc_profiles=np.asarray(profiles),
        latitudes=np.asarray(latitudes),
        longitudes=np.asarray(longitudes),
    )


def profiles_on_common_rl(dataset: CPTDataset) -> tuple[np.ndarray, np.ndarray]:
    """Resample qc onto a common absolute reduced-level grid without extrapolation."""

    step = 0.2
    lower = math.floor(np.min(dataset.elevations_m - dataset.depths_m[-1]) / step) * step
    upper = math.ceil(np.max(dataset.elevations_m - dataset.depths_m[0]) / step) * step
    rl = np.arange(lower, upper + step / 2, step)
    profiles = np.full((len(dataset.ids), len(rl)), np.nan)
    for index, (elevation, values) in enumerate(
        zip(dataset.elevations_m, dataset.qc_profiles)
    ):
        source_rl = elevation - dataset.depths_m
        order = np.argsort(source_rl)
        inside = (rl >= source_rl.min()) & (rl <= source_rl.max())
        profiles[index, inside] = np.interp(
            rl[inside], source_rl[order], values[order]
        )
    return rl, profiles
