"""Visualise the three-layer spherical cone geometry / Schwarz / OriginQ output.

This script is deliberately post-processing only. It reads files already
produced by ``layer_schwarz_qpu_controller.py`` and never solves the PDE,
builds a quantum circuit, or connects to OriginQ.

Default products
----------------
* ``layer_overview_3d_<metric>.png``: actual spherical three-layer geometry,
  inward-oriented local frusta, geometric links, and realised handoffs.
* ``layer_unwrapped_<metric>.png``: azimuth/polar-angle unwrapped map with
  reciprocal side/patch relationships.
* ``geometry_patch_audit.png``: link-by-link donor/receiver side and patch map.
* ``schwarz_convergence.png``: interface residual by global time slab/iteration.
* ``visualization_summary.csv`` and ``visualization_report.html``.
* Optional execution GIF and one-cone internal concentration report.

Batch behaviour
---------------
When ``--output-dir`` points to a parent directory such as ``outputs/fig6``,
the script recursively discovers every completed case containing real-QPU
result files and visualises each case separately. When it points directly to a
single case directory, the original single-case behaviour is preserved.

Examples
--------
python src/layer_geometry_visualizer.py --output-dir outputs/fig6
python src/layer_geometry_visualizer.py --output-dir outputs/fig6 \
    --metric nuclear_total_flux --animate
python src/layer_geometry_visualizer.py --output-dir outputs/fig6/parameters_1 \
    --cone-id L1-C001 --phi-deg 45
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping
import argparse
import ast
import csv
import html
import json
import math
import os
import sys
import warnings

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, cm, colors
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import proj3d

SPECIES = ("free", "stationary", "mobile")
ROLES = ("target", "upper", "lower")
ROLE_Y = {"lower": 0.0, "target": 1.0, "upper": 2.0}
ROLE_LABEL = {"lower": "Lower layer", "target": "Target layer", "upper": "Upper layer"}
SOURCE_PRIORITY = ("qpu", "carleman", "nonlinear")
QPU_RESULT_MARKERS = (
    "cloud_report.json",
    "qpu_target_concentrations.csv",
    "qpu_nuclear_side_concentrations.csv",
)

AVAILABLE_METRICS = (
    "nuclear_free_mean",
    "nuclear_stationary_mean",
    "nuclear_mobile_mean",
    "nuclear_total_flux",
    "final_mass_free",
    "final_mass_stationary",
    "final_mass_mobile",
    "side_initial_mass_free",
    "side_initial_mass_stationary",
    "side_initial_mass_mobile",
    "membrane_input_mass_free",
    "membrane_input_mass_stationary",
    "membrane_input_mass_mobile",
    "incoming_transfer_mass_total",
)

METRIC_LABELS = {
    "nuclear_free_mean": "Nuclear free concentration",
    "nuclear_stationary_mean": "Nuclear stationary-bound concentration",
    "nuclear_mobile_mean": "Nuclear mobile-bound concentration",
    "nuclear_total_flux": "Total nuclear capture flux",
    "final_mass_free": "Final free mass",
    "final_mass_stationary": "Final stationary-bound mass",
    "final_mass_mobile": "Final mobile-bound mass",
    "side_initial_mass_free": "Neighbour side-input free mass",
    "side_initial_mass_stationary": "Neighbour side-input stationary mass",
    "side_initial_mass_mobile": "Neighbour side-input mobile mass",
    "membrane_input_mass_free": "Plasma-membrane free input mass",
    "membrane_input_mass_stationary": "Plasma-membrane stationary input mass",
    "membrane_input_mass_mobile": "Plasma-membrane mobile input mass",
    "incoming_transfer_mass_total": "Delivered neighbour mass",
}


@dataclass(frozen=True)
class ConeNode:
    cone_id: str
    role: str
    layer: int
    sector: int
    schedule_order: int
    azimuth_deg: float
    polar_deg: float
    position: np.ndarray
    axis: np.ndarray
    right: np.ndarray
    top: np.ndarray


@dataclass(frozen=True)
class GeometryLink:
    link_id: str
    edge_type: str
    node_a: str
    node_b: str
    distance: float
    weight: float
    a_side: str
    b_side: str
    a_local_phi_deg: float
    b_local_phi_deg: float
    a_patch_id: str
    b_patch_id: str
    reciprocal_ok: bool


@dataclass(frozen=True)
class Handoff:
    donor_id: str
    receiver_id: str
    link_id: str
    edge_type: str
    donor_patch_id: str
    receiver_patch_id: str
    time_slab: int
    iteration: int
    delivered: Mapping[str, float]

    @property
    def total(self) -> float:
        return float(sum(float(self.delivered.get(name, 0.0)) for name in SPECIES))


class Arrow3D(FancyArrowPatch):
    """Matplotlib 3-D arrow compatible with static figures and GIF frames."""

    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return float(np.min(zs))


# ---------------------------------------------------------------------------
# Generic loading helpers
# ---------------------------------------------------------------------------


def read_csv(path: Path, *, required: bool = True) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        if required:
            raise FileNotFoundError(f"Missing required CSV: {path}")
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def value_float(row: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    raw = row.get(key, "")
    if raw in (None, ""):
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def value_int(row: Mapping[str, Any], key: str, default: int = 0) -> int:
    return int(round(value_float(row, key, float(default))))


def parse_mapping(raw: Any) -> dict[str, float]:
    if isinstance(raw, Mapping):
        return {str(k): float(v) for k, v in raw.items()}
    text = str(raw or "").strip()
    if not text:
        return {}
    for loader in (json.loads, ast.literal_eval):
        try:
            result = loader(text)
            if isinstance(result, Mapping):
                return {str(k): float(v) for k, v in result.items()}
        except Exception:
            continue
    return {}


def load_nodes(output_dir: Path) -> dict[str, ConeNode]:
    nodes: dict[str, ConeNode] = {}
    for row in read_csv(output_dir / "geometry_nodes.csv"):
        node = ConeNode(
            cone_id=str(row["cone_id"]),
            role=str(row.get("role", "target")),
            layer=value_int(row, "layer", 1),
            sector=value_int(row, "sector", 1),
            schedule_order=value_int(row, "schedule_order", 0),
            azimuth_deg=value_float(row, "azimuth_deg"),
            polar_deg=value_float(row, "polar_deg", 90.0),
            position=np.asarray(
                [value_float(row, "x"), value_float(row, "y"), value_float(row, "z")],
                dtype=float,
            ),
            axis=np.asarray(
                [value_float(row, "axis_x"), value_float(row, "axis_y"), value_float(row, "axis_z")],
                dtype=float,
            ),
            right=np.asarray(
                [value_float(row, "right_x"), value_float(row, "right_y"), value_float(row, "right_z")],
                dtype=float,
            ),
            top=np.asarray(
                [value_float(row, "top_x"), value_float(row, "top_y"), value_float(row, "top_z")],
                dtype=float,
            ),
        )
        nodes[node.cone_id] = node
    if not nodes:
        raise ValueError("geometry_nodes.csv contains no cones")
    return nodes


def load_links(output_dir: Path) -> list[GeometryLink]:
    links: list[GeometryLink] = []
    for row in read_csv(output_dir / "geometry_links.csv"):
        links.append(
            GeometryLink(
                link_id=str(row["link_id"]),
                edge_type=str(row.get("edge_type", "unknown")),
                node_a=str(row["node_a"]),
                node_b=str(row["node_b"]),
                distance=value_float(row, "distance"),
                weight=value_float(row, "weight", 1.0),
                a_side=str(row.get("a_side", "")),
                b_side=str(row.get("b_side", "")),
                a_local_phi_deg=value_float(row, "a_local_phi_deg"),
                b_local_phi_deg=value_float(row, "b_local_phi_deg"),
                a_patch_id=str(row.get("a_patch_id", "")),
                b_patch_id=str(row.get("b_patch_id", "")),
                reciprocal_ok=bool(value_int(row, "opposite_patch_check", 0)),
            )
        )
    return links


def load_schedule(output_dir: Path) -> list[str]:
    rows = read_csv(output_dir / "geometry_schedule.csv")
    return [row["cone_id"] for row in sorted(rows, key=lambda row: value_int(row, "order"))]


def find_time_slabs(output_dir: Path) -> list[int]:
    values: list[int] = []
    for path in output_dir.glob("time_*"):
        if not path.is_dir():
            continue
        try:
            values.append(int(path.name.split("_", 1)[1]))
        except Exception:
            continue
    return sorted(values)


def resolve_time_slab(output_dir: Path, requested: str) -> int:
    slabs = find_time_slabs(output_dir)
    if not slabs:
        raise FileNotFoundError(f"No time_XXX output directories under {output_dir}")
    if requested.lower() == "latest":
        return slabs[-1]
    value = int(requested)
    if value not in slabs:
        raise ValueError(f"Requested time slab {value} does not exist; available: {slabs}")
    return value


def cone_output_dir(output_dir: Path, time_slab: int, cone_id: str) -> Path:
    path = output_dir / f"time_{time_slab:03d}" / "converged" / cone_id
    if path.exists():
        return path
    # Partial/in-progress run fallback: choose the highest available iteration.
    candidates = sorted(output_dir.glob(f"time_{time_slab:03d}/iter_*/**/{cone_id}/qpu_output"))
    if candidates:
        return candidates[-1]
    candidates = sorted(output_dir.glob(f"time_{time_slab:03d}/iter_*/**/*_{cone_id}/qpu_output"))
    if candidates:
        return candidates[-1]
    raise FileNotFoundError(f"No output found for cone {cone_id} in time slab {time_slab}")


def locate_detailed_output(output_dir: Path, time_slab: int, cone_id: str) -> Path:
    """Return a qpu_output directory when detailed CSV/JSON files are needed."""
    candidates = sorted(output_dir.glob(f"time_{time_slab:03d}/iter_*/**/*_{cone_id}/qpu_output"))
    if candidates:
        return candidates[-1]
    return cone_output_dir(output_dir, time_slab, cone_id)


def load_parameters(config: Path | None, output_dir: Path) -> dict[str, Any]:
    candidates = [config] if config else []
    candidates.extend(
        [
            output_dir / "resolved_parameters.json",
            output_dir.parent / "layer_geometry_parameters.json",
            Path("layer_geometry_parameters.json"),
        ]
    )
    for candidate in candidates:
        if candidate is not None and candidate.exists():
            return json.loads(candidate.read_text(encoding="utf-8"))
    return {}


# ---------------------------------------------------------------------------
# Metric extraction
# ---------------------------------------------------------------------------


def preferred_source(sources: Mapping[str, Any], requested: str) -> str | None:
    if requested != "auto":
        return requested if requested in sources else None
    for source in SOURCE_PRIORITY:
        payload = sources.get(source)
        if isinstance(payload, Mapping):
            # A smoke-test QPU entry may exist without fitted coefficients/statistics.
            usable = any(
                isinstance(payload.get(species), Mapping)
                and payload[species].get("area_weighted_mean") is not None
                for species in SPECIES
            )
            if usable or payload.get("total_nuclear_capture_flux") is not None:
                return source
    return None


def state_mass(path: Path, species: str, model: Mapping[str, Any]) -> float:
    with np.load(path, allow_pickle=False) as data:
        if species not in data:
            return float("nan")
        values = np.asarray(data[species], dtype=float).ravel()
        # Use exact cell volumes when a model module is not imported. For the compact
        # finite-volume grid, these can be reconstructed from z/ring/sector indices.
        if all(key in data for key in ("z_index", "ring_index", "sector_index", "ring_counts")):
            volumes = compact_cell_volumes(data, model)
            if volumes.shape == values.shape:
                return float(np.nansum(values * volumes))
        return float(np.nansum(values))


def model_grid_indices(model: Mapping[str, Any]) -> dict[str, np.ndarray]:
    nz = int(model.get("nz", 1))
    nr = int(model.get("nr", 1))
    counts = np.asarray(
        model.get("azimuth_counts_by_ring", [int(model.get("nphi", 1))] * nr),
        dtype=int,
    )
    if counts.size != nr:
        raise ValueError("model.azimuth_counts_by_ring length must equal nr")
    z_index: list[int] = []
    ring_index: list[int] = []
    sector_index: list[int] = []
    for iz in range(nz):
        for ir, count in enumerate(counts):
            for ip in range(int(count)):
                z_index.append(iz); ring_index.append(ir); sector_index.append(ip)
    return {
        "z_index": np.asarray(z_index, dtype=int),
        "ring_index": np.asarray(ring_index, dtype=int),
        "sector_index": np.asarray(sector_index, dtype=int),
        "ring_counts": counts,
    }


def csv_species_mass(path: Path, species: str, model: Mapping[str, Any]) -> float:
    if not path.exists() or path.stat().st_size == 0:
        return 0.0
    rows = read_csv(path, required=False)
    if not rows:
        return 0.0
    # Prefer an explicit mass/integral column when present.
    for key in (f"{species}_mass", f"mass_{species}", f"{species}_volume_integral"):
        if key in rows[0]:
            return float(sum(value_float(row, key) for row in rows))
    candidate_keys = [
        species,
        f"initial_{species}",
        f"side_initial_{species}",
        f"{species}_initial",
        f"{species}_concentration",
    ]
    key = next((name for name in candidate_keys if name in rows[0]), None)
    if key is None:
        return 0.0
    values = np.asarray([value_float(row, key) for row in rows], dtype=float)
    if "area" in rows[0]:
        areas = np.asarray([value_float(row, "area") for row in rows], dtype=float)
        return float(np.nansum(values * areas))
    if "cell" in rows[0]:
        grid = model_grid_indices(model)
        volumes_all = compact_cell_volumes(grid, model)
        cells = np.asarray([value_int(row, "cell", -1) for row in rows], dtype=int)
        valid = (cells >= 0) & (cells < volumes_all.size)
        return float(np.nansum(values[valid] * volumes_all[cells[valid]]))
    return float(np.nansum(values))


def compact_cell_volumes(data: Mapping[str, Any], model: Mapping[str, Any]) -> np.ndarray:
    z_index = np.asarray(data["z_index"], dtype=int).ravel()
    ring_index = np.asarray(data["ring_index"], dtype=int).ravel()
    counts = np.asarray(data["ring_counts"], dtype=int).ravel()
    nz = int(model.get("nz", int(np.max(z_index)) + 1 if z_index.size else 1))
    nr = int(model.get("nr", len(counts) if len(counts) else int(np.max(ring_index)) + 1))
    rn = float(model.get("radius_nucleus", 0.35))
    rm = float(model.get("radius_membrane", 0.70))
    z_edges = np.linspace(0.0, 1.0, nz + 1)
    rho_edges = np.linspace(0.0, 1.0, nr + 1)
    result = np.zeros(z_index.size, dtype=float)
    for i, (iz, ir) in enumerate(zip(z_index, ring_index, strict=True)):
        z0, z1 = z_edges[iz], z_edges[iz + 1]
        r0z0 = (rn + (rm - rn) * z0) * rho_edges[ir]
        r1z0 = (rn + (rm - rn) * z0) * rho_edges[ir + 1]
        r0z1 = (rn + (rm - rn) * z1) * rho_edges[ir]
        r1z1 = (rn + (rm - rn) * z1) * rho_edges[ir + 1]
        area0 = math.pi * (r1z0**2 - r0z0**2)
        area1 = math.pi * (r1z1**2 - r0z1**2)
        frustum_volume = (z1 - z0) * (area0 + math.sqrt(area0 * area1) + area1) / 3.0
        result[i] = frustum_volume / max(int(counts[ir]), 1)
    return result


def load_handoffs(output_dir: Path, time_slab: int | None = None) -> list[Handoff]:
    rows = read_csv(output_dir / "handoff_log.csv", required=False)
    result: list[Handoff] = []
    for row in rows:
        slab = value_int(row, "time_slab", 0)
        if time_slab is not None and slab != time_slab:
            continue
        result.append(
            Handoff(
                donor_id=str(row.get("donor_id", "")),
                receiver_id=str(row.get("receiver_id", "")),
                link_id=str(row.get("link_id", "")),
                edge_type=str(row.get("edge_type", "unknown")),
                donor_patch_id=str(row.get("donor_patch_id", "")),
                receiver_patch_id=str(row.get("receiver_patch_id", "")),
                time_slab=slab,
                iteration=value_int(row, "schwarz_iteration", value_int(row, "iteration", 0)),
                delivered=parse_mapping(row.get("delivered_mass", "")),
            )
        )
    return result


def summarize_cones(
    output_dir: Path,
    nodes: Mapping[str, ConeNode],
    time_slab: int,
    source_request: str,
    model: Mapping[str, Any],
    handoffs: Iterable[Handoff],
) -> dict[str, dict[str, Any]]:
    incoming = {cone_id: 0.0 for cone_id in nodes}
    for item in handoffs:
        if item.receiver_id in incoming:
            incoming[item.receiver_id] += item.total

    summary: dict[str, dict[str, Any]] = {}
    for cone_id, node in nodes.items():
        row: dict[str, Any] = {
            "cone_id": cone_id,
            "role": node.role,
            "layer": node.layer,
            "sector": node.sector,
            "azimuth_deg": node.azimuth_deg,
            "polar_deg": node.polar_deg,
            "incoming_transfer_mass_total": incoming.get(cone_id, 0.0),
            "available": False,
            "nuclear_source": "",
        }
        try:
            cone_dir = cone_output_dir(output_dir, time_slab, cone_id)
        except FileNotFoundError:
            summary[cone_id] = row
            continue
        row["available"] = True
        nuclear_path = cone_dir / "nuclear_output.json"
        if nuclear_path.exists():
            payload = json.loads(nuclear_path.read_text(encoding="utf-8"))
            sources = payload.get("nuclear_face", {}).get("sources", {})
            source = preferred_source(sources, source_request)
            row["nuclear_source"] = source or ""
            source_data = sources.get(source, {}) if source else {}
            for species in SPECIES:
                row[f"nuclear_{species}_mean"] = float(
                    source_data.get(species, {}).get("area_weighted_mean", np.nan)
                )
            row["nuclear_total_flux"] = float(
                source_data.get("total_nuclear_capture_flux", np.nan)
            )
        state_path = cone_dir / "final_state.npz"
        if state_path.exists():
            for species in SPECIES:
                row[f"final_mass_{species}"] = state_mass(state_path, species, model)
        detailed = locate_detailed_output(output_dir, time_slab, cone_id)
        for species in SPECIES:
            row[f"side_initial_mass_{species}"] = csv_species_mass(
                detailed / "side_initial_input.csv", species, model
            )
            row[f"membrane_input_mass_{species}"] = csv_species_mass(
                detailed / "membrane_input.csv", species, model
            )
        summary[cone_id] = row
    return summary


def metric_values(
    nodes: Mapping[str, ConeNode],
    summary: Mapping[str, Mapping[str, Any]],
    metric: str,
) -> dict[str, float]:
    if metric not in AVAILABLE_METRICS:
        raise ValueError(f"Unknown metric {metric}; choices: {', '.join(AVAILABLE_METRICS)}")
    return {cone_id: float(summary.get(cone_id, {}).get(metric, np.nan)) for cone_id in nodes}


def finite_limits(values: Iterable[float]) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return 0.0, 1.0
    lower, upper = float(np.min(array)), float(np.max(array))
    if math.isclose(lower, upper, rel_tol=1e-12, abs_tol=1e-15):
        pad = max(abs(lower) * 0.05, 1e-12)
        return lower - pad, upper + pad
    return lower, upper


# ---------------------------------------------------------------------------
# Geometry drawing
# ---------------------------------------------------------------------------


def normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-15 else vector


def frustum_faces(
    node: ConeNode,
    *,
    length: float,
    radius_nuclear: float,
    radius_membrane: float,
    segments: int = 12,
) -> list[list[np.ndarray]]:
    # geometry_nodes position is on the cell surface; axis points toward nucleus.
    membrane_center = node.position
    nuclear_center = node.position + length * normalize(node.axis)
    right = normalize(node.right)
    top = normalize(node.top)
    outer: list[np.ndarray] = []
    inner: list[np.ndarray] = []
    for i in range(segments):
        angle = 2.0 * math.pi * i / segments
        radial = math.cos(angle) * right + math.sin(angle) * top
        outer.append(membrane_center + radius_membrane * radial)
        inner.append(nuclear_center + radius_nuclear * radial)
    faces: list[list[np.ndarray]] = []
    for i in range(segments):
        j = (i + 1) % segments
        faces.append([outer[i], outer[j], inner[j], inner[i]])
    faces.append(outer)
    faces.append(list(reversed(inner)))
    return faces


def set_equal_3d(ax, points: np.ndarray, padding: float = 0.15) -> None:
    if points.size == 0:
        return
    lower = np.min(points, axis=0)
    upper = np.max(points, axis=0)
    center = 0.5 * (lower + upper)
    radius = 0.5 * max(float(np.max(upper - lower)), 1.0) * (1.0 + padding)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))


def representative_spacing(nodes: Mapping[str, ConeNode]) -> float:
    positions = list(nodes.values())
    distances: list[float] = []
    for i, a in enumerate(positions):
        for b in positions[i + 1 :]:
            if a.layer != b.layer:
                continue
            distance = float(np.linalg.norm(a.position - b.position))
            if distance > 1e-10:
                distances.append(distance)
    return float(np.median(distances)) if distances else 1.0


def draw_sphere_guides(ax, nodes: Mapping[str, ConeNode]) -> None:
    radius = float(np.median([np.linalg.norm(node.position) for node in nodes.values()]))
    u = np.linspace(0, 2 * math.pi, 72)
    for polar in sorted({round(node.polar_deg, 8) for node in nodes.values()}):
        theta = math.radians(polar)
        rxy = radius * math.sin(theta)
        z = radius * math.cos(theta)
        ax.plot(rxy * np.cos(u), rxy * np.sin(u), np.full_like(u, z), alpha=0.20, linewidth=0.8)
    # Sparse meridians provide context without hiding the local cones.
    theta = np.linspace(0, math.pi, 50)
    for azimuth in np.linspace(0, 2 * math.pi, 8, endpoint=False):
        ax.plot(
            radius * np.sin(theta) * math.cos(azimuth),
            radius * np.sin(theta) * math.sin(azimuth),
            radius * np.cos(theta),
            alpha=0.10,
            linewidth=0.55,
        )


def handoff_linewidths(handoffs: Iterable[Handoff]) -> dict[tuple[str, str, int], float]:
    items = list(handoffs)
    maximum = max((item.total for item in items), default=0.0)
    result: dict[tuple[str, str, int], float] = {}
    for item in items:
        width = 0.8 if maximum <= 0 else 0.7 + 3.4 * math.sqrt(item.total / maximum)
        result[(item.donor_id, item.receiver_id, item.iteration)] = width
    return result


def plot_overview_3d(
    output_path: Path,
    nodes: Mapping[str, ConeNode],
    links: Iterable[GeometryLink],
    handoffs: Iterable[Handoff],
    values: Mapping[str, float],
    metric: str,
    *,
    labels: bool,
    show: bool,
) -> None:
    figure = plt.figure(figsize=(13, 11), constrained_layout=True)
    ax = figure.add_subplot(111, projection="3d")
    lower, upper = finite_limits(values.values())
    norm = colors.Normalize(lower, upper)
    cmap = plt.get_cmap("viridis")
    spacing = representative_spacing(nodes)
    length = 0.52 * spacing
    r_large = 0.18 * spacing
    r_small = 0.09 * spacing

    draw_sphere_guides(ax, nodes)
    for link in links:
        if link.node_a not in nodes or link.node_b not in nodes:
            continue
        p0, p1 = nodes[link.node_a].position, nodes[link.node_b].position
        ax.plot(*zip(p0, p1), linewidth=0.6 + 0.5 * link.weight, alpha=0.22, color="black")

    for cone_id, node in nodes.items():
        value = values.get(cone_id, np.nan)
        face_color = cmap(norm(value)) if np.isfinite(value) else (0.65, 0.65, 0.65, 0.65)
        collection = Poly3DCollection(
            frustum_faces(
                node,
                length=length,
                radius_nuclear=r_small,
                radius_membrane=r_large,
            ),
            facecolors=[face_color],
            edgecolors="black",
            linewidths=0.35,
            alpha=0.88,
        )
        ax.add_collection3d(collection)
        if labels:
            location = node.position - 0.05 * spacing * normalize(node.axis)
            ax.text(*location, cone_id, fontsize=6.5, ha="center")

    widths = handoff_linewidths(handoffs)
    for item in handoffs:
        if item.donor_id not in nodes or item.receiver_id not in nodes:
            continue
        p0 = nodes[item.donor_id].position
        p1 = nodes[item.receiver_id].position
        direction = p1 - p0
        start = p0 + 0.20 * direction
        end = p0 + 0.82 * direction
        arrow = Arrow3D(
            [start[0], end[0]],
            [start[1], end[1]],
            [start[2], end[2]],
            mutation_scale=12,
            linewidth=widths[(item.donor_id, item.receiver_id, item.iteration)],
            arrowstyle="-|>",
            color="black",
            alpha=0.62,
        )
        ax.add_artist(arrow)

    points = np.asarray([node.position for node in nodes.values()], dtype=float)
    set_equal_3d(ax, points, padding=0.30)
    ax.set_xlabel("Global x")
    ax.set_ylabel("Global y")
    ax.set_zlabel("Global z")
    ax.set_title(f"Three-layer spherical cone geometry — {METRIC_LABELS[metric]}")
    ax.view_init(elev=24, azim=-55)
    scalar = cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar.set_array([])
    figure.colorbar(scalar, ax=ax, shrink=0.68, pad=0.06, label=METRIC_LABELS[metric])
    figure.savefig(output_path, dpi=190)
    if show:
        plt.show()
    plt.close(figure)


def shortest_unwrapped(origin: float, target: float) -> float:
    return origin + math.degrees(
        math.atan2(math.sin(math.radians(target - origin)), math.cos(math.radians(target - origin)))
    )


def plot_unwrapped(
    output_path: Path,
    nodes: Mapping[str, ConeNode],
    links: Iterable[GeometryLink],
    handoffs: Iterable[Handoff],
    values: Mapping[str, float],
    metric: str,
    *,
    labels: bool,
    show: bool,
) -> None:
    figure, ax = plt.subplots(figsize=(14, 7), constrained_layout=True)
    lower, upper = finite_limits(values.values())
    norm = colors.Normalize(lower, upper)
    cmap = plt.get_cmap("viridis")

    for link in links:
        if link.node_a not in nodes or link.node_b not in nodes:
            continue
        a, b = nodes[link.node_a], nodes[link.node_b]
        bx = shortest_unwrapped(a.azimuth_deg, b.azimuth_deg)
        ax.plot(
            [a.azimuth_deg, bx],
            [ROLE_Y[a.role], ROLE_Y[b.role]],
            color="black",
            alpha=0.18,
            linewidth=0.75 + 0.45 * link.weight,
        )

    widths = handoff_linewidths(handoffs)
    for item in handoffs:
        if item.donor_id not in nodes or item.receiver_id not in nodes:
            continue
        a, b = nodes[item.donor_id], nodes[item.receiver_id]
        bx = shortest_unwrapped(a.azimuth_deg, b.azimuth_deg)
        ax.annotate(
            "",
            xy=(bx, ROLE_Y[b.role]),
            xytext=(a.azimuth_deg, ROLE_Y[a.role]),
            arrowprops={
                "arrowstyle": "-|>",
                "color": "black",
                "alpha": 0.62,
                "linewidth": widths[(item.donor_id, item.receiver_id, item.iteration)],
            },
        )

    ordered = sorted(nodes.values(), key=lambda node: node.schedule_order)
    for role in ROLES:
        group = [node for node in ordered if node.role == role]
        x = [node.azimuth_deg for node in group]
        y = [ROLE_Y[role]] * len(group)
        c = [values.get(node.cone_id, np.nan) for node in group]
        ax.scatter(x, y, c=c, cmap=cmap, norm=norm, s=190, edgecolors="black", linewidths=0.6, zorder=4)
        if labels:
            for node in group:
                ax.text(node.azimuth_deg, ROLE_Y[role] + 0.11, node.cone_id, fontsize=7, ha="center")

    ax.set_xlim(-5.0, 365.0)
    ax.set_ylim(-0.45, 2.45)
    ax.set_xticks(np.arange(0, 361, 45))
    ax.set_yticks([ROLE_Y[role] for role in ROLES])
    ax.set_yticklabels([ROLE_LABEL[role] for role in ROLES])
    ax.set_xlabel("Global azimuth (degrees)")
    ax.set_title(f"Unwrapped staggered layers — {METRIC_LABELS[metric]}")
    ax.grid(axis="x", alpha=0.18)
    scalar = cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar.set_array([])
    figure.colorbar(scalar, ax=ax, pad=0.02, label=METRIC_LABELS[metric])
    figure.savefig(output_path, dpi=190)
    if show:
        plt.show()
    plt.close(figure)


def plot_patch_audit(output_path: Path, links: Iterable[GeometryLink], *, show: bool) -> None:
    rows = list(links)
    if not rows:
        return
    rows.sort(key=lambda item: (item.edge_type, item.link_id))
    figure, ax = plt.subplots(figsize=(14, max(6, 0.32 * len(rows))), constrained_layout=True)
    y = np.arange(len(rows))
    for index, link in enumerate(rows):
        a = link.a_local_phi_deg % 360.0
        b = link.b_local_phi_deg % 360.0
        # unwrap b around the expected opposite direction from a.
        expected = (a + 180.0) % 360.0
        b_plot = expected + math.degrees(math.atan2(math.sin(math.radians(b - expected)), math.cos(math.radians(b - expected))))
        ax.plot([a, b_plot], [index, index], color="black", alpha=0.35, linewidth=1.0)
        ax.scatter([a], [index], marker="o", s=42)
        ax.scatter([b_plot], [index], marker="s", s=42)
        ax.text(a, index + 0.16, f"{link.node_a} {link.a_side}/{link.a_patch_id}", fontsize=7, ha="center")
        ax.text(b_plot, index - 0.22, f"{link.node_b} {link.b_side}/{link.b_patch_id}", fontsize=7, ha="center")
        if not link.reciprocal_ok:
            ax.text(360, index, "PATCH ERROR", fontsize=8, va="center")
    ax.axvline(180.0, linewidth=0.8, alpha=0.25)
    ax.set_xlim(-20, 380)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{link.link_id} · {link.edge_type}" for link in rows], fontsize=7)
    ax.set_xlabel("Local side direction (degrees); circle=side A, square=side B")
    ax.set_title("Geometry-side and reciprocal-patch audit")
    ax.grid(axis="x", alpha=0.18)
    figure.savefig(output_path, dpi=185)
    if show:
        plt.show()
    plt.close(figure)


def plot_schwarz_convergence(output_path: Path, output_dir: Path, *, show: bool) -> bool:
    rows = read_csv(output_dir / "schwarz_iterations.csv", required=False)
    finite_rows = [row for row in rows if math.isfinite(value_float(row, "interface_relative_residual", np.inf))]
    if not finite_rows:
        return False
    figure, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    slabs = sorted({value_int(row, "time_slab") for row in finite_rows})
    for slab in slabs:
        selected = [row for row in finite_rows if value_int(row, "time_slab") == slab]
        selected.sort(key=lambda row: value_int(row, "iteration"))
        ax.plot(
            [value_int(row, "iteration") + 1 for row in selected],
            [value_float(row, "interface_relative_residual") for row in selected],
            marker="o",
            label=f"Time slab {slab}",
        )
        tolerance = value_float(selected[-1], "tolerance", np.nan)
        if np.isfinite(tolerance):
            ax.axhline(tolerance, linestyle="--", linewidth=0.9, alpha=0.55)
    ax.set_yscale("log")
    ax.set_xlabel("Schwarz iteration")
    ax.set_ylabel("Relative interface residual")
    ax.set_title("Schwarz interface convergence")
    ax.grid(alpha=0.22)
    if len(slabs) > 1:
        ax.legend()
    figure.savefig(output_path, dpi=190)
    if show:
        plt.show()
    plt.close(figure)
    return True


# ---------------------------------------------------------------------------
# Animation
# ---------------------------------------------------------------------------


def load_execution_rows(output_dir: Path, time_slab: int) -> list[dict[str, str]]:
    rows = read_csv(output_dir / "run_summary.csv", required=False)
    selected = [row for row in rows if value_int(row, "time_slab", 0) == time_slab]
    selected.sort(key=lambda row: (value_int(row, "iteration"), value_int(row, "schedule_order")))
    return selected


def animate_execution(
    output_path: Path,
    nodes: Mapping[str, ConeNode],
    values: Mapping[str, float],
    execution_rows: list[dict[str, str]],
    handoffs: Iterable[Handoff],
    metric: str,
    *,
    fps: float,
    labels: bool,
    show: bool,
) -> bool:
    if not execution_rows:
        return False
    lower, upper = finite_limits(values.values())
    norm = colors.Normalize(lower, upper)
    cmap = plt.get_cmap("viridis")
    figure, ax = plt.subplots(figsize=(13, 7), constrained_layout=True)

    all_handoffs = list(handoffs)

    def prepare_axes() -> None:
        ax.set_xlim(-5, 365)
        ax.set_ylim(-0.45, 2.55)
        ax.set_xticks(np.arange(0, 361, 45))
        ax.set_yticks([ROLE_Y[role] for role in ROLES])
        ax.set_yticklabels([ROLE_LABEL[role] for role in ROLES])
        ax.set_xlabel("Global azimuth (degrees)")
        ax.grid(axis="x", alpha=0.18)

    def update(frame: int):
        ax.clear()
        prepare_axes()
        active_rows = execution_rows[: frame + 1]
        active_ids = {row.get("cone_id", "") for row in active_rows}
        for role in ROLES:
            group = [node for node in nodes.values() if node.role == role]
            ax.scatter(
                [node.azimuth_deg for node in group],
                [ROLE_Y[role]] * len(group),
                c=[values.get(node.cone_id, np.nan) if node.cone_id in active_ids else np.nan for node in group],
                cmap=cmap,
                norm=norm,
                s=[190 if node.cone_id in active_ids else 80 for node in group],
                edgecolors="black",
                linewidths=0.5,
                alpha=[0.95 if node.cone_id in active_ids else 0.18 for node in group],
            )
            if labels:
                for node in group:
                    if node.cone_id in active_ids:
                        ax.text(node.azimuth_deg, ROLE_Y[role] + 0.12, node.cone_id, fontsize=7, ha="center")
        current_row = execution_rows[frame]
        current_iteration = value_int(current_row, "iteration")
        executed_pairs = {(row.get("cone_id", ""), value_int(row, "iteration")) for row in active_rows}
        for item in all_handoffs:
            if item.iteration > current_iteration:
                continue
            if (item.receiver_id, item.iteration) not in executed_pairs:
                continue
            if item.donor_id not in nodes or item.receiver_id not in nodes:
                continue
            a, b = nodes[item.donor_id], nodes[item.receiver_id]
            bx = shortest_unwrapped(a.azimuth_deg, b.azimuth_deg)
            ax.annotate(
                "",
                xy=(bx, ROLE_Y[b.role]),
                xytext=(a.azimuth_deg, ROLE_Y[a.role]),
                arrowprops={"arrowstyle": "-|>", "color": "black", "alpha": 0.55, "linewidth": 1.2},
            )
        ax.text(
            0.01,
            0.97,
            f"Step {frame+1}/{len(execution_rows)} | iteration {current_iteration} | "
            f"running {current_row.get('cone_id','')} | incoming: {current_row.get('incoming_neighbors','') or 'none'}",
            transform=ax.transAxes,
            va="top",
        )
        ax.set_title(f"Geometry-aware multiplicative Schwarz execution — {METRIC_LABELS[metric]}")
        return []

    movie = animation.FuncAnimation(
        figure,
        update,
        frames=len(execution_rows),
        interval=1000.0 / max(fps, 0.1),
        repeat=True,
        blit=False,
    )
    scalar = cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar.set_array([])
    figure.colorbar(scalar, ax=ax, pad=0.02, label=METRIC_LABELS[metric])
    movie.save(output_path, writer=animation.PillowWriter(fps=max(fps, 0.1)), dpi=130)
    if show:
        plt.show()
    plt.close(figure)
    return True


# ---------------------------------------------------------------------------
# Compact local-cone reconstruction / report
# ---------------------------------------------------------------------------


def compact_coordinates(data: Mapping[str, Any], model: Mapping[str, Any]) -> dict[str, np.ndarray]:
    z_index = np.asarray(data["z_index"], dtype=int).ravel()
    ring_index = np.asarray(data["ring_index"], dtype=int).ravel()
    sector_index = np.asarray(data["sector_index"], dtype=int).ravel()
    ring_counts = np.asarray(data["ring_counts"], dtype=int).ravel()
    nz = int(model.get("nz", int(np.max(z_index)) + 1))
    nr = int(model.get("nr", len(ring_counts)))
    rn = float(model.get("radius_nucleus", 0.35))
    rm = float(model.get("radius_membrane", 0.70))
    z = (z_index + 0.5) / nz
    rho_metric = (ring_index + 0.5) / nr
    phi = 2.0 * math.pi * (sector_index + 0.5) / ring_counts[ring_index]
    radius = rn + (rm - rn) * z
    display_rho = np.where(ring_index == 0, 0.0, rho_metric)
    physical_r = radius * display_rho
    x = physical_r * np.cos(phi)
    y = physical_r * np.sin(phi)
    return {
        "z": z,
        "rho": display_rho,
        "rho_metric": rho_metric,
        "phi": phi,
        "x": x,
        "y": y,
        "r": physical_r,
        "z_index": z_index,
        "ring_index": ring_index,
        "sector_index": sector_index,
        "ring_counts": ring_counts,
        "nz": np.asarray(nz),
        "nr": np.asarray(nr),
    }


def nearest_phi_slice(values: np.ndarray, coords: Mapping[str, np.ndarray], requested_deg: float) -> np.ndarray:
    nz = int(coords["nz"])
    nr = int(coords["nr"])
    result = np.full((nr, nz), np.nan, dtype=float)
    requested = math.radians(requested_deg) % (2.0 * math.pi)
    for iz in range(nz):
        for ir in range(nr):
            cells = np.flatnonzero((coords["z_index"] == iz) & (coords["ring_index"] == ir))
            if cells.size == 0:
                continue
            if ir == 0:
                selected = int(cells[0])
            else:
                distance = np.abs(
                    np.arctan2(
                        np.sin(coords["phi"][cells] - requested),
                        np.cos(coords["phi"][cells] - requested),
                    )
                )
                selected = int(cells[int(np.argmin(distance))])
            result[ir, iz] = float(values[selected])
    return result


def plot_internal_cone(
    output_path: Path,
    state_path: Path,
    model: Mapping[str, Any],
    cone_id: str,
    *,
    phi_deg: float,
    show: bool,
) -> None:
    with np.load(state_path, allow_pickle=False) as data:
        missing = [name for name in (*SPECIES, "z_index", "ring_index", "sector_index", "ring_counts") if name not in data]
        if missing:
            raise ValueError(f"{state_path} lacks compact-grid keys: {missing}")
        arrays = {name: np.asarray(data[name], dtype=float).ravel() for name in SPECIES}
        coords = compact_coordinates(data, model)

    figure = plt.figure(figsize=(16, 10.5), constrained_layout=True)
    labels = {
        "free": "Free",
        "stationary": "Stationary-bound",
        "mobile": "Mobile-bound",
    }
    for panel, species in enumerate(SPECIES, start=1):
        ax = figure.add_subplot(2, 3, panel, projection="3d")
        lower, upper = finite_limits(arrays[species])
        scatter = ax.scatter(
            coords["z"],
            coords["x"],
            coords["y"],
            c=arrays[species],
            cmap="viridis",
            norm=colors.Normalize(lower, upper),
            s=22,
            alpha=0.88,
            linewidths=0.0,
        )
        ax.set_xlabel("z: nucleus → membrane")
        ax.set_ylabel("local x")
        ax.set_zlabel("local y")
        ax.set_title(labels[species])
        ax.view_init(elev=23, azim=-60)
        figure.colorbar(scatter, ax=ax, shrink=0.62, pad=0.06)

    ax_slice = figure.add_subplot(2, 3, 4)
    slice_values = nearest_phi_slice(arrays["free"], coords, phi_deg)
    image = ax_slice.imshow(
        slice_values,
        origin="lower",
        aspect="auto",
        extent=(0.0, 1.0, 0.0, 1.0),
        cmap="viridis",
    )
    ax_slice.set_xlabel("z: nucleus → membrane")
    ax_slice.set_ylabel("normalised radius ρ")
    ax_slice.set_title(f"Free z–ρ slice near φ={phi_deg:.1f}°")
    figure.colorbar(image, ax=ax_slice)

    # Nuclear face polar map from the compact z=0 layer.
    ax_nuclear = figure.add_subplot(2, 3, 5, projection="polar")
    nuclear_cells = np.flatnonzero(coords["z_index"] == 0)
    polar = ax_nuclear.scatter(
        coords["phi"][nuclear_cells],
        coords["rho_metric"][nuclear_cells],
        c=arrays["free"][nuclear_cells],
        cmap="viridis",
        s=70,
        edgecolors="black",
        linewidths=0.25,
    )
    ax_nuclear.set_ylim(0.0, 1.0)
    ax_nuclear.set_title("Nuclear-face free concentration")
    figure.colorbar(polar, ax=ax_nuclear, pad=0.12)

    ax_profile = figure.add_subplot(2, 3, 6)
    volumes = compact_cell_volumes(
        {
            "z_index": coords["z_index"],
            "ring_index": coords["ring_index"],
            "ring_counts": coords["ring_counts"],
        },
        model,
    )
    z_axis: list[float] = []
    for iz in range(int(coords["nz"])):
        cells = np.flatnonzero(coords["z_index"] == iz)
        weights = volumes[cells]
        z_axis.append(float(np.mean(coords["z"][cells])))
        for species in SPECIES:
            denominator = max(float(np.sum(weights)), 1e-15)
            mean = float(np.sum(arrays[species][cells] * weights) / denominator)
            ax_profile.plot([], []) if False else None
            # Store temporary values on the axis for one pass below.
            line_values = getattr(ax_profile, f"_{species}_values", [])
            line_values.append(mean)
            setattr(ax_profile, f"_{species}_values", line_values)
    for species in SPECIES:
        ax_profile.plot(z_axis, getattr(ax_profile, f"_{species}_values"), marker="o", label=labels[species])
    ax_profile.set_xlabel("z: nucleus → membrane")
    ax_profile.set_ylabel("Volume-weighted mean concentration")
    ax_profile.set_title("Axial concentration profiles")
    ax_profile.grid(alpha=0.22)
    ax_profile.legend(fontsize=8)

    figure.suptitle(f"Compact true-3-D local cone — {cone_id}")
    figure.savefig(output_path, dpi=190)
    if show:
        plt.show()
    plt.close(figure)


# ---------------------------------------------------------------------------
# Tabular / HTML outputs
# ---------------------------------------------------------------------------


def json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_summary_csv(path: Path, summary: Mapping[str, Mapping[str, Any]]) -> None:
    rows = [dict(row) for _, row in sorted(summary.items())]
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                keys.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_html_report(
    path: Path,
    *,
    metric: str,
    time_slab: int,
    source: str,
    summary: Mapping[str, Mapping[str, Any]],
    images: Iterable[Path],
) -> None:
    available = [row for row in summary.values() if row.get("available")]
    values = [float(row.get(metric, np.nan)) for row in available]
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    stats = {
        "Cones with output": len(available),
        "Metric minimum": float(np.min(finite)) if finite.size else float("nan"),
        "Metric mean": float(np.mean(finite)) if finite.size else float("nan"),
        "Metric maximum": float(np.max(finite)) if finite.size else float("nan"),
    }
    image_html = "\n".join(
        f'<section><h2>{html.escape(image.stem.replace("_", " ").title())}</h2><img src="{html.escape(image.name)}" alt="{html.escape(image.stem)}"></section>'
        for image in images
        if image.exists()
    )
    stats_html = "".join(
        f"<div class='metric'><span>{html.escape(label)}</span><strong>{value:.6g}</strong></div>"
        if isinstance(value, float)
        else f"<div class='metric'><span>{html.escape(label)}</span><strong>{value}</strong></div>"
        for label, value in stats.items()
    )
    document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Layer geometry visualization report</title>
<style>
body{{font-family:Arial,"Microsoft YaHei",sans-serif;margin:0;background:#f4f6f8;color:#17202a}}
main{{max-width:1200px;margin:auto;padding:24px}}h1{{font-size:26px}}h2{{font-size:19px;margin-top:8px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px;margin:18px 0}}
.metric,section{{background:white;border:1px solid #dfe4e8;border-radius:12px;padding:14px}}
.metric span{{display:block;color:#5f6b76;font-size:13px}}.metric strong{{font-size:21px}}
section{{margin:18px 0}}img{{width:100%;height:auto;border-radius:8px}}
.meta{{color:#5f6b76}}code{{background:#eef1f4;padding:2px 5px;border-radius:4px}}
</style></head><body><main>
<h1>三层球面截锥融合后可视化</h1>
<p class="meta">time slab: <code>{time_slab}</code> · metric: <code>{html.escape(metric)}</code> · source: <code>{html.escape(source)}</code></p>
<div class="grid">{stats_html}</div>{image_html}
</main></body></html>"""
    path.write_text(document, encoding="utf-8")


# ---------------------------------------------------------------------------
# Result discovery / batch reporting
# ---------------------------------------------------------------------------


def is_result_directory(path: Path) -> bool:
    """Return True for a controller output root with at least one time slab."""
    return (
        path.is_dir()
        and (path / "geometry_nodes.csv").is_file()
        and (path / "geometry_links.csv").is_file()
        and bool(find_time_slabs(path))
    )


def discover_result_directories(root: Path) -> list[Path]:
    """Discover complete or partial case roots below *root*.

    ``geometry_nodes.csv`` is emitted once at the case root by the Schwarz
    controller, so it is a stable marker that does not confuse per-cone
    ``qpu_output`` directories with independent cases.
    """
    if is_result_directory(root):
        return [root]
    if not root.exists():
        raise FileNotFoundError(root)
    candidates = {
        marker.parent.resolve()
        for marker in root.rglob("geometry_nodes.csv")
        if is_result_directory(marker.parent)
    }
    return sorted(candidates, key=lambda path: str(path).lower())


def find_qpu_output_directories(output_dir: Path) -> list[Path]:
    """Return unique per-cone directories containing real-QPU result markers."""
    directories: set[Path] = set()
    for marker_name in QPU_RESULT_MARKERS:
        for marker in output_dir.rglob(marker_name):
            if marker.is_file():
                directories.add(marker.parent.resolve())
    return sorted(directories, key=lambda path: str(path).lower())


def case_label(root: Path, output_dir: Path) -> str:
    try:
        relative = output_dir.relative_to(root)
        return relative.as_posix() if str(relative) != "." else output_dir.name
    except ValueError:
        return output_dir.name


def write_batch_report(
    report_dir: Path,
    *,
    search_root: Path,
    rows: list[dict[str, Any]],
) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "case",
        "status",
        "output_dir",
        "time_slab",
        "qpu_output_directories",
        "cones_with_selected_source",
        "visualization_dir",
        "report",
        "error",
    )
    csv_path = report_dir / "visualization_batch_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in fieldnames} for row in rows)

    json_path = report_dir / "visualization_batch_summary.json"
    json_path.write_text(
        json.dumps(
            {
                "search_root": str(search_root),
                "cases": json_safe(rows),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    table_rows: list[str] = []
    for row in rows:
        report = Path(str(row.get("report", ""))) if row.get("report") else None
        if report is not None and report.exists():
            href = Path(os.path.relpath(report, report_dir)).as_posix()
            report_cell = f'<a href="{html.escape(href)}">Open report</a>'
        else:
            report_cell = "—"
        table_rows.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('case', '')))}</td>"
            f"<td>{html.escape(str(row.get('status', '')))}</td>"
            f"<td>{html.escape(str(row.get('time_slab', '')))}</td>"
            f"<td>{html.escape(str(row.get('qpu_output_directories', '')))}</td>"
            f"<td>{html.escape(str(row.get('cones_with_selected_source', '')))}</td>"
            f"<td>{report_cell}</td>"
            f"<td>{html.escape(str(row.get('error', '')))}</td>"
            "</tr>"
        )
    html_path = report_dir / "visualization_batch_index.html"
    html_path.write_text(
        """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>QPU visualization batch index</title><style>
body{font-family:Arial,sans-serif;margin:0;background:#f4f6f8;color:#17202a}
main{max-width:1200px;margin:auto;padding:24px}table{width:100%;border-collapse:collapse;background:white}
th,td{padding:10px;border:1px solid #dfe4e8;text-align:left;vertical-align:top}
th{background:#eef2f5}code{background:#eef1f4;padding:2px 5px;border-radius:4px}
</style></head><body><main><h1>QPU visualization batch index</h1>"""
        f"<p>Search root: <code>{html.escape(str(search_root))}</code></p>"
        "<table><thead><tr><th>Case</th><th>Status</th><th>Time slab</th>"
        "<th>QPU output directories</th><th>Cones with selected source</th>"
        "<th>Report</th><th>Error</th></tr></thead><tbody>"
        + "".join(table_rows)
        + "</tbody></table></main></body></html>",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualise one Schwarz output directory, or automatically discover "
            "and visualise every existing QPU case below a parent directory."
        )
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/fig6",
        help=(
            "A single case directory or a parent such as outputs/fig6. Parent "
            "directories are searched recursively for existing result cases."
        ),
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--time-slab", default="latest", help="Integer or 'latest'")
    parser.add_argument(
        "--source",
        choices=("auto", "qpu", "carleman", "nonlinear"),
        default="qpu",
        help=(
            "Data source for nuclear metrics. The default 'qpu' also filters "
            "batch discovery to cases containing real-QPU result markers."
        ),
    )
    parser.add_argument("--metric", choices=AVAILABLE_METRICS, default="nuclear_free_mean")
    parser.add_argument("--cone-id", default=None)
    parser.add_argument("--phi-deg", type=float, default=0.0)
    parser.add_argument("--animate", action="store_true")
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--labels", action="store_true")
    parser.add_argument("--show", action="store_true")
    parser.add_argument(
        "--save-dir",
        default=None,
        help=(
            "Single-case destination, or a parent destination in batch mode. "
            "By default, each case writes to its own visualization/ directory."
        ),
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop at the first failed case instead of continuing the batch.",
    )
    if "ipykernel" in sys.modules or "ipykernel_launcher" in Path(sys.argv[0]).name:
        return parser.parse_known_args()[0]
    return parser.parse_args()


def visualise_one_case(
    output_dir: Path,
    save_dir: Path,
    *,
    args: argparse.Namespace,
    config: Path | None,
) -> dict[str, Any]:
    save_dir.mkdir(parents=True, exist_ok=True)
    raw = load_parameters(config, output_dir)
    model = dict(raw.get("model", raw))
    time_slab = resolve_time_slab(output_dir, str(args.time_slab))

    nodes = load_nodes(output_dir)
    links = load_links(output_dir)
    schedule = load_schedule(output_dir)
    handoffs = load_handoffs(output_dir, time_slab)
    summary = summarize_cones(output_dir, nodes, time_slab, args.source, model, handoffs)
    values = metric_values(nodes, summary, args.metric)

    print(f"Loaded geometry cones: {len(nodes)}")
    print(f"Geometry links: {len(links)}")
    print(f"Handoffs in time slab {time_slab}: {len(handoffs)}")
    print(f"Metric: {args.metric}; nuclear source policy: {args.source}")

    generated: list[Path] = []
    overview = save_dir / f"layer_overview_3d_{args.metric}.png"
    plot_overview_3d(overview, nodes, links, handoffs, values, args.metric, labels=args.labels, show=args.show)
    generated.append(overview)
    print(f"Saved: {overview}")

    unwrapped = save_dir / f"layer_unwrapped_{args.metric}.png"
    plot_unwrapped(unwrapped, nodes, links, handoffs, values, args.metric, labels=args.labels, show=args.show)
    generated.append(unwrapped)
    print(f"Saved: {unwrapped}")

    patch_audit = save_dir / "geometry_patch_audit.png"
    plot_patch_audit(patch_audit, links, show=args.show)
    generated.append(patch_audit)
    print(f"Saved: {patch_audit}")

    convergence = save_dir / "schwarz_convergence.png"
    if plot_schwarz_convergence(convergence, output_dir, show=args.show):
        generated.append(convergence)
        print(f"Saved: {convergence}")

    if args.animate:
        execution = load_execution_rows(output_dir, time_slab)
        gif = save_dir / f"layer_execution_{args.metric}.gif"
        if animate_execution(
            gif,
            nodes,
            values,
            execution,
            handoffs,
            args.metric,
            fps=args.fps,
            labels=args.labels,
            show=args.show,
        ):
            generated.append(gif)
            print(f"Saved: {gif}")
        else:
            warnings.warn("run_summary.csv has no execution rows; GIF was not generated")

    if args.cone_id:
        if args.cone_id not in nodes:
            raise ValueError(f"Unknown cone ID {args.cone_id}; examples: {', '.join(schedule[:6])}")
        detailed = locate_detailed_output(output_dir, time_slab, args.cone_id)
        state_path = detailed / "final_state.npz"
        if not state_path.exists():
            state_path = cone_output_dir(output_dir, time_slab, args.cone_id) / "final_state.npz"
        internal = save_dir / f"cone_{args.cone_id}_internal.png"
        plot_internal_cone(internal, state_path, model, args.cone_id, phi_deg=args.phi_deg, show=args.show)
        generated.append(internal)
        print(f"Loaded state: {state_path}")
        print(f"Saved: {internal}")

    summary_csv = save_dir / "visualization_summary.csv"
    write_summary_csv(summary_csv, summary)
    print(f"Saved: {summary_csv}")
    summary_json = save_dir / "visualization_summary.json"
    summary_json.write_text(json.dumps(json_safe(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {summary_json}")

    report = save_dir / "visualization_report.html"
    write_html_report(
        report,
        metric=args.metric,
        time_slab=time_slab,
        source=args.source,
        summary=summary,
        images=generated,
    )
    print(f"Saved: {report}")
    print(f"Visualization directory: {save_dir}")
    cones_with_source = sum(
        1
        for row in summary.values()
        if bool(row.get("available")) and row.get("nuclear_source") == args.source
    )
    if args.source == "auto":
        cones_with_source = sum(
            1
            for row in summary.values()
            if bool(row.get("available")) and bool(row.get("nuclear_source"))
        )
    return {
        "status": "complete",
        "output_dir": str(output_dir),
        "time_slab": time_slab,
        "qpu_output_directories": len(find_qpu_output_directories(output_dir)),
        "cones_with_selected_source": cones_with_source,
        "visualization_dir": str(save_dir),
        "report": str(report),
        "error": "",
    }


def main() -> int:
    args = parse_arguments()
    search_root = Path(args.output_dir).resolve()
    discovered = discover_result_directories(search_root)
    if not discovered:
        raise FileNotFoundError(
            f"No Schwarz result directories were found below {search_root}"
        )

    if args.source == "qpu":
        selected: list[Path] = []
        for output_dir in discovered:
            qpu_directories = find_qpu_output_directories(output_dir)
            if qpu_directories:
                selected.append(output_dir)
            else:
                print(f"Skip (no existing QPU result markers): {output_dir}")
    else:
        selected = discovered

    if not selected:
        raise FileNotFoundError(
            f"No result directories containing QPU outputs were found below {search_root}"
        )
    if args.config and len(selected) > 1:
        raise ValueError("--config can only be used when visualising one case")

    batch_mode = len(selected) > 1 or not is_result_directory(search_root)
    custom_save_root = Path(args.save_dir).resolve() if args.save_dir else None
    config = Path(args.config).resolve() if args.config else None
    batch_rows: list[dict[str, Any]] = []
    failures = 0

    print(f"Discovered result cases: {len(discovered)}")
    print(f"Selected for visualization: {len(selected)}")
    for index, output_dir in enumerate(selected, start=1):
        label = case_label(search_root, output_dir)
        if custom_save_root is None:
            save_dir = output_dir / "visualization"
        elif batch_mode:
            relative = Path(label) if label != output_dir.name else Path(output_dir.name)
            save_dir = custom_save_root / relative
        else:
            save_dir = custom_save_root

        print(f"\n===== Visualize case {index}/{len(selected)}: {label} =====")
        try:
            row = visualise_one_case(
                output_dir,
                save_dir,
                args=args,
                config=config,
            )
            row["case"] = label
        except Exception as error:
            failures += 1
            row = {
                "case": label,
                "status": "failed",
                "output_dir": str(output_dir),
                "time_slab": "",
                "qpu_output_directories": len(find_qpu_output_directories(output_dir)),
                "cones_with_selected_source": 0,
                "visualization_dir": str(save_dir),
                "report": "",
                "error": str(error),
            }
            print(f"Visualization failed for {label}: {error}", file=sys.stderr)
            if args.fail_fast:
                raise
        batch_rows.append(row)

    if batch_mode:
        batch_report_dir = custom_save_root or search_root / "visualization_all_qpu"
        write_batch_report(
            batch_report_dir,
            search_root=search_root,
            rows=batch_rows,
        )
        print(f"\nBatch index: {batch_report_dir / 'visualization_batch_index.html'}")
        print(f"Batch CSV: {batch_report_dir / 'visualization_batch_summary.csv'}")

    print(
        f"Visualization complete: {len(selected) - failures} succeeded, "
        f"{failures} failed."
    )
    return 1 if failures else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"\nVisualization failed: {error}", file=sys.stderr)
        raise
