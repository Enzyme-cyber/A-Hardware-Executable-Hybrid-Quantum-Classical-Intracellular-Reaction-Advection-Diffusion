"""Three-layer geometric multiplicative-Schwarz controller for OriginQ cones.

Every global time slab is solved repeatedly from the SAME slab-start state.
Only lateral interface data changes between Schwarz iterations.  Therefore K
iterations do not accidentally advance physical time K times.

The contact surface for every donor/receiver pair is selected from the actual
3-D geometry in :mod:`layer_geometry`, then mapped to an exactly reciprocal
pair of compact-cone SIDE_xx patches.

Execution uses a target-centred triangular dependency cell.  For each angular
cell the upper and lower cones are solved first, then the target cone receives
both cross-layer interfaces together with the already-computed target from the
active half-ring sweep.  Two sweeps leave the first cone in opposite directions
and meet at the antipodal cone, avoiding a single large cyclic seam error.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import argparse
import copy
import csv
import json
import math
import shutil
import subprocess
import sys

import numpy as np

from cone3d_model import build_model_system, load_parameters
from cone_interfaces import (
    build_neighbor_state,
    load_interface_parameters,
    restrict_to_lateral_initial,
    save_neighbor_state,
)
from layer_geometry import (
    ConeNode,
    DirectedInterface,
    LayerGeometry,
    ScheduleEntry,
    build_layer_geometry,
    geometry_from_dict,
    geometry_summary,
)


SPECIES = ("free", "stationary", "mobile")


@dataclass(frozen=True)
class SchwarzParameters:
    time_start: float = 0.0
    time_window: float = 0.08
    time_slabs: int = 1
    max_iterations: int = 2
    minimum_iterations: int = 2
    tolerance: float = 0.03
    relaxation: float = 0.65
    output_dir: str = "layer_geometry_schwarz_output"


@dataclass(frozen=True)
class TransferParameters:
    efficiency: float = 0.28
    free_fraction: float = 1.0
    stationary_fraction: float = 0.0
    mobile_fraction: float = 0.60
    maximum_injected_concentration: float = 5.0
    max_neighbors: int = 4


@dataclass(frozen=True)
class InitialisationParameters:
    root_cone_ids: tuple[str, ...] = ("L1-C001",)
    root_base_initial_scale: float = 1.0
    nonroot_base_initial_scale: float = 0.0
    layer_scales: tuple[float, float, float] = (1.0, 1.0, 1.0)


def _dataclass_from_section(raw: dict[str, Any], section: str, cls: Any) -> Any:
    data = dict(raw.get(section, {}))
    valid = set(cls.__dataclass_fields__)
    unknown = sorted(set(data) - valid)
    if unknown:
        raise ValueError(f"Unknown {section} parameters: {unknown}")
    if cls is InitialisationParameters:
        if "root_cone_ids" in data:
            data["root_cone_ids"] = tuple(str(v) for v in data["root_cone_ids"])
        if "layer_scales" in data:
            data["layer_scales"] = tuple(float(v) for v in data["layer_scales"])
    result = cls(**data)
    return result


def load_schwarz(raw: dict[str, Any]) -> SchwarzParameters:
    result = _dataclass_from_section(raw, "schwarz", SchwarzParameters)
    if result.time_window <= 0.0 or result.time_slabs < 1:
        raise ValueError("time_window must be positive and time_slabs >= 1")
    if result.max_iterations < 1:
        raise ValueError("max_iterations must be >= 1")
    if result.minimum_iterations < 1:
        raise ValueError("minimum_iterations must be >= 1")
    return result


def load_transfer(raw: dict[str, Any]) -> TransferParameters:
    return _dataclass_from_section(raw, "transfer", TransferParameters)


def load_initialisation(raw: dict[str, Any]) -> InitialisationParameters:
    result = _dataclass_from_section(raw, "initialisation", InitialisationParameters)
    if len(result.layer_scales) != 3:
        raise ValueError("initialisation.layer_scales must contain three values")
    return result


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exact three-layer geometry + multiplicative Schwarz + OriginQ"
    )
    parser.add_argument("--config", default="configs/fig6/parameters_1.json")
    parser.add_argument("--mode", choices=("qpu", "classical_mock"))
    parser.add_argument("--dry-plan", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-sectors", type=int, default=None)
    parser.add_argument("--max-cones", type=int, default=None)
    parser.add_argument("--max-targets-per-cone", type=int, default=None)
    parser.add_argument("--max-schwarz-iterations", type=int, default=None)
    if "ipykernel" in sys.modules or "ipykernel_launcher" in Path(sys.argv[0]).name:
        return parser.parse_known_args()[0]
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def run_subprocess(command: list[str], log_path: Path) -> None:
    print("COMMAND:", subprocess.list2cmdline(command))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
        return_code = process.wait()
    if return_code:
        raise RuntimeError(
            f"Cone runner exited with code {return_code}; see {log_path}"
        )


def _angle_in_range(angle_deg: float, start_deg: float, end_deg: float) -> bool:
    angle = angle_deg % 360.0
    start = start_deg % 360.0
    end = end_deg % 360.0
    return start <= angle <= end if start <= end else angle >= start or angle <= end


def input_matches(node: ConeNode, specification: dict[str, Any]) -> bool:
    selector = dict(specification.get("selector", {}))
    if bool(selector.get("all", False)):
        return True
    if "cone_ids" in selector and node.cone_id not in set(map(str, selector["cone_ids"])):
        return False
    if "roles" in selector and node.role not in set(map(str, selector["roles"])):
        return False
    if "layers" in selector and node.layer + 1 not in {int(v) for v in selector["layers"]}:
        return False
    if "cone_indices" in selector and node.sector + 1 not in {
        int(v) for v in selector["cone_indices"]
    }:
        return False
    if "angle_ranges_deg" in selector:
        angle = math.degrees(node.angle) % 360.0
        if not any(
            _angle_in_range(angle, float(pair[0]), float(pair[1]))
            for pair in selector["angle_ranges_deg"]
        ):
            return False
    every_n = int(selector.get("every_n", 0))
    if every_n > 0 and node.sector % every_n != int(selector.get("offset", 0)) % every_n:
        return False
    return bool(selector) or bool(specification.get("apply_to_all", False))


def resolve_surface_inputs(
    node: ConeNode,
    specifications: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for item in specifications:
        if not bool(item.get("enabled", True)) or not input_matches(node, item):
            continue
        payload = {
            key: copy.deepcopy(value)
            for key, value in item.items()
            if key not in {"selector", "apply_to_all"}
        }
        payload["cone_id"] = node.cone_id
        payload.setdefault("input_id", f"INPUT-{len(result)+1:03d}")
        payload.setdefault("surface", "sidewall")
        payload.setdefault("mode", "source")
        payload.setdefault(
            "concentrations", {"free": 0.0, "stationary": 0.0, "mobile": 0.0}
        )
        result.append(payload)
    return result


def save_compact_state(path: Path, system: Any, state: np.ndarray, metadata: dict[str, Any]) -> None:
    n = system.grid.z.size
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        free=np.asarray(state[:n], dtype=float),
        stationary=np.asarray(state[n : 2 * n], dtype=float),
        mobile=np.asarray(state[2 * n : 3 * n], dtype=float),
        ring_counts=np.asarray(system.grid.ring_counts, dtype=int),
        z_index=system.grid.z_index,
        ring_index=system.grid.ring_index,
        sector_index=system.grid.sector_index,
        metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False)),
    )


def make_cone_config(
    raw: dict[str, Any],
    *,
    initial_scale: float,
    output_dir: Path,
) -> dict[str, Any]:
    payload = copy.deepcopy(raw)
    model = payload.setdefault("model", {})
    for key in (
        "initial_free_amplitude",
        "initial_stationary_occupancy",
        "initial_mobile_occupancy",
    ):
        if key in model:
            model[key] = float(model[key]) * float(initial_scale)
    payload.setdefault("hardware", {})["output_dir"] = str(output_dir)
    return payload


def directed_for_receiver(
    geometry: LayerGeometry,
    receiver_id: str,
) -> list[DirectedInterface]:
    links = geometry.links_by_node[receiver_id]
    directed: list[DirectedInterface] = []
    for link in links:
        donor = link.node_b if link.node_a == receiver_id else link.node_a
        directed.append(link.oriented(donor, receiver_id))
    return directed


def _donor_output_path(
    donor_id: str,
    current_outputs: dict[str, Path],
    previous_iter_outputs: dict[str, Path],
) -> tuple[Path | None, str]:
    current = current_outputs.get(donor_id)
    if current is not None:
        return current, "current_iteration"
    previous = previous_iter_outputs.get(donor_id)
    if previous is not None:
        return previous, "previous_iteration"
    return None, "unavailable"


def _target_predecessor_ids(
    geometry: LayerGeometry,
    entry: ScheduleEntry,
) -> set[str]:
    return {
        f"L1-C{sector + 1:03d}" for sector in entry.predecessor_sectors
    }


def select_contacts_for_step(
    *,
    geometry: LayerGeometry,
    receiver_id: str,
    entry: ScheduleEntry,
    selected_ids: set[str],
    current_outputs: dict[str, Path],
    previous_iter_outputs: dict[str, Path],
) -> tuple[list[DirectedInterface], list[Path], list[str]]:
    """Select geometry-consistent donors for one scheduled cone.

    Target-layer policy
    -------------------
    * Always take its resolved upper and lower donor (current iteration is
      preferred; the schedule makes both available immediately beforehand).
    * Take only the preceding target on the active sweep arm.
    * At the antipodal meeting cone, take both arm predecessors.
    * At the start cone, use both same-layer neighbours from the previous
      Schwarz iteration when available, but never create a current-iteration
      wraparound dependency.

    Other layers retain multiplicative-Schwarz behaviour: all available
    geometric neighbours are accepted, preferring current-iteration values.
    """
    candidates = directed_for_receiver(geometry, receiver_id)
    if entry.role == "target":
        predecessor_ids = _target_predecessor_ids(geometry, entry)
        allowed: list[DirectedInterface] = []
        for contact in candidates:
            if contact.edge_type in {"target_upper", "target_lower"}:
                allowed.append(contact)
            elif contact.edge_type == "same_layer":
                if contact.donor_id in predecessor_ids:
                    allowed.append(contact)
                elif entry.is_start and contact.donor_id in previous_iter_outputs:
                    # The start has no directional predecessor.  Old values from
                    # both sides close the Schwarz iteration without creating a
                    # current-iteration cyclic dependency.
                    allowed.append(contact)
        candidates = allowed

    selected_contacts: list[DirectedInterface] = []
    donor_reports: list[Path] = []
    donor_epochs: list[str] = []
    edge_rank = {
        "target_upper": 0,
        "target_lower": 1,
        "same_layer": 2,
    }
    for contact in sorted(
        candidates,
        key=lambda item: (
            edge_rank.get(item.edge_type, 9),
            item.distance / max(item.weight, 1e-12),
            item.donor_id,
        ),
    ):
        if contact.donor_id not in selected_ids:
            continue
        donor_output, epoch = _donor_output_path(
            contact.donor_id, current_outputs, previous_iter_outputs
        )
        if donor_output is None:
            continue
        report_path = donor_output / "patch_coefficients.json"
        if not report_path.exists():
            continue
        selected_contacts.append(contact)
        donor_reports.append(report_path)
        donor_epochs.append(epoch)
    return selected_contacts, donor_reports, donor_epochs


def normalised_interface_weights(interfaces: list[DirectedInterface]) -> list[float]:
    if not interfaces:
        return []
    raw = np.asarray(
        [item.weight / max(item.distance, 1e-12) for item in interfaces], dtype=float
    )
    total = float(np.sum(raw))
    return (raw / total).tolist() if total > 0.0 else [1.0 / len(raw)] * len(raw)


def create_neighbor_input(
    *,
    system: Any,
    interface_config: Any,
    donor_report_path: Path,
    output_path: Path,
    contact: DirectedInterface,
    scale: float,
    transfer: TransferParameters,
    time_slab: int,
    iteration: int,
) -> dict[str, Any]:
    report = json.loads(donor_report_path.read_text(encoding="utf-8"))
    state, metadata = build_neighbor_state(
        system,
        report,
        interface_config,
        source_patch_id=contact.donor_patch_id,
        destination_patch_index=contact.receiver_patch_index,
        scale=float(scale),
    )
    # Enforce the side-only contract before the file is written.  The single-cone
    # runner applies the same restriction again defensively when loading it.
    state = restrict_to_lateral_initial(state, system, interface_config)

    fractions = {
        "free": transfer.free_fraction,
        "stationary": transfer.stationary_fraction,
        "mobile": transfer.mobile_fraction,
    }
    delivered_mass: dict[str, float] = {}
    for species, sl in system.species_slices.items():
        values = np.asarray(state[sl], dtype=float) * float(fractions[species])
        if transfer.maximum_injected_concentration > 0.0:
            values = np.minimum(values, transfer.maximum_injected_concentration)
        state[sl] = values
        delivered_mass[species] = float(np.sum(values * system.grid.volume))
    metadata.update(
        {
            "link_id": contact.link_id,
            "donor_id": contact.donor_id,
            "receiver_id": contact.receiver_id,
            "edge_type": contact.edge_type,
            "donor_semantic_side": contact.donor_side,
            "receiver_semantic_side": contact.receiver_side,
            "donor_local_phi_deg": math.degrees(contact.donor_local_phi) % 360.0,
            "receiver_local_phi_deg": math.degrees(contact.receiver_local_phi) % 360.0,
            "donor_patch_id": contact.donor_patch_id,
            "receiver_patch_id": contact.receiver_patch_id,
            "side_patch_count": int(interface_config.side_patch_count),
            "transfer_scale": float(scale),
            "species_transfer_fractions": fractions,
            "delivered_mass": delivered_mass,
            "time_slab": int(time_slab),
            "schwarz_iteration": int(iteration),
            "protocol": "geometry_mapped_additive_lateral_initial_state",
            "explicitly_not_membrane_input": True,
        }
    )
    save_neighbor_state(output_path, system, state, metadata)
    return {"path": str(output_path), **metadata}


def _preferred_coefficients(patch: dict[str, Any], species: str) -> np.ndarray | None:
    sources = patch.get("sources", {})
    for source in ("qpu", "carleman", "nonlinear"):
        payload = sources.get(source, {}).get(species)
        if payload and payload.get("coefficients") is not None:
            return np.asarray(payload["coefficients"], dtype=float)
    return None


def interface_residual(previous: dict[str, Path], current: dict[str, Path]) -> float:
    numerator = 0.0
    denominator = 0.0
    comparisons = 0
    for cone_id, current_path in current.items():
        old_path = previous.get(cone_id)
        if old_path is None or not old_path.exists() or not current_path.exists():
            continue
        old = json.loads(old_path.read_text(encoding="utf-8"))
        new = json.loads(current_path.read_text(encoding="utf-8"))
        old_patches = {item["patch_id"]: item for item in old.get("side_patches", [])}
        new_patches = {item["patch_id"]: item for item in new.get("side_patches", [])}
        for patch_id in sorted(set(old_patches) & set(new_patches)):
            for species in SPECIES:
                a = _preferred_coefficients(old_patches[patch_id], species)
                b = _preferred_coefficients(new_patches[patch_id], species)
                if a is None or b is None or a.shape != b.shape:
                    continue
                numerator += float(np.sum((b - a) ** 2))
                denominator += float(np.sum(a**2))
                comparisons += 1
    if comparisons == 0:
        return float("inf")
    return math.sqrt(numerator / max(denominator, 1e-24))


def write_geometry_files(output_dir: Path, geometry: LayerGeometry) -> None:
    node_rows: list[dict[str, Any]] = []
    for node in geometry.nodes:
        node_rows.append(
            {
                "schedule_order": node.order_index + 1,
                "cone_id": node.cone_id,
                "role": node.role,
                "layer": node.layer + 1,
                "sector": node.sector + 1,
                "azimuth_deg": math.degrees(node.angle) % 360.0,
                "polar_deg": math.degrees(node.polar_angle),
                "x": node.position[0],
                "y": node.position[1],
                "z": node.position[2],
                "axis_x": node.axis[0],
                "axis_y": node.axis[1],
                "axis_z": node.axis[2],
                "right_x": node.tangent[0],
                "right_y": node.tangent[1],
                "right_z": node.tangent[2],
                "top_x": node.vertical[0],
                "top_y": node.vertical[1],
                "top_z": node.vertical[2],
            }
        )
    link_rows: list[dict[str, Any]] = []
    for link in geometry.links:
        link_rows.append(
            {
                "link_id": link.link_id,
                "edge_type": link.edge_type,
                "node_a": link.node_a,
                "node_b": link.node_b,
                "distance": link.distance,
                "weight": link.weight,
                "a_side": link.a_side,
                "b_side": link.b_side,
                "a_local_phi_deg": math.degrees(link.a_local_phi) % 360.0,
                "b_local_phi_deg": math.degrees(link.b_local_phi) % 360.0,
                "a_patch_id": link.a_patch_id,
                "b_patch_id": link.b_patch_id,
                "a_patch_index": link.a_patch_index,
                "b_patch_index": link.b_patch_index,
                "opposite_patch_check": int(
                    (link.a_patch_index + geometry.parameters.side_patch_count // 2)
                    % geometry.parameters.side_patch_count
                    == link.b_patch_index
                ),
                "actual_opposition_error_deg": link.actual_opposition_error_deg,
            }
        )
    execution_order = {entry.cone_id: entry.order for entry in geometry.schedule_entries}
    for row in node_rows:
        row["execution_order"] = execution_order.get(row["cone_id"], "")

    schedule_rows = [
        {
            "order": entry.order,
            "anchor_sector": entry.sector + 1,
            "cone_actual_sector": geometry.nodes_by_id[entry.cone_id].sector + 1,
            "within_triangle": entry.role,
            "cone_id": entry.cone_id,
            "sweep_arm": entry.sweep_arm,
            "sweep_direction": entry.sweep_direction,
            "predecessor_target_sectors": ";".join(
                str(value + 1) for value in entry.predecessor_sectors
            ),
            "is_start": int(entry.is_start),
            "is_antipodal_meeting": int(entry.is_meeting),
        }
        for entry in geometry.schedule_entries
    ]
    triangle_rows = []
    entries = geometry.schedule_by_sector_role
    for sector in range(geometry.cone_count_per_layer):
        target = entries[(sector, "target")]
        upper = entries[(sector, "upper")]
        lower = entries[(sector, "lower")]
        triangle_rows.append(
            {
                "anchor_sector": sector + 1,
                "upper_cone": upper.cone_id,
                "lower_cone": lower.cone_id,
                "target_cone": target.cone_id,
                "target_predecessor_sectors": ";".join(
                    str(value + 1) for value in target.predecessor_sectors
                ),
                "sweep_arm": target.sweep_arm,
                "is_antipodal_meeting": int(target.is_meeting),
                "execution_order": f"{upper.order}->{lower.order}->{target.order}",
            }
        )
    write_csv(output_dir / "geometry_nodes.csv", node_rows)
    write_csv(output_dir / "geometry_links.csv", link_rows)
    write_csv(output_dir / "geometry_schedule.csv", schedule_rows)
    write_csv(output_dir / "geometry_triangular_cells.csv", triangle_rows)
    (output_dir / "geometry_summary.json").write_text(
        json.dumps(geometry_summary(geometry), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> int:
    args = parse_arguments()
    config_path = Path(args.config).resolve()
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    model, _ = load_parameters(config_path)
    geometry = build_layer_geometry(geometry_from_dict(raw))
    schwarz = load_schwarz(raw)
    transfer = load_transfer(raw)
    initialisation = load_initialisation(raw)
    dispatch = dict(raw.get("qpu_dispatch", {}))
    mode = args.mode or str(dispatch.get("mode", "qpu"))
    if mode not in {"qpu", "classical_mock"}:
        raise ValueError("mode must be qpu or classical_mock")

    max_iterations = (
        args.max_schwarz_iterations
        if args.max_schwarz_iterations is not None
        else schwarz.max_iterations
    )
    output_dir = Path(schwarz.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_geometry_files(output_dir, geometry)

    schedule = list(geometry.schedule)
    if args.max_sectors is not None:
        schedule = schedule[: max(0, args.max_sectors) * 3]
    if args.max_cones is not None:
        schedule = schedule[: max(0, args.max_cones)]
    selected_ids = set(schedule)
    nodes_by_id = geometry.nodes_by_id
    schedule_entries_by_cone = geometry.schedule_by_cone

    system, _ = build_model_system(model)
    interface_config = load_interface_parameters(raw)
    if interface_config.side_patch_count != geometry.parameters.side_patch_count:
        raise ValueError(
            "geometry.side_patch_count must equal interfaces.side_patch_count"
        )

    surface_specs = list(raw.get("surface_inputs", []))
    input_map = {
        cone_id: resolve_surface_inputs(nodes_by_id[cone_id], surface_specs)
        for cone_id in schedule
    }
    (output_dir / "resolved_surface_input_map.json").write_text(
        json.dumps(input_map, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "resolved_parameters.json").write_text(
        json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(json.dumps(geometry_summary(geometry), ensure_ascii=False, indent=2))
    print(f"Selected cones this run: {len(schedule)}")
    if args.dry_plan:
        print(f"Geometry plan only. Outputs: {output_dir}")
        return 0

    runner_raw = Path(dispatch.get("quantum_runner", "originq_3d_hardware_runner.py"))
    if runner_raw.is_absolute():
        runner = runner_raw.resolve()
    else:
        # Configurations are stored below configs/, whereas the executable
        # runner is stored next to this controller below src/. Prefer an
        # explicitly config-relative runner when present, then fall back to
        # the release's source directory.
        config_relative_runner = (config_path.parent / runner_raw).resolve()
        source_relative_runner = (Path(__file__).resolve().parent / runner_raw).resolve()
        runner = (
            config_relative_runner
            if config_relative_runner.exists()
            else source_relative_runner
        )
    if not runner.exists():
        raise FileNotFoundError(f"Quantum runner not found: {runner_raw}")
    python_executable = str(dispatch.get("python_executable", sys.executable))
    max_targets = (
        args.max_targets_per_cone
        if args.max_targets_per_cone is not None
        else int(dispatch.get("max_targets_per_cone", 0))
    )

    # Build first-slab analytic initial-state snapshots for auditing.  The runner
    # uses the equivalent scaled cone config in slab 0 so membrane_input.csv
    # correctly records it as a membrane input rather than continuation state.
    root_ids = set(initialisation.root_cone_ids)
    initial_paths: dict[str, Path] = {}
    initial_scales: dict[str, float] = {}
    for cone_id in schedule:
        node = nodes_by_id[cone_id]
        scale = (
            initialisation.root_base_initial_scale
            if cone_id in root_ids
            else initialisation.nonroot_base_initial_scale
        ) * initialisation.layer_scales[node.layer]
        initial_scales[cone_id] = float(scale)
        initial_state = np.asarray(system.initial_state, dtype=float) * float(scale)
        state_path = output_dir / "initial_states" / cone_id / "state.npz"
        save_compact_state(
            state_path,
            system,
            initial_state,
            {
                "cone_id": cone_id,
                "role": node.role,
                "initial_scale": scale,
                "meaning": "plasma_membrane_base_input",
            },
        )
        initial_paths[cone_id] = state_path

    run_rows: list[dict[str, Any]] = []
    handoff_rows: list[dict[str, Any]] = []
    iteration_rows: list[dict[str, Any]] = []
    slab_base_paths = dict(initial_paths)
    converged_outputs: dict[str, Path] = {}

    for slab in range(schwarz.time_slabs):
        t0 = schwarz.time_start + slab * schwarz.time_window
        t1 = t0 + schwarz.time_window
        print(f"\n===== GLOBAL TIME SLAB {slab}: [{t0}, {t1}] =====")
        previous_iter_outputs: dict[str, Path] = {}
        current_outputs: dict[str, Path] = {}

        for iteration in range(max_iterations):
            print(f"\n--- SCHWARZ ITERATION {iteration} ---")
            current_outputs = {}
            for order, cone_id in enumerate(schedule, start=1):
                node = nodes_by_id[cone_id]
                cone_dir = (
                    output_dir
                    / f"time_{slab:03d}"
                    / f"iter_{iteration:03d}"
                    / f"{order:03d}_{cone_id}"
                )
                qpu_output = cone_dir / "qpu_output"
                qpu_output.mkdir(parents=True, exist_ok=True)

                completed = (
                    (qpu_output / "final_state.npz").exists()
                    and (qpu_output / "patch_coefficients.json").exists()
                )
                if args.resume and completed:
                    print(f"Reuse completed {cone_id}: {qpu_output}")
                    current_outputs[cone_id] = qpu_output
                    continue

                entry = schedule_entries_by_cone[cone_id]
                contacts, donor_sources, donor_epochs = select_contacts_for_step(
                    geometry=geometry,
                    receiver_id=cone_id,
                    entry=entry,
                    selected_ids=selected_ids,
                    current_outputs=current_outputs,
                    previous_iter_outputs=previous_iter_outputs,
                )
                if transfer.max_neighbors > 0 and len(contacts) > transfer.max_neighbors:
                    ranked = sorted(
                        zip(contacts, donor_sources, donor_epochs, strict=True),
                        key=lambda pair: pair[0].distance / max(pair[0].weight, 1e-12),
                    )[: transfer.max_neighbors]
                    contacts = [pair[0] for pair in ranked]
                    donor_sources = [pair[1] for pair in ranked]
                    donor_epochs = [pair[2] for pair in ranked]
                weights = normalised_interface_weights(contacts)

                if entry.role == "target" and not entry.is_start:
                    required_predecessors = _target_predecessor_ids(geometry, entry)
                    received_ids = {contact.donor_id for contact in contacts}
                    missing_predecessors = required_predecessors - received_ids
                    if missing_predecessors:
                        print(
                            "WARNING: target triangular predecessor unavailable:",
                            cone_id,
                            sorted(missing_predecessors),
                        )

                side_files: list[Path] = []
                for donor_number, (contact, donor_report, donor_epoch, weight) in enumerate(
                    zip(contacts, donor_sources, donor_epochs, weights, strict=True), start=1
                ):
                    side_path = (
                        cone_dir
                        / "neighbor_inputs"
                        / f"{donor_number:02d}_{contact.donor_id}_to_{cone_id}_{contact.receiver_patch_id}.npz"
                    )
                    record = create_neighbor_input(
                        system=system,
                        interface_config=interface_config,
                        donor_report_path=donor_report,
                        output_path=side_path,
                        contact=contact,
                        scale=transfer.efficiency * schwarz.relaxation * weight,
                        transfer=transfer,
                        time_slab=slab,
                        iteration=iteration,
                    )
                    record["donor_epoch"] = donor_epoch
                    record["receiver_sweep_arm"] = entry.sweep_arm
                    record["receiver_is_antipodal_meeting"] = int(entry.is_meeting)
                    handoff_rows.append(record)
                    side_files.append(side_path)

                initial_scale = initial_scales[cone_id] if slab == 0 else 0.0
                cone_config = make_cone_config(
                    raw, initial_scale=initial_scale, output_dir=qpu_output
                )
                cone_config_path = cone_dir / "cone_config.json"
                cone_config_path.write_text(
                    json.dumps(cone_config, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                surface_payload = {
                    "cone_id": cone_id,
                    "role": node.role,
                    "resolved_surface_inputs": input_map.get(cone_id, []),
                }
                surface_path = cone_dir / "surface_inputs.json"
                surface_path.write_text(
                    json.dumps(surface_payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

                command = [
                    python_executable,
                    str(runner),
                    "--config",
                    str(cone_config_path),
                    "--output-dir",
                    str(qpu_output),
                    "--surface-inputs",
                    str(surface_path),
                    "--time-start",
                    str(t0),
                    "--time-end",
                    str(t1),
                    "--max-targets",
                    str(max_targets),
                ]
                # Slab 0 uses the scaled model initial condition directly.  All
                # later slabs use the same cone's converged state from slab start.
                if slab > 0:
                    command.extend(["--input-state", str(slab_base_paths[cone_id])])
                for path in side_files:
                    command.extend(["--side-input-state", str(path)])
                if mode == "classical_mock":
                    command.append("--classical-only")
                if bool(dispatch.get("skip_preflight", True)):
                    command.append("--skip-preflight")
                for option, flag in (
                    ("parallel_jobs", "--parallel-jobs"),
                    ("batch_size", "--batch-size"),
                    ("shots", "--shots"),
                    ("backend", "--backend"),
                    ("max_pauli_terms", "--max-pauli-terms"),
                    ("max_pauli_weight", "--max-pauli-weight"),
                    ("max_initial_components", "--max-initial-components"),
                ):
                    if dispatch.get(option) is not None:
                        command.extend([flag, str(dispatch[option])])

                row = {
                    "time_slab": slab,
                    "iteration": iteration,
                    "schedule_order": order,
                    "cone_id": cone_id,
                    "role": node.role,
                    "anchor_sector": entry.sector + 1,
                    "sweep_arm": entry.sweep_arm,
                    "sweep_direction": entry.sweep_direction,
                    "is_antipodal_meeting": int(entry.is_meeting),
                    "status": "running",
                    "incoming_neighbors": ";".join(c.donor_id for c in contacts),
                    "incoming_links": ";".join(c.link_id for c in contacts),
                    "incoming_receiver_sides": ";".join(c.receiver_side for c in contacts),
                    "incoming_receiver_patches": ";".join(c.receiver_patch_id for c in contacts),
                    "incoming_donor_epochs": ";".join(donor_epochs),
                    "side_input_files": ";".join(str(path) for path in side_files),
                    "output_dir": str(qpu_output),
                }
                run_rows.append(row)
                write_csv(output_dir / "run_summary.csv", run_rows)
                run_subprocess(command, cone_dir / "cone_process.log")
                row["status"] = "complete"
                current_outputs[cone_id] = qpu_output
                write_csv(output_dir / "run_summary.csv", run_rows)
                write_csv(output_dir / "handoff_log.csv", handoff_rows)

            current_patch_paths = {
                cone_id: path / "patch_coefficients.json"
                for cone_id, path in current_outputs.items()
            }
            previous_patch_paths = {
                cone_id: path / "patch_coefficients.json"
                for cone_id, path in previous_iter_outputs.items()
            }
            residual = interface_residual(previous_patch_paths, current_patch_paths)
            iteration_rows.append(
                {
                    "time_slab": slab,
                    "iteration": iteration,
                    "interface_relative_residual": residual,
                    "tolerance": schwarz.tolerance,
                    "completed_cones": len(current_outputs),
                }
            )
            write_csv(output_dir / "schwarz_iterations.csv", iteration_rows)
            print(f"Schwarz interface relative residual: {residual:.6e}")
            previous_iter_outputs = dict(current_outputs)
            if (
                iteration + 1 >= schwarz.minimum_iterations
                and math.isfinite(residual)
                and residual <= schwarz.tolerance
            ):
                print("Interface converged; advance to next global time slab.")
                break

        converged_outputs = dict(current_outputs)
        if len(converged_outputs) != len(schedule):
            missing = sorted(set(schedule) - set(converged_outputs))
            raise RuntimeError(f"Missing converged cone outputs: {missing}")
        converged_dir = output_dir / f"time_{slab:03d}" / "converged"
        converged_dir.mkdir(parents=True, exist_ok=True)
        next_base: dict[str, Path] = {}
        for cone_id, qpu_output in converged_outputs.items():
            cone_folder = converged_dir / cone_id
            cone_folder.mkdir(parents=True, exist_ok=True)
            for name in (
                "final_state.npz",
                "patch_coefficients.json",
                "nuclear_output.json",
                "membrane_input.csv",
                "side_initial_input.csv",
            ):
                source = qpu_output / name
                if source.exists():
                    shutil.copy2(source, cone_folder / name)
            next_base[cone_id] = cone_folder / "final_state.npz"
        slab_base_paths = next_base

    summary = {
        "mode": mode,
        "geometry": geometry_summary(geometry),
        "selected_cones": schedule,
        "time_slabs": schwarz.time_slabs,
        "max_schwarz_iterations": max_iterations,
        "completed_cones_last_slab": len(converged_outputs),
        "output_dir": str(output_dir),
        "interface_contract": {
            "plasma_membrane_input": "model initial or explicit --surface-inputs at z=1",
            "same_cone_next_time_slab": "--input-state final_state.npz",
            "neighbor_lateral_handoff": "--side-input-state using geometry-matched reciprocal patches",
            "target_triangular_inputs": "current upper + current lower + preceding target on active sweep arm; antipodal target receives both arm predecessors",
            "ring_schedule": "start, clockwise half-arm, counterclockwise half-arm, antipodal meeting",
            "nuclear_result": "nuclear_output.json at z=0",
        },
    }
    (output_dir / "run_complete.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nLayer geometry/Schwarz run complete: {output_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"\nLayer geometry run failed: {error}", file=sys.stderr)
        raise
