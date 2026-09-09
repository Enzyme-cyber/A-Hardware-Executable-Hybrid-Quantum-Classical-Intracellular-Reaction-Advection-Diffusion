"""Surface/interface utilities for the compact true-3-D conical model.

The compact grid collapses the central radial ring to one control volume per z
layer, so state arrays are stored as flat compact vectors rather than a dense
(nz, nr, nphi) block with duplicate centre points.

This module focuses on the two surfaces that matter for cone coupling:
* nuclear-envelope face (lower z face), used as the biological result output;
* lateral sidewall patches, used to seed neighbouring cones.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
import csv
import json
import math

import numpy as np
from numpy.polynomial.legendre import legvander, legval

from cone3d_model import Grid3D, ModelSystem, periodic_angle_difference


@dataclass(frozen=True)
class InterfaceParameters:
    nuclear_enabled: bool = True
    nuclear_radial_order: int = 1
    nuclear_fourier_modes: int = 1
    nuclear_sample_count: int = 5

    side_enabled: bool = True
    side_patch_count: int = 6
    side_legendre_order: int = 1
    side_samples_per_patch: int = 2

    # Real-QPU point samples.  Free material is the state that actually crosses
    # the open sidewall in the present biological model.  Other species remain
    # available as classical interface outputs and can be enabled here.
    # The nuclear output needs both free and motor-bound material because both
    # contribute to nuclear capture.  Lateral handoff currently transfers only
    # the freely exchangeable material, avoiding unnecessary sidewall circuits.
    nuclear_qpu_species: tuple[str, ...] = ("free", "mobile")
    side_qpu_species: tuple[str, ...] = ("free",)
    classical_species: tuple[str, ...] = ("free", "stationary", "mobile")

    export_neighbor_states: bool = True
    neighbor_transfer_species: tuple[str, ...] = ("free",)
    neighbor_radial_depth: float = 0.22
    neighbor_azimuth_sigma_deg: float = 35.0
    # A neighbouring cone contributes an initial field along the lateral shell,
    # not to the plasma-membrane input face.
    side_initial_exclude_membrane_face: bool = True
    side_initial_radial_cutoff_sigma: float = 3.0
    handoff_scale: float = 1.0
    prefer_qpu_coefficients: bool = True


def load_interface_parameters(raw: dict[str, Any]) -> InterfaceParameters:
    data = dict(raw.get("interfaces", {}))
    # Backward compatibility: an old qpu_species entry applies to both faces.
    if "qpu_species" in data:
        legacy = tuple(str(v) for v in data.pop("qpu_species"))
        data.setdefault("nuclear_qpu_species", legacy)
        data.setdefault("side_qpu_species", legacy)
    for key in (
        "nuclear_qpu_species", "side_qpu_species", "classical_species",
        "neighbor_transfer_species",
    ):
        if key in data:
            data[key] = tuple(str(v) for v in data[key])
    valid = set(InterfaceParameters.__dataclass_fields__)
    unknown = sorted(set(data) - valid)
    if unknown:
        raise ValueError(f"Unknown interface parameters: {unknown}")
    result = InterfaceParameters(**data)
    if result.side_patch_count < 1:
        raise ValueError("side_patch_count must be positive")
    if result.side_samples_per_patch < result.side_legendre_order + 1:
        raise ValueError(
            "side_samples_per_patch must be at least side_legendre_order + 1"
        )
    return result


def nuclear_layer_indices(grid: Grid3D) -> np.ndarray:
    """Cells touching the nuclear-envelope end face (z=0)."""
    return np.flatnonzero(grid.z_index == int(np.min(grid.z_index))).astype(int)


def membrane_input_layer_indices(grid: Grid3D) -> np.ndarray:
    """Cells touching the plasma-membrane input end face (z=1)."""
    return np.flatnonzero(grid.z_index == int(np.max(grid.z_index))).astype(int)


def sidewall_indices(grid: Grid3D) -> np.ndarray:
    return np.flatnonzero(grid.ring_index == len(grid.ring_counts) - 1).astype(int)


def patch_index_from_phi(phi: float, patch_count: int) -> int:
    wrapped = float(phi % (2.0 * math.pi))
    return min(int(wrapped / (2.0 * math.pi) * patch_count), patch_count - 1)


def patch_id(index: int) -> str:
    return f"SIDE_{index:02d}"


def patch_interval(index: int, patch_count: int) -> tuple[float, float, float]:
    width = 2.0 * math.pi / patch_count
    start = index * width
    end = (index + 1) * width
    center = (index + 0.5) * width
    return start, end, center


def side_patch_indices(grid: Grid3D, patch: int, patch_count: int) -> np.ndarray:
    side = sidewall_indices(grid)
    selected = [
        int(cell)
        for cell in side
        if patch_index_from_phi(float(grid.phi[cell]), patch_count) == patch
    ]
    return np.asarray(selected, dtype=int)


def _normalised_weights(weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=float)
    total = float(np.sum(weights))
    if total <= 0.0:
        return np.full(weights.shape, 1.0 / max(weights.size, 1), dtype=float)
    return weights / total


def _weighted_fit(
    matrix: np.ndarray,
    values: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, float]:
    matrix = np.asarray(matrix, dtype=float)
    values = np.asarray(values, dtype=float)
    weights = np.maximum(np.asarray(weights, dtype=float), 1e-14)
    root = np.sqrt(weights)
    design = matrix * root[:, None]
    target = values * root
    coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
    residual = values - matrix @ coefficients
    denominator = max(float(np.linalg.norm(values * root)), 1e-14)
    relative_residual = float(np.linalg.norm(residual * root) / denominator)
    return coefficients, relative_residual


def nuclear_basis(
    rho: np.ndarray,
    phi: np.ndarray,
    config: InterfaceParameters,
) -> tuple[np.ndarray, list[str]]:
    x = 2.0 * np.asarray(rho, dtype=float) - 1.0
    blocks = [legvander(x, config.nuclear_radial_order)]
    labels = [f"P_rho_{order}" for order in range(config.nuclear_radial_order + 1)]
    phi = np.asarray(phi, dtype=float)
    for mode in range(1, config.nuclear_fourier_modes + 1):
        blocks.append(np.cos(mode * phi)[:, None])
        blocks.append(np.sin(mode * phi)[:, None])
        labels.extend([f"cos_phi_{mode}", f"sin_phi_{mode}"])
    return np.hstack(blocks), labels


def side_basis(z: np.ndarray, order: int) -> tuple[np.ndarray, list[str]]:
    x = 2.0 * np.asarray(z, dtype=float) - 1.0
    return legvander(x, order), [f"P_z_{degree}" for degree in range(order + 1)]


def _surface_statistics(values: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    total_area = float(np.sum(weights))
    mean = float(np.sum(weights * values) / max(total_area, 1e-14))
    return {
        "area": total_area,
        "area_weighted_mean": mean,
        "area_integral": float(np.sum(weights * values)),
        "minimum": float(np.min(values)) if values.size else 0.0,
        "maximum": float(np.max(values)) if values.size else 0.0,
    }


def compute_classical_interface_report(
    system: ModelSystem,
    fields: dict[str, np.ndarray],
    config: InterfaceParameters,
    *,
    time_start: float,
    time_end: float,
    membrane_initial_state: np.ndarray | None = None,
    side_initial_state: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute nuclear and side-patch outputs for multiple solution sources.

    fields maps a source name (for example ``nonlinear`` or ``carleman``) to the
    full three-species physical state vector.
    """
    grid = system.grid
    report: dict[str, Any] = {
        "format_version": "cone-interface-v3-nuclear-output",
        "time_start": float(time_start),
        "time_end": float(time_end),
        "compact_layout": True,
        "spatial_cells": int(grid.z.size),
        "ring_counts": list(grid.ring_counts),
        "direction_convention": {
            "plasma_membrane_input": "z=1",
            "nuclear_output": "z=0",
            "lateral_handoff": "outer sidewall patches",
        },
        "membrane_input": {},
        "side_initial_input": {},
        "nuclear_face": {},
        "side_patches": [],
    }

    # The plasma-membrane input is recorded from the membrane component only.
    # Neighbour handoff fields are kept in a separate lateral-initial component
    # and therefore never redefine this membrane input record.
    membrane_state = (
        np.asarray(membrane_initial_state, dtype=float)
        if membrane_initial_state is not None
        else np.asarray(system.initial_state, dtype=float)
    )
    membrane_indices = membrane_input_layer_indices(grid)
    membrane_weights = grid.upper_z_area[membrane_indices]
    membrane_payload: dict[str, Any] = {
        "cell_indices": membrane_indices.tolist(),
        "role": "plasma_membrane_input",
        "sources": {"initial": {}},
    }
    for species in config.classical_species:
        species_slice = system.species_slices[species]
        values = np.asarray(membrane_state[species_slice][membrane_indices], dtype=float)
        membrane_payload["sources"]["initial"][species] = _surface_statistics(
            values, membrane_weights
        )
    report["membrane_input"] = membrane_payload

    # Record the neighbour-derived side initial field independently.  It is a
    # volumetric seed adjacent to the outer sidewall, not a plasma-membrane input.
    if side_initial_state is not None:
        side_state = np.asarray(side_initial_state, dtype=float)
        side_payload: dict[str, Any] = {
            "role": "lateral_initial_condition",
            "total_nonzero_cells": int(np.count_nonzero(np.abs(side_state) > 0.0)),
            "patches": [],
        }
        for patch in range(config.side_patch_count):
            indices = side_patch_indices(grid, patch, config.side_patch_count)
            if indices.size == 0:
                continue
            weights = grid.outer_side_area[indices]
            item: dict[str, Any] = {
                "patch_id": patch_id(patch),
                "cell_indices": indices.tolist(),
                "species": {},
            }
            for species in config.classical_species:
                sl = system.species_slices[species]
                values = np.asarray(side_state[sl][indices], dtype=float)
                item["species"][species] = _surface_statistics(values, weights)
            side_payload["patches"].append(item)
        report["side_initial_input"] = side_payload

    if config.nuclear_enabled:
        indices = nuclear_layer_indices(grid)
        weights = grid.lower_z_area[indices]
        basis, labels = nuclear_basis(grid.rho_metric[indices], grid.phi[indices], config)
        nuclear: dict[str, Any] = {
            "cell_indices": indices.tolist(),
            "basis_labels": labels,
            "sources": {},
        }
        for source_name, state in fields.items():
            source_payload: dict[str, Any] = {}
            for species in config.classical_species:
                species_slice = system.species_slices[species]
                values = np.asarray(state[species_slice][indices], dtype=float)
                coefficients, residual = _weighted_fit(basis, values, weights)
                source_payload[species] = {
                    **_surface_statistics(values, weights),
                    "coefficients": coefficients.tolist(),
                    "relative_fit_residual": residual,
                }
            free_values = np.asarray(
                state[system.species_slices["free"]][indices], dtype=float
            )
            mobile_values = np.asarray(
                state[system.species_slices["mobile"]][indices], dtype=float
            )
            source_payload["total_nuclear_capture_flux"] = float(
                system.parameters.nuclear_permeability_free
                * np.sum(weights * free_values)
                + system.parameters.nuclear_permeability_mobile
                * np.sum(weights * mobile_values)
            )
            nuclear["sources"][source_name] = source_payload
        report["nuclear_face"] = nuclear

    if config.side_enabled:
        for patch in range(config.side_patch_count):
            indices = side_patch_indices(grid, patch, config.side_patch_count)
            if indices.size == 0:
                continue
            weights = grid.outer_side_area[indices]
            basis, labels = side_basis(grid.z[indices], config.side_legendre_order)
            start, end, center = patch_interval(patch, config.side_patch_count)
            patch_payload: dict[str, Any] = {
                "patch_id": patch_id(patch),
                "patch_index": patch,
                "phi_start_deg": math.degrees(start),
                "phi_end_deg": math.degrees(end),
                "phi_center_deg": math.degrees(center),
                "cell_indices": indices.tolist(),
                "basis_labels": labels,
                "sources": {},
            }
            for source_name, state in fields.items():
                source_payload: dict[str, Any] = {}
                for species in config.classical_species:
                    species_slice = system.species_slices[species]
                    values = np.asarray(state[species_slice][indices], dtype=float)
                    coefficients, residual = _weighted_fit(basis, values, weights)
                    payload = {
                        **_surface_statistics(values, weights),
                        "coefficients": coefficients.tolist(),
                        "relative_fit_residual": residual,
                    }
                    if species == "free":
                        bulk = system.side_bulk[indices]
                        outward_flux = (
                            system.parameters.side_exchange_rate
                            * float(np.sum(weights * (values - bulk)))
                        )
                        payload["net_outward_exchange_flux"] = outward_flux
                    source_payload[species] = payload
                patch_payload["sources"][source_name] = source_payload
            report["side_patches"].append(patch_payload)
    return report


def _choose_evenly(indices: list[int], count: int) -> list[int]:
    if not indices or count <= 0:
        return []
    if count >= len(indices):
        return indices
    positions = np.linspace(0, len(indices) - 1, count)
    return [indices[int(round(position))] for position in positions]


def build_interface_target_manifest(
    system: ModelSystem,
    metadata: dict[str, Any],
    config: InterfaceParameters,
    *,
    max_targets: int = 0,
) -> list[dict[str, Any]]:
    """Select shallow point targets only on the nuclear face and sidewall.

    Point targets preserve the previously successful basis-transition hardware
    circuit.  Legendre/Fourier coefficients are fitted from these QPU-measured
    surface values after execution, avoiding an arbitrary controlled target-state
    preparation that would make the circuit too deep.
    """
    grid = system.grid
    target_map: dict[tuple[str, int], dict[str, Any]] = {}

    def register(species: str, cell: int, role: dict[str, Any]) -> None:
        key = (species, int(cell))
        if key not in target_map:
            physical = system.species_slices[species].start + int(cell)
            target_map[key] = {
                "species": species,
                "cell": int(cell),
                "physical_index": int(physical),
                "lifted_index": int(metadata["first_order_start"] + physical),
                "z": float(grid.z[cell]),
                "rho": float(grid.rho[cell]),
                "rho_metric": float(grid.rho_metric[cell]),
                "phi_deg": float(math.degrees(grid.phi[cell])),
                "r": float(grid.r[cell]),
                "radial_ring": int(grid.ring_index[cell]),
                "sector_index": int(grid.sector_index[cell]),
                "roles": [],
            }
        target_map[key]["roles"].append(role)

    if config.nuclear_enabled:
        nuclear = nuclear_layer_indices(grid).tolist()
        core = [cell for cell in nuclear if bool(grid.core_mask[cell])]
        noncore = sorted(
            [cell for cell in nuclear if not bool(grid.core_mask[cell])],
            key=lambda cell: (grid.ring_index[cell], grid.phi[cell]),
        )
        chosen: list[int] = core[:1]
        chosen.extend(_choose_evenly(noncore, max(config.nuclear_sample_count - len(chosen), 0)))
        for species in config.nuclear_qpu_species:
            for cell in dict.fromkeys(chosen):
                register(species, cell, {"surface": "nuclear_face"})

    if config.side_enabled:
        # Select equally spaced axial samples in every side patch.
        desired_z = np.linspace(0, system.parameters.nz - 1, config.side_samples_per_patch)
        for patch in range(config.side_patch_count):
            cells = side_patch_indices(grid, patch, config.side_patch_count).tolist()
            if not cells:
                continue
            by_layer: dict[int, list[int]] = {}
            for cell in cells:
                by_layer.setdefault(int(grid.z_index[cell]), []).append(cell)
            selected: list[int] = []
            for wanted in desired_z:
                layer = min(by_layer, key=lambda value: abs(value - wanted))
                candidates = by_layer[layer]
                _, _, center = patch_interval(patch, config.side_patch_count)
                cell = min(
                    candidates,
                    key=lambda value: abs(float(periodic_angle_difference(
                        np.asarray([grid.phi[value]]), center
                    )[0])),
                )
                selected.append(int(cell))
            for species in config.side_qpu_species:
                for sample_index, cell in enumerate(dict.fromkeys(selected)):
                    register(
                        species,
                        cell,
                        {
                            "surface": "sidewall",
                            "patch_id": patch_id(patch),
                            "patch_index": patch,
                            "sample_index": sample_index,
                        },
                    )

    targets = list(target_map.values())
    targets.sort(
        key=lambda item: (
            0 if any(role["surface"] == "nuclear_face" for role in item["roles"]) else 1,
            item["species"],
            item["z"],
            item["phi_deg"],
        )
    )
    if max_targets > 0:
        targets = targets[:max_targets]
    return targets


def add_qpu_coefficients(
    report: dict[str, Any],
    system: ModelSystem,
    qpu_targets: list[dict[str, Any]],
    config: InterfaceParameters,
) -> dict[str, Any]:
    """Fit interface coefficients from QPU-measured point concentrations."""
    lookup = {
        (str(item["species"]), int(item["cell"])): float(item["qpu_real"])
        for item in qpu_targets
    }
    grid = system.grid

    nuclear = report.get("nuclear_face")
    if nuclear:
        indices = np.asarray(nuclear["cell_indices"], dtype=int)
        for species in config.nuclear_qpu_species:
            available = [cell for cell in indices if (species, int(cell)) in lookup]
            if not available:
                continue
            cells = np.asarray(available, dtype=int)
            values = np.asarray([lookup[(species, int(cell))] for cell in cells], dtype=float)
            weights = grid.lower_z_area[cells]
            basis, labels = nuclear_basis(grid.rho_metric[cells], grid.phi[cells], config)
            if basis.shape[0] < basis.shape[1]:
                # Under-determined smoke tests still get sample values but no fake fit.
                payload = {
                    "sample_cell_indices": cells.tolist(),
                    "sample_values": values.tolist(),
                    "basis_labels": labels,
                    "coefficients": None,
                    "relative_fit_residual": None,
                    "status": "insufficient_qpu_samples",
                }
            else:
                coefficients, residual = _weighted_fit(basis, values, weights)
                payload = {
                    **_surface_statistics(values, weights),
                    "sample_cell_indices": cells.tolist(),
                    "sample_values": values.tolist(),
                    "basis_labels": labels,
                    "coefficients": coefficients.tolist(),
                    "relative_fit_residual": residual,
                    "status": "fitted",
                }
            nuclear.setdefault("sources", {}).setdefault("qpu", {})[species] = payload

        qpu_payload = nuclear.setdefault("sources", {}).setdefault("qpu", {})
        free_fit = qpu_payload.get("free", {})
        mobile_fit = qpu_payload.get("mobile", {})
        if (
            free_fit.get("coefficients") is not None
            and mobile_fit.get("coefficients") is not None
        ):
            full_indices = np.asarray(nuclear["cell_indices"], dtype=int)
            full_basis, _ = nuclear_basis(
                grid.rho_metric[full_indices], grid.phi[full_indices], config
            )
            free_reconstructed = full_basis @ np.asarray(
                free_fit["coefficients"], dtype=float
            )
            mobile_reconstructed = full_basis @ np.asarray(
                mobile_fit["coefficients"], dtype=float
            )
            full_weights = grid.lower_z_area[full_indices]
            qpu_payload["total_nuclear_capture_flux"] = float(
                system.parameters.nuclear_permeability_free
                * np.sum(full_weights * free_reconstructed)
                + system.parameters.nuclear_permeability_mobile
                * np.sum(full_weights * mobile_reconstructed)
            )

    patch_by_id = {item["patch_id"]: item for item in report.get("side_patches", [])}
    for patch_number in range(config.side_patch_count):
        pid = patch_id(patch_number)
        payload = patch_by_id.get(pid)
        if payload is None:
            continue
        patch_cells = set(int(value) for value in payload["cell_indices"])
        for species in config.side_qpu_species:
            available = sorted(
                cell for (sp, cell), value in lookup.items()
                if sp == species and cell in patch_cells
            )
            if not available:
                continue
            cells = np.asarray(available, dtype=int)
            values = np.asarray([lookup[(species, int(cell))] for cell in cells], dtype=float)
            weights = grid.outer_side_area[cells]
            basis, labels = side_basis(grid.z[cells], config.side_legendre_order)
            if basis.shape[0] < basis.shape[1]:
                fitted = {
                    "sample_cell_indices": cells.tolist(),
                    "sample_values": values.tolist(),
                    "basis_labels": labels,
                    "coefficients": None,
                    "relative_fit_residual": None,
                    "status": "insufficient_qpu_samples",
                }
            else:
                coefficients, residual = _weighted_fit(basis, values, weights)
                fitted = {
                    **_surface_statistics(values, weights),
                    "sample_cell_indices": cells.tolist(),
                    "sample_values": values.tolist(),
                    "basis_labels": labels,
                    "coefficients": coefficients.tolist(),
                    "relative_fit_residual": residual,
                    "status": "fitted",
                }
            payload.setdefault("sources", {}).setdefault("qpu", {})[species] = fitted
    return report


def compact_to_dense(values: np.ndarray, grid: Grid3D, fill: float = np.nan) -> np.ndarray:
    dense = np.full(grid.shape, fill, dtype=float)
    for cell in range(grid.z.size):
        dense[
            int(grid.z_index[cell]),
            int(grid.ring_index[cell]),
            int(grid.sector_index[cell]),
        ] = float(values[cell])
    return dense


def dense_to_compact(values: np.ndarray, grid: Grid3D) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.shape != grid.shape:
        raise ValueError(f"Dense state shape {values.shape} != expected {grid.shape}")
    compact = np.empty(grid.z.size, dtype=float)
    for cell in range(grid.z.size):
        compact[cell] = values[
            int(grid.z_index[cell]),
            int(grid.ring_index[cell]),
            int(grid.sector_index[cell]),
        ]
    return compact


def save_final_state(
    path: Path,
    system: ModelSystem,
    nonlinear: np.ndarray,
    carleman: np.ndarray,
    *,
    time_start: float,
    time_end: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, Any] = {
        "time_start": np.asarray(time_start),
        "time_end": np.asarray(time_end),
        "ring_counts": np.asarray(system.grid.ring_counts, dtype=int),
        "z_index": system.grid.z_index,
        "ring_index": system.grid.ring_index,
        "sector_index": system.grid.sector_index,
    }
    for species, species_slice in system.species_slices.items():
        values = np.asarray(nonlinear[species_slice], dtype=float)
        approximation = np.asarray(carleman[species_slice], dtype=float)
        # The exact protocol keys remain free/stationary/mobile.  They are flat
        # compact arrays because the centre ring is intentionally collapsed.
        arrays[species] = values
        arrays[f"{species}_carleman"] = approximation
        arrays[f"{species}_dense"] = compact_to_dense(values, system.grid)
    np.savez_compressed(path, **arrays)


def load_initial_state(path: str | Path, system: ModelSystem) -> np.ndarray:
    path = Path(path)
    n = system.grid.z.size
    pieces: list[np.ndarray] = []
    with np.load(path, allow_pickle=False) as data:
        for species in ("free", "stationary", "mobile"):
            if species not in data:
                raise ValueError(f"{path} is missing NPZ key: {species}")
            value = np.asarray(data[species], dtype=float)
            if value.shape == (n,):
                compact = value
            elif value.shape == system.grid.shape:
                compact = dense_to_compact(value, system.grid)
            else:
                dense_key = f"{species}_dense"
                if dense_key in data and np.asarray(data[dense_key]).shape == system.grid.shape:
                    compact = dense_to_compact(np.asarray(data[dense_key], dtype=float), system.grid)
                else:
                    raise ValueError(
                        f"{species}: shape {value.shape}; expected compact {(n,)} "
                        f"or dense {system.grid.shape}"
                    )
            if not np.all(np.isfinite(compact)):
                raise ValueError(f"{species} contains non-finite values")
            pieces.append(compact)
    return np.concatenate(pieces)


def restrict_to_lateral_initial(
    state: np.ndarray,
    system: ModelSystem,
    config: InterfaceParameters,
) -> np.ndarray:
    """Force an imported neighbour field to remain a lateral initial seed.

    The generated neighbour files are already side-localised.  This additional
    validation prevents an arbitrary NPZ from silently modifying the plasma-
    membrane input face or deep interior cells.
    """
    state = np.asarray(state, dtype=float).copy()
    n = system.grid.z.size
    if state.shape != (3 * n,):
        raise ValueError(f"side initial state shape {state.shape} != {(3*n,)}")

    cutoff = max(0.0, 1.0 - config.side_initial_radial_cutoff_sigma * config.neighbor_radial_depth)
    allowed = system.grid.rho_metric >= cutoff
    if config.side_initial_exclude_membrane_face:
        allowed[membrane_input_layer_indices(system.grid)] = False

    for species, sl in system.species_slices.items():
        values = state[sl]
        values[~allowed] = 0.0
        if species not in config.neighbor_transfer_species:
            values[:] = 0.0
        values[values < 0.0] = 0.0
    return state


def write_initial_component_npz(
    path: Path,
    system: ModelSystem,
    membrane_initial_state: np.ndarray,
    side_initial_state: np.ndarray,
    combined_initial_state: np.ndarray,
) -> None:
    """Save membrane, side and combined initial fields as separate arrays."""
    n = system.grid.z.size
    arrays: dict[str, np.ndarray] = {
        "ring_counts": np.asarray(system.grid.ring_counts, dtype=int),
        "z_index": system.grid.z_index,
        "ring_index": system.grid.ring_index,
        "sector_index": system.grid.sector_index,
    }
    for label, state in (
        ("membrane", membrane_initial_state),
        ("side", side_initial_state),
        ("combined", combined_initial_state),
    ):
        for species, sl in system.species_slices.items():
            arrays[f"{label}_{species}"] = np.asarray(state[sl], dtype=float)
    np.savez_compressed(path, **arrays)


def write_surface_csvs(
    output_dir: Path,
    system: ModelSystem,
    fields: dict[str, np.ndarray],
    config: InterfaceParameters,
    *,
    membrane_initial_state: np.ndarray | None = None,
    side_initial_state: np.ndarray | None = None,
) -> None:
    """Write the nuclear result face, plasma-membrane input face, and sidewall."""
    output_dir.mkdir(parents=True, exist_ok=True)
    grid = system.grid

    def write_result_surface(path: Path, cells: Iterable[int], surface: str) -> None:
        cells = list(int(value) for value in cells)
        with path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.writer(handle)
            header = [
                "surface", "patch_id", "cell", "z", "rho", "phi_deg", "area",
                "side_bulk",
            ]
            for source in fields:
                for species in ("free", "stationary", "mobile"):
                    header.append(f"{source}_{species}")
            writer.writerow(header)
            for cell in cells:
                if surface == "nuclear_face":
                    pid = "NUCLEAR_FACE"
                    area = grid.lower_z_area[cell]
                else:
                    pid = patch_id(
                        patch_index_from_phi(grid.phi[cell], config.side_patch_count)
                    )
                    area = grid.outer_side_area[cell]
                row = [
                    surface, pid, cell, grid.z[cell], grid.rho[cell],
                    math.degrees(grid.phi[cell]), area, system.side_bulk[cell],
                ]
                for state in fields.values():
                    for species in ("free", "stationary", "mobile"):
                        row.append(state[system.species_slices[species]][cell])
                writer.writerow(row)

    write_result_surface(
        output_dir / "nuclear_face.csv", nuclear_layer_indices(grid), "nuclear_face"
    )
    write_result_surface(
        output_dir / "sidewall_surface.csv", sidewall_indices(grid), "sidewall"
    )

    # Export the membrane input component only.  Neighbour-derived side input
    # is written to side_initial_input.csv below.
    membrane_state = (
        np.asarray(membrane_initial_state, dtype=float)
        if membrane_initial_state is not None
        else np.asarray(system.initial_state, dtype=float)
    )
    cells = membrane_input_layer_indices(grid).tolist()
    with (output_dir / "membrane_input.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "surface", "role", "cell", "z", "rho", "phi_deg", "area",
            "initial_free", "initial_stationary", "initial_mobile",
        ])
        for cell in cells:
            writer.writerow([
                "plasma_membrane", "input", cell, grid.z[cell], grid.rho[cell],
                math.degrees(grid.phi[cell]), grid.upper_z_area[cell],
                membrane_state[system.species_slices["free"]][cell],
                membrane_state[system.species_slices["stationary"]][cell],
                membrane_state[system.species_slices["mobile"]][cell],
            ])

    if side_initial_state is not None:
        side_state = np.asarray(side_initial_state, dtype=float)
        active = np.flatnonzero(
            np.any(
                np.vstack([
                    np.abs(side_state[system.species_slices[name]]) > 0.0
                    for name in ("free", "stationary", "mobile")
                ]),
                axis=0,
            )
        )
        with (output_dir / "side_initial_input.csv").open(
            "w", newline="", encoding="utf-8-sig"
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow([
                "surface", "role", "patch_id", "cell", "z", "rho",
                "phi_deg", "side_initial_free",
                "side_initial_stationary", "side_initial_mobile",
            ])
            for cell in active.tolist():
                writer.writerow([
                    "lateral_shell", "neighbour_initial_condition",
                    patch_id(patch_index_from_phi(grid.phi[cell], config.side_patch_count)),
                    cell, grid.z[cell], grid.rho[cell], math.degrees(grid.phi[cell]),
                    side_state[system.species_slices["free"]][cell],
                    side_state[system.species_slices["stationary"]][cell],
                    side_state[system.species_slices["mobile"]][cell],
                ])


def _pick_coefficients(
    patch_payload: dict[str, Any],
    species: str,
    prefer_qpu: bool,
) -> tuple[list[float] | None, str | None]:
    sources = patch_payload.get("sources", {})
    order = ["qpu", "carleman", "nonlinear"] if prefer_qpu else ["carleman", "nonlinear", "qpu"]
    for source in order:
        payload = sources.get(source, {}).get(species)
        if payload and payload.get("coefficients") is not None:
            return list(payload["coefficients"]), source
    return None, None


def build_neighbor_state(
    system: ModelSystem,
    interface_report: dict[str, Any],
    config: InterfaceParameters,
    *,
    source_patch_id: str,
    destination_patch_index: int | None = None,
    scale: float | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Reconstruct a neighbour-cone seed from one lateral interface patch."""
    source_patch = next(
        (item for item in interface_report.get("side_patches", []) if item["patch_id"] == source_patch_id),
        None,
    )
    if source_patch is None:
        raise ValueError(f"Unknown side patch: {source_patch_id}")
    if destination_patch_index is None:
        # Opposite-facing side on the neighbour is a useful deterministic default.
        destination_patch_index = (
            int(source_patch["patch_index"]) + config.side_patch_count // 2
        ) % config.side_patch_count
    if not (0 <= destination_patch_index < config.side_patch_count):
        raise ValueError("destination_patch_index outside valid range")

    grid = system.grid
    n = grid.z.size
    result = {
        "free": np.zeros(n, dtype=float),
        "stationary": np.zeros(n, dtype=float),
        "mobile": np.zeros(n, dtype=float),
    }
    _, _, destination_center = patch_interval(destination_patch_index, config.side_patch_count)
    radial_depth = max(config.neighbor_radial_depth, 1e-6)
    angular_sigma = math.radians(max(config.neighbor_azimuth_sigma_deg, 1e-6))
    radial_weight = np.exp(-0.5 * ((1.0 - grid.rho_metric) / radial_depth) ** 2)
    angular_distance = periodic_angle_difference(grid.phi, destination_center)
    angular_weight = np.exp(-0.5 * (angular_distance / angular_sigma) ** 2)
    localiser = radial_weight * angular_weight
    applied_scale = config.handoff_scale if scale is None else float(scale)

    sources_used: dict[str, str] = {}
    for species in config.neighbor_transfer_species:
        coefficients, source = _pick_coefficients(
            source_patch, species, config.prefer_qpu_coefficients
        )
        if coefficients is None:
            continue
        profile = legval(2.0 * grid.z - 1.0, np.asarray(coefficients, dtype=float))
        result[species] = np.maximum(applied_scale * profile * localiser, 0.0)
        sources_used[species] = str(source)

    metadata = {
        "format_version": "cone-neighbor-state-v2",
        "source_patch_id": source_patch_id,
        "destination_patch_id": patch_id(destination_patch_index),
        "destination_patch_index": destination_patch_index,
        "coefficient_sources": sources_used,
        "handoff_scale": applied_scale,
        "neighbor_radial_depth": config.neighbor_radial_depth,
        "neighbor_azimuth_sigma_deg": config.neighbor_azimuth_sigma_deg,
    }
    return np.concatenate([result["free"], result["stationary"], result["mobile"]]), metadata


def save_neighbor_state(
    path: Path,
    system: ModelSystem,
    state: np.ndarray,
    metadata: dict[str, Any],
) -> None:
    n = system.grid.z.size
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        free=np.asarray(state[:n], dtype=float),
        stationary=np.asarray(state[n:2*n], dtype=float),
        mobile=np.asarray(state[2*n:3*n], dtype=float),
        ring_counts=np.asarray(system.grid.ring_counts, dtype=int),
        z_index=system.grid.z_index,
        ring_index=system.grid.ring_index,
        sector_index=system.grid.sector_index,
    )
    path.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def export_all_neighbor_states(
    output_dir: Path,
    system: ModelSystem,
    interface_report: dict[str, Any],
    config: InterfaceParameters,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not config.export_neighbor_states:
        return records
    folder = output_dir / "neighbor_inputs"
    folder.mkdir(parents=True, exist_ok=True)
    for patch in interface_report.get("side_patches", []):
        state, metadata = build_neighbor_state(
            system,
            interface_report,
            config,
            source_patch_id=str(patch["patch_id"]),
        )
        path = folder / f"from_{patch['patch_id']}_to_{metadata['destination_patch_id']}.npz"
        save_neighbor_state(path, system, state, metadata)
        records.append({"path": str(path), **metadata})
    (folder / "neighbor_input_manifest.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return records
