"""Apply resolved external surface inputs to the compact cone model.

This is deliberately separate from neighbour handoff.  Direct experimental or
drug inputs may act on the plasma-membrane end, the lateral sidewall, or (when
explicitly requested) the nuclear end.  Neighbour states are handled by the
runner's --side-input-state option and can never redefine the membrane input.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any
import json
import math

import numpy as np
from scipy.sparse import csr_matrix

from cone3d_model import ModelSystem, periodic_angle_difference


def _gaussian(values: np.ndarray, center: float, sigma: float) -> np.ndarray:
    if sigma <= 0.0:
        return np.isclose(values, center).astype(float)
    return np.exp(-0.5 * ((values - center) / sigma) ** 2)


def _surface_shape(system: ModelSystem, event: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, str]:
    grid = system.grid
    surface = str(event.get("surface", "sidewall")).lower()
    axisymmetric = bool(event.get("axisymmetric", False))
    center_phi = math.radians(float(event.get("center_phi_deg", 0.0)))
    sigma_phi = math.radians(float(event.get("sigma_phi_deg", 360.0)))
    if axisymmetric or sigma_phi >= 2.0 * math.pi:
        phi_part = np.ones_like(grid.phi)
    else:
        delta = periodic_angle_difference(grid.phi, center_phi)
        phi_part = np.exp(-0.5 * (delta / max(sigma_phi, 1e-9)) ** 2)

    if surface == "sidewall":
        mask = grid.ring_index == len(grid.ring_counts) - 1
        z_part = _gaussian(
            grid.z,
            float(event.get("center_z", 0.5)),
            float(event.get("sigma_z", 1.0)),
        )
        shape = z_part * phi_part * mask
        area = grid.outer_side_area
        canonical = "sidewall"
    elif surface in {"cell_membrane", "membrane_end", "z1", "plasma_membrane"}:
        mask = grid.z_index == int(np.max(grid.z_index))
        rho_part = _gaussian(
            grid.rho_metric,
            float(event.get("center_rho", 0.5)),
            float(event.get("sigma_rho", 1.0)),
        )
        shape = rho_part * phi_part * mask
        area = grid.upper_z_area
        canonical = "cell_membrane"
    elif surface in {"nuclear_membrane", "nuclear_end", "z0", "nuclear_face"}:
        mask = grid.z_index == int(np.min(grid.z_index))
        rho_part = _gaussian(
            grid.rho_metric,
            float(event.get("center_rho", 0.5)),
            float(event.get("sigma_rho", 1.0)),
        )
        shape = rho_part * phi_part * mask
        area = grid.lower_z_area
        canonical = "nuclear_membrane"
    else:
        raise ValueError(f"Unsupported surface input: {surface}")

    peak = float(np.max(shape)) if shape.size else 0.0
    if peak > 0.0:
        shape = shape / peak
    return np.asarray(shape, dtype=float), np.asarray(area, dtype=float), canonical


def apply_surface_inputs(
    system: ModelSystem,
    input_path: str | Path | None,
    *,
    time_start: float,
    time_end: float,
) -> tuple[ModelSystem, dict[str, np.ndarray], list[dict[str, Any]]]:
    """Return a model modified by resolved initial/source surface events."""
    nstate = system.initial_state.size
    components = {
        "cell_membrane": np.zeros(nstate, dtype=float),
        "sidewall": np.zeros(nstate, dtype=float),
        "nuclear_membrane": np.zeros(nstate, dtype=float),
    }
    if input_path is None:
        return system, components, []

    path = Path(input_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    events = list(payload.get("resolved_surface_inputs", payload.get("surface_inputs", [])))
    linear = system.linear_matrix.tolil(copy=True)
    source = np.asarray(system.source_vector, dtype=float).copy()
    initial = np.asarray(system.initial_state, dtype=float).copy()
    window_duration = max(float(time_end - time_start), 1e-12)
    report: list[dict[str, Any]] = []

    for event in events:
        if not bool(event.get("enabled", True)):
            continue
        shape, area, surface = _surface_shape(system, event)
        mode = str(event.get("mode", "source")).lower()
        if mode not in {"initial", "source", "both"}:
            raise ValueError(f"Unsupported input mode: {mode}")
        concentrations = dict(event.get("concentrations", {}))
        exchange_rate = float(event.get("exchange_rate", system.parameters.side_exchange_rate))
        duration = float(event.get("duration", window_duration))
        duty = float(np.clip(duration / window_duration, 0.0, 1.0))
        initial_fraction = float(np.clip(event.get("initial_fraction", 0.5), 0.0, 1.0))
        entry = {
            "input_id": str(event.get("input_id", "unnamed")),
            "surface": surface,
            "mode": mode,
            "exchange_rate": exchange_rate,
            "duration": duration,
            "duty_fraction": duty,
            "support_cells": int(np.count_nonzero(shape > 1e-10)),
            "species": {},
        }

        for species, raw_concentration in concentrations.items():
            if species not in system.species_slices:
                raise ValueError(f"Unknown input species: {species}")
            concentration = float(raw_concentration)
            if concentration == 0.0:
                continue
            sl = system.species_slices[species]
            offset = int(sl.start)
            deposited_mass = 0.0
            source_amount = 0.0

            if mode in {"initial", "both"}:
                fraction = 1.0 if mode == "initial" else initial_fraction
                addition = concentration * fraction * shape
                initial[sl] += addition
                components[surface][sl] += addition
                deposited_mass = float(np.sum(addition * system.grid.volume))

            if mode in {"source", "both"}:
                fraction = 1.0 if mode == "source" else 1.0 - initial_fraction
                coefficient = (
                    exchange_rate * duty * fraction * shape * area
                    / np.maximum(system.grid.volume, 1e-15)
                )
                for cell in np.flatnonzero(coefficient > 0.0):
                    physical = offset + int(cell)
                    value = float(coefficient[cell])
                    linear[physical, physical] -= value
                    source[physical] += value * concentration
                source_amount = float(
                    np.sum(exchange_rate * duty * fraction * shape * area * concentration)
                )

            entry["species"][species] = {
                "concentration": concentration,
                "initial_deposited_mass": deposited_mass,
                "nominal_source_amount_per_time": source_amount,
            }
        report.append(entry)

    initial = np.maximum(initial, 0.0)
    modified = replace(
        system,
        linear_matrix=csr_matrix(linear),
        source_vector=source,
        initial_state=initial,
    )
    return modified, components, report
