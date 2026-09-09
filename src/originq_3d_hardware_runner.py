"""
OriginQ real-hardware runner for the compact true-3-D conical-frustum cell model.

The plasma-membrane end (z=1) is the input end. The nuclear-envelope end
(z=0) is the biological result-output face; lateral patches are exported as
initial conditions for neighbouring cones.

This program preserves all biological mechanisms and uses the shallow
basis-transition Hadamard test that previously compiled successfully on the
OriginQ real backend.  It never embeds a dense arbitrary Oracle gate.

Environment variable:
    QPANDA_QCLOUD_API_KEY=<your OriginQ API token>

Examples:
    python src/originq_3d_hardware_runner.py --config configs/fig6/parameters_1.json --dry-run
    python src/originq_3d_hardware_runner.py --config configs/fig6/parameters_1.json --max-targets 1 --parallel-jobs 8 --skip-preflight
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from itertools import combinations, product
from pathlib import Path
from typing import Any, Iterable
import argparse
import csv
import json
import math
import os
import sys
import threading
import time

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.sparse import csr_matrix, eye
from scipy.sparse.linalg import eigsh

from cone3d_model import (
    ModelSystem,
    build_model_system,
    build_selected_carleman,
    load_parameters,
    nuclear_flux,
    nuclear_mean,
    parameter_summary,
    relative_error,
    solve_nonlinear_reference,
    solve_selected_carleman,
    write_classical_csv,
)
from layer_surface_inputs import apply_surface_inputs

from cone_interfaces import (
    add_qpu_coefficients,
    build_interface_target_manifest,
    compute_classical_interface_report,
    export_all_neighbor_states,
    load_initial_state,
    load_interface_parameters,
    restrict_to_lateral_initial,
    save_final_state,
    write_initial_component_npz,
    write_surface_csvs,
)


# ---------------------------------------------------------------------------
# Hardware and LCHS parameters
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HardwareParameters:
    backend: str = "auto"
    shots: int = 500
    batch_size: int = 1
    parallel_jobs: int = 8

    lchs_beta: float = 0.65
    lchs_half_width: float = 5.0
    lchs_nodes: int = 2

    pauli_threshold: float = 1e-10
    max_pauli_terms: int = 2
    max_pauli_weight: int = 2
    rotation_angle_threshold: float = 1e-3

    max_initial_components: int = 2
    initial_component_threshold: float = 1e-8

    trotter_order: int = 1
    trotter_steps: int = 1

    mapping: bool = True
    optimization: bool = True
    amend: bool = False

    run_preflight_bell: bool = False
    preflight_shots: int = 200
    # highest_reference avoids selecting the low-signal collapsed axis core.
    # Alternatives: axis, manual_first.
    target_strategy: str = "highest_reference"
    output_dir: str = "originq_3d_compact_output"


@dataclass(frozen=True)
class LCHSData:
    shift: float
    loss: csr_matrix
    rotation: csr_matrix
    k_nodes: np.ndarray
    coefficients: np.ndarray


@dataclass(frozen=True)
class PauliTerm:
    coefficient: float
    xmask: int
    zmask: int
    weight: int
    label: str


@dataclass(frozen=True)
class CircuitRecord:
    program: Any
    lchs_index: int
    target_position: int
    initial_position: int
    component: str
    target_lifted_index: int
    initial_lifted_index: int


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def load_hardware(raw: dict[str, Any]) -> HardwareParameters:
    data = dict(raw.get("hardware", {}))
    valid = set(HardwareParameters.__dataclass_fields__)
    unknown = sorted(set(data) - valid)
    if unknown:
        raise ValueError(f"Unknown hardware parameters: {unknown}")
    return HardwareParameters(**data)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "True-3D conical-cell OriginQ runner focused on nuclear-envelope results "
            "and lateral neighbour interfaces"
        )
    )
    parser.add_argument("--config", default="configs/fig6/parameters_1.json")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--classical-only", action="store_true")
    parser.add_argument("--skip-preflight", action="store_true")
    # 0 means all configured surface samples.  Use 1 for a cheap smoke test.
    parser.add_argument("--max-targets", type=int, default=0)
    parser.add_argument("--parallel-jobs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--shots", type=int)
    parser.add_argument("--backend")
    parser.add_argument("--max-pauli-terms", type=int)
    parser.add_argument("--max-pauli-weight", type=int)
    parser.add_argument("--max-initial-components", type=int)

    # Small file protocol for time-window continuation and cone-to-cone coupling.
    # --input-state is a complete state of the same cone and replaces the base
    # state.  --side-input-state is a neighbour-derived lateral seed and is
    # added only to the side-initial component.
    parser.add_argument(
        "--input-state", default=None,
        help="Complete same-cone state for time-window continuation (replacement).",
    )
    parser.add_argument(
        "--side-input-state", action="append", default=[],
        help=(
            "Neighbour-derived lateral initial NPZ. May be supplied more than "
            "once; every file is added to the side initial condition, not to "
            "the plasma-membrane input."
        ),
    )
    parser.add_argument("--side-input-scale", type=float, default=1.0)
    parser.add_argument(
        "--surface-inputs", default=None,
        help="Resolved direct surface-input JSON for this cone. Kept separate from neighbour handoff.",
    )
    # Backward compatibility with the immediately preceding package.
    parser.add_argument(
        "--input-state-mode", choices=("replace", "add"), default="replace",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--time-start", type=float, default=0.0)
    parser.add_argument("--time-end", type=float, default=None)

    if "ipykernel" in sys.modules or "ipykernel_launcher" in Path(sys.argv[0]).name:
        return parser.parse_known_args()[0]
    return parser.parse_args()


def override_hardware(h: HardwareParameters, args: argparse.Namespace) -> HardwareParameters:
    data = dict(h.__dict__)
    for name, argname in [
        ("parallel_jobs", "parallel_jobs"),
        ("batch_size", "batch_size"),
        ("shots", "shots"),
        ("backend", "backend"),
        ("max_pauli_terms", "max_pauli_terms"),
        ("max_pauli_weight", "max_pauli_weight"),
        ("max_initial_components", "max_initial_components"),
    ]:
        value = getattr(args, argname)
        if value is not None:
            data[name] = value
    return HardwareParameters(**data)


# ---------------------------------------------------------------------------
# LCHS and low-weight Pauli expansion
# ---------------------------------------------------------------------------


def prepare_lchs(matrix: csr_matrix, h: HardwareParameters) -> LCHSData:
    hermitian_part = 0.5 * (matrix + matrix.getH())
    try:
        largest = float(np.real(eigsh(hermitian_part, k=1, which="LA", return_eigenvectors=False)[0]))
    except Exception:
        # Conservative Gershgorin-style fallback.
        largest = float(np.max(np.asarray(np.abs(hermitian_part).sum(axis=1)).ravel()))
    shift = largest + max(1e-6, 0.02 * max(1.0, abs(largest)))
    a = shift * eye(matrix.shape[0], dtype=np.complex128, format="csr") - matrix
    loss = 0.5 * (a + a.getH())
    rotation = (a - a.getH()) / (2.0j)

    nodes, weights = leggauss(h.lchs_nodes)
    k_nodes = h.lchs_half_width * nodes
    scaled_weights = h.lchs_half_width * weights
    beta = h.lchs_beta
    fbeta = np.exp(2.0**beta - (1.0 + 1.0j * k_nodes) ** beta) / (2.0 * math.pi)
    coefficients = scaled_weights * fbeta / (1.0 - 1.0j * k_nodes)
    return LCHSData(
        shift=shift,
        loss=loss.tocsr(),
        rotation=rotation.tocsr(),
        k_nodes=k_nodes,
        coefficients=coefficients,
    )


def _pauli_label(q: int, xmask: int, zmask: int) -> str:
    labels: list[str] = []
    for bit in range(q):
        x = (xmask >> bit) & 1
        z = (zmask >> bit) & 1
        if x and z:
            labels.append(f"Y{bit}")
        elif x:
            labels.append(f"X{bit}")
        elif z:
            labels.append(f"Z{bit}")
    return "I" if not labels else " ".join(labels)


def enumerate_low_weight_paulis(q: int, max_weight: int) -> Iterable[tuple[int, int, int]]:
    yield 0, 0, 0
    for weight in range(1, max_weight + 1):
        for qubits in combinations(range(q), weight):
            for letters in product("XYZ", repeat=weight):
                xmask = 0
                zmask = 0
                for qb, letter in zip(qubits, letters, strict=True):
                    if letter in "XY":
                        xmask |= 1 << qb
                    if letter in "YZ":
                        zmask |= 1 << qb
                yield xmask, zmask, weight


def pauli_coefficient_sparse(
    entries: dict[tuple[int, int], complex],
    dimension: int,
    padded_dim: int,
    xmask: int,
    zmask: int,
) -> complex:
    """Return Tr(P H)/2^q without constructing the padded dense matrix."""
    y_count = (xmask & zmask).bit_count()
    prefactor = (1.0j) ** y_count
    total = 0.0j
    for k in range(dimension):
        partner = k ^ xmask
        if partner >= dimension:
            continue
        parity = (zmask & k).bit_count() & 1
        phase = prefactor * (-1.0 if parity else 1.0)
        value = entries.get((k, partner), 0.0j)
        total += phase * value
    return total / padded_dim


def select_pauli_terms(
    matrix: csr_matrix,
    data_qubits: int,
    h: HardwareParameters,
) -> tuple[list[PauliTerm], dict[str, float]]:
    padded_dim = 1 << data_qubits
    coo = matrix.tocoo()
    entries = {
        (int(row), int(col)): complex(value)
        for row, col, value in zip(coo.row, coo.col, coo.data, strict=True)
    }
    candidates: list[PauliTerm] = []
    candidate_norm2 = 0.0
    for xmask, zmask, weight in enumerate_low_weight_paulis(data_qubits, h.max_pauli_weight):
        coefficient = pauli_coefficient_sparse(
            entries, matrix.shape[0], padded_dim, xmask, zmask
        )
        if abs(coefficient.imag) > 2e-7:
            raise RuntimeError(f"Hermitian Pauli coefficient has large imaginary part: {coefficient}")
        value = float(coefficient.real)
        candidate_norm2 += value * value
        if abs(value) >= h.pauli_threshold:
            candidates.append(
                PauliTerm(value, xmask, zmask, weight, _pauli_label(data_qubits, xmask, zmask))
            )
    candidates.sort(key=lambda term: abs(term.coefficient), reverse=True)
    if h.max_pauli_terms >= 0:
        selected = candidates[: h.max_pauli_terms]
    else:
        selected = candidates
    retained_norm2 = sum(term.coefficient**2 for term in selected)
    frobenius = float(np.sum(np.abs(matrix.data) ** 2)) / padded_dim
    return selected, {
        "candidate_count": float(len(candidates)),
        "candidate_l2_coverage": candidate_norm2 / frobenius if frobenius else 1.0,
        "retained_l2_coverage": retained_norm2 / frobenius if frobenius else 1.0,
        "frobenius_pauli_norm2": frobenius,
    }


# ---------------------------------------------------------------------------
# Shallow transition-amplitude circuits
# ---------------------------------------------------------------------------


def _bits(integer: int, qubits: int) -> list[int]:
    return [bit for bit in range(qubits) if (integer >> bit) & 1]


def controlled_pauli_rotation(circuit: Any, term: PauliTerm, theta: float, ancilla: int) -> None:
    """Append controlled exp(-i theta P), controlling only the central RZ/phase."""
    from pyqpanda3.core import H, S, RZ, CNOT

    if abs(theta) == 0.0:
        return
    active = [bit for bit in range(max(term.xmask.bit_length(), term.zmask.bit_length())) if ((term.xmask | term.zmask) >> bit) & 1]
    if not active:
        # Relative phase e^{-i theta} on the ancilla |1> branch.
        circuit << RZ(ancilla, -theta)
        return

    for q in active:
        x = (term.xmask >> q) & 1
        z = (term.zmask >> q) & 1
        if x and z:  # Y -> Z basis
            circuit << S(q).dagger()
            circuit << H(q)
        elif x:  # X -> Z basis
            circuit << H(q)
    target = active[-1]
    for q in active[:-1]:
        circuit << CNOT(q, target)
    circuit << RZ(target, 2.0 * theta).control([ancilla])
    for q in reversed(active[:-1]):
        circuit << CNOT(q, target)
    for q in reversed(active):
        x = (term.xmask >> q) & 1
        z = (term.zmask >> q) & 1
        if x and z:
            circuit << H(q)
            circuit << S(q)
        elif x:
            circuit << H(q)


def append_controlled_trotter(
    circuit: Any,
    terms: list[PauliTerm],
    evolution_time: float,
    ancilla: int,
    h: HardwareParameters,
) -> None:
    if h.trotter_steps < 1:
        raise ValueError("trotter_steps must be at least 1.")
    dt = evolution_time / h.trotter_steps
    if h.trotter_order == 1:
        schedule = [(term, dt) for term in terms]
    elif h.trotter_order == 2:
        schedule = [(term, 0.5 * dt) for term in terms]
        schedule += [(term, 0.5 * dt) for term in reversed(terms)]
    else:
        raise ValueError("Only first- and second-order Trotter are supported.")
    for _ in range(h.trotter_steps):
        for term, scale in schedule:
            theta = term.coefficient * scale
            if abs(theta) < h.rotation_angle_threshold:
                continue
            controlled_pauli_rotation(circuit, term, theta, ancilla)


def build_transition_program(
    data_qubits: int,
    initial_index: int,
    target_index: int,
    terms: list[PauliTerm],
    evolution_time: float,
    component: str,
    h: HardwareParameters,
) -> Any:
    from pyqpanda3.core import QCircuit, QProg, H, S, X, CNOT, measure

    ancilla = data_qubits
    circuit = QCircuit()
    circuit << H(ancilla)

    # Prepare (|0>|target> + |1>|initial>)/sqrt(2) with only X/CNOT gates.
    for q in _bits(target_index, data_qubits):
        circuit << X(q)
    for q in _bits(initial_index ^ target_index, data_qubits):
        circuit << CNOT(ancilla, q)

    append_controlled_trotter(circuit, terms, evolution_time, ancilla, h)

    if component == "real":
        circuit << H(ancilla)
    elif component == "imag":
        circuit << S(ancilla).dagger()
        circuit << H(ancilla)
    else:
        raise ValueError(component)
    program = QProg()
    program << circuit
    program << measure(ancilla, 0)
    return program


def probability_expectation(probs: dict[str, float]) -> float:
    p0 = 0.0
    p1 = 0.0
    for key, value in probs.items():
        cleaned = str(key).lower().replace("0x", "")
        try:
            bit = int(cleaned, 16) & 1 if "0x" in str(key).lower() else int(cleaned, 2) & 1
        except ValueError:
            bit = int(float(cleaned)) & 1
        if bit == 0:
            p0 += float(value)
        else:
            p1 += float(value)
    total = p0 + p1
    if total <= 0.0:
        raise RuntimeError(f"Invalid probability result: {probs}")
    return (p0 - p1) / total


# ---------------------------------------------------------------------------
# OriginQ cloud execution
# ---------------------------------------------------------------------------


def create_cloud_backend(h: HardwareParameters):
    from pyqpanda3.qcloud import QCloudService

    api_key = os.environ.get("QPANDA_QCLOUD_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "Environment variable QPANDA_QCLOUD_API_KEY is missing. "
            "Set it before running the real-hardware step."
        )
    service = QCloudService(api_key=api_key)
    backends = service.backends()
    if h.backend != "auto":
        if not backends.get(h.backend, False):
            raise RuntimeError(f"Requested backend {h.backend!r} is unavailable: {backends}")
        name = h.backend
    else:
        preferred = ["WK_C180", "WK_C102_400", "72"]
        name = next((candidate for candidate in preferred if backends.get(candidate, False)), None)
        if name is None:
            name = next(
                (key for key, available in backends.items() if available and "amplitude" not in key),
                None,
            )
        if name is None:
            raise RuntimeError(f"No available physical backend: {backends}")
    return service, service.backend(name), name, backends


def cloud_options(h: HardwareParameters):
    from pyqpanda3.qcloud import QCloudOptions

    options = QCloudOptions()
    options.set_mapping(h.mapping)
    options.set_optimization(h.optimization)
    options.set_amend(h.amend)
    return options


def run_preflight_bell(backend: Any, h: HardwareParameters) -> dict[str, float]:
    from pyqpanda3.core import H, CNOT, QProg, measure

    prog = QProg()
    prog << H(0) << CNOT(0, 1) << measure(0, 0) << measure(1, 1)
    result = backend.run(prog, h.preflight_shots, cloud_options(h)).result()
    return result.get_probs()


def _submit_batch(
    backend: Any,
    programs: list[Any],
    h: HardwareParameters,
    batch_number: int,
    job_log: list[dict[str, Any]],
    lock: threading.Lock,
) -> list[float]:
    job = backend.run(programs, h.shots, cloud_options(h))
    job_id = job.job_id()
    with lock:
        print(f"Submitted cloud batch {batch_number}: programs={len(programs)}, job={job_id}")
        job_log.append({"batch": batch_number, "program_count": len(programs), "job_id": job_id})
    result = job.result()
    if hasattr(result, "error_message"):
        message = result.error_message()
        if message:
            raise RuntimeError(f"Cloud batch {job_id} failed: {message}")
    probs_list = result.get_probs_list()
    return [probability_expectation(probs) for probs in probs_list]


def submit_programs_parallel(
    backend: Any,
    records: list[CircuitRecord],
    h: HardwareParameters,
    output_dir: Path,
) -> list[float]:
    batches = [records[i : i + h.batch_size] for i in range(0, len(records), h.batch_size)]
    answers: list[list[float] | None] = [None] * len(batches)
    job_log: list[dict[str, Any]] = []
    lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=h.parallel_jobs) as pool:
        future_to_index = {
            pool.submit(
                _submit_batch,
                backend,
                [record.program for record in batch],
                h,
                index + 1,
                job_log,
                lock,
            ): index
            for index, batch in enumerate(batches)
        }
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            answers[index] = future.result()
            (output_dir / "job_ids.json").write_text(
                json.dumps(sorted(job_log, key=lambda item: item["batch"]), indent=2),
                encoding="utf-8",
            )
    flattened: list[float] = []
    for answer in answers:
        if answer is None:
            raise RuntimeError("A cloud batch did not return a result.")
        flattened.extend(answer)
    return flattened


# ---------------------------------------------------------------------------
# Target and initial-state selection
# ---------------------------------------------------------------------------


def select_initial_components(z0: np.ndarray, h: HardwareParameters) -> tuple[list[tuple[int, complex]], float]:
    norm = float(np.linalg.norm(z0))
    if norm == 0.0:
        raise ValueError("Zero initial lifted state.")
    amplitudes = z0 / norm
    indices = np.argsort(np.abs(amplitudes))[::-1]
    selected: list[tuple[int, complex]] = []
    for index in indices:
        amplitude = amplitudes[index]
        if abs(amplitude) < h.initial_component_threshold:
            continue
        selected.append((int(index), complex(amplitude)))
        if h.max_initial_components >= 0 and len(selected) >= h.max_initial_components:
            break
    coverage = float(sum(abs(amplitude) ** 2 for _, amplitude in selected))
    return selected, coverage


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------


def main() -> int:
    args = parse_arguments()
    model, raw = load_parameters(args.config)
    hardware = override_hardware(load_hardware(raw), args)
    interfaces = load_interface_parameters(raw)
    output_dir = Path(args.output_dir or hardware.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    time_start = float(args.time_start)
    time_end = (
        float(args.time_end)
        if args.time_end is not None
        else time_start + float(model.final_time)
    )
    duration = time_end - time_start
    if duration <= 0.0:
        raise ValueError("time_end must be larger than time_start")

    print("\n=== 1. Build compact true-3-D conical-frustum model ===")
    system, resolved_organelles = build_model_system(model)

    configured_membrane_state = np.asarray(system.initial_state, dtype=float).copy()
    continuation_path = args.input_state
    side_paths = list(args.side_input_state or [])

    # Compatibility: an old command using --input-state-mode add now means that
    # the specified NPZ is a lateral neighbour seed. It never becomes part of
    # the plasma-membrane input component.
    if args.input_state_mode == "add" and args.input_state:
        print(
            "Warning: --input-state-mode add is deprecated; treating --input-state "
            "as --side-input-state."
        )
        side_paths.insert(0, args.input_state)
        continuation_path = None

    if continuation_path:
        pre_direct_base = load_initial_state(continuation_path, system)
        configured_membrane_component = np.zeros_like(pre_direct_base)
        print(f"Complete continuation state loaded: {Path(continuation_path).resolve()}")
    else:
        pre_direct_base = configured_membrane_state.copy()
        configured_membrane_component = configured_membrane_state.copy()

    # Direct experimental/drug inputs are resolved independently of neighbour
    # handoff. Source-mode events modify A and b; initial-mode events are added
    # only to the requested physical surface.
    system = replace(system, initial_state=pre_direct_base)
    system, direct_components, surface_input_report = apply_surface_inputs(
        system,
        args.surface_inputs,
        time_start=time_start,
        time_end=time_end,
    )
    base_initial_state = np.asarray(system.initial_state, dtype=float).copy()
    membrane_input_component = (
        configured_membrane_component + direct_components["cell_membrane"]
    )
    direct_side_component = direct_components["sidewall"]

    neighbour_side_state = np.zeros_like(base_initial_state)
    loaded_side_paths: list[str] = []
    for side_path in side_paths:
        imported_side = load_initial_state(side_path, system)
        imported_side = restrict_to_lateral_initial(imported_side, system, interfaces)
        neighbour_side_state += float(args.side_input_scale) * imported_side
        loaded_side_paths.append(str(Path(side_path).resolve()))
        print(f"Lateral neighbour initial state added: {Path(side_path).resolve()}")

    # Direct sidewall input and neighbour handoff are reported together as the
    # lateral component, but only the neighbour part is newly added here because
    # the direct component is already included in base_initial_state.
    side_initial_state = direct_side_component + neighbour_side_state
    combined_initial_state = base_initial_state + neighbour_side_state
    system = replace(system, initial_state=combined_initial_state)
    cells = system.grid.z.size
    print(
        f"Compact grid: nz={model.nz}, radial rings={model.nr}, "
        f"azimuth sectors/ring={list(system.grid.ring_counts)} -> {cells} cells"
    )
    print(f"Time window: [{time_start:.6g}, {time_end:.6g}] (duration={duration:.6g})")
    print(f"Species variables: {3*cells}; ellipsoidal organelles: {len(model.organelles)}")
    print("Primary outputs: nuclear-envelope face and open lateral side patches. Plasma membrane is the input end.")

    print("\n=== 2. Solve original nonlinear system ===")
    nonlinear = solve_nonlinear_reference(
        system, time_start=time_start, time_end=time_end
    )

    print("\n=== 3. Build selected second-order Carleman system ===")
    lifted, z0, metadata = build_selected_carleman(system)
    selected_final = solve_selected_carleman(lifted, z0, duration)
    first_order = np.real(
        selected_final[
            metadata["first_order_start"] : metadata["first_order_stop"]
        ]
    )
    carleman_error = relative_error(nonlinear, first_order)
    lifted_dimension = lifted.shape[0]
    data_qubits = math.ceil(math.log2(lifted_dimension))
    total_qubits = data_qubits + 1
    print(f"Selected lifted dimension: {lifted_dimension}")
    print(f"Data qubits: {data_qubits}; Hadamard ancilla: 1; total: {total_qubits}")
    print(f"Carleman vs nonlinear relative error: {carleman_error:.6e}")

    print("\n=== 4. Build nuclear/side interface outputs ===")
    fields = {"nonlinear": nonlinear, "carleman": first_order}
    interface_report = compute_classical_interface_report(
        system,
        fields,
        interfaces,
        time_start=time_start,
        time_end=time_end,
        membrane_initial_state=membrane_input_component,
        side_initial_state=side_initial_state,
    )
    targets = build_interface_target_manifest(
        system,
        metadata,
        interfaces,
        max_targets=args.max_targets,
    )
    print(f"Configured QPU surface targets: {len(targets)}")
    print(
        "Each target remains a shallow computational-basis concentration target; "
        "Legendre/Fourier interface coefficients are fitted after QPU measurement."
    )

    # Full classical accompanying state for the next time window.  The exact
    # protocol keys free/stationary/mobile are retained, but they are compact flat
    # arrays because the centre ring is intentionally collapsed.
    save_final_state(
        output_dir / "final_state.npz",
        system,
        nonlinear,
        first_order,
        time_start=time_start,
        time_end=time_end,
    )
    write_surface_csvs(
        output_dir, system, fields, interfaces,
        membrane_initial_state=membrane_input_component,
        side_initial_state=side_initial_state,
    )
    write_initial_component_npz(
        output_dir / "initial_state_components.npz",
        system,
        membrane_input_component,
        side_initial_state,
        combined_initial_state,
    )
    (output_dir / "patch_coefficients.json").write_text(
        json.dumps(interface_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "resolved_surface_inputs.json").write_text(
        json.dumps(surface_input_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "nuclear_output.json").write_text(
        json.dumps(
            {
                "time_start": time_start,
                "time_end": time_end,
                "role": "biological_result_output",
                "face": "z=0 nuclear envelope",
                "nuclear_face": interface_report.get("nuclear_face", {}),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (output_dir / "interface_target_manifest.json").write_text(
        json.dumps(targets, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_classical_csv(
        output_dir / "classical_3d_results.csv", system, nonlinear, first_order
    )
    (output_dir / "resolved_organelles.json").write_text(
        json.dumps(resolved_organelles, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # LCHS/Pauli preprocessing is still done during dry-run so that resource
    # estimates reflect the exact time window and interface target count.
    lchs = prepare_lchs(lifted, hardware)
    branch_terms: list[list[PauliTerm]] = []
    branch_reports: list[dict[str, Any]] = []
    for branch, k_value in enumerate(lchs.k_nodes):
        hamiltonian = (k_value * lchs.loss + lchs.rotation).tocsr()
        terms, branch_report = select_pauli_terms(
            hamiltonian, data_qubits, hardware
        )
        branch_terms.append(terms)
        branch_reports.append(
            {
                "branch": branch,
                "k": float(k_value),
                "coefficient_real": float(lchs.coefficients[branch].real),
                "coefficient_imag": float(lchs.coefficients[branch].imag),
                "selected_terms": [term.__dict__ for term in terms],
                **branch_report,
            }
        )
        print(
            f"LCHS branch {branch}: Pauli terms={len(terms)}, "
            f"retained L2={branch_report['retained_l2_coverage']:.4f}"
        )

    initial_components, initial_coverage = select_initial_components(z0, hardware)
    expected_circuits = (
        len(lchs.k_nodes) * len(targets) * len(initial_components) * 2
    )
    resource_report = {
        "model": parameter_summary(model),
        "interfaces": interfaces.__dict__,
        "hardware": hardware.__dict__,
        "time_start": time_start,
        "time_end": time_end,
        "duration": duration,
        "continuation_input_state": continuation_path,
        "side_input_states": loaded_side_paths,
        "side_input_scale": args.side_input_scale,
        "surface_inputs_file": args.surface_inputs,
        "resolved_surface_inputs": surface_input_report,
        "initial_state_components_file": "initial_state_components.npz",
        "spatial_cells": cells,
        "physical_variables": 3 * cells,
        "lifted_dimension": lifted_dimension,
        "data_qubits": data_qubits,
        "total_qubits": total_qubits,
        "carleman_relative_error": carleman_error,
        "initial_component_coverage": initial_coverage,
        "initial_components": [
            {
                "lifted_index": index,
                "amplitude_real": amplitude.real,
                "amplitude_imag": amplitude.imag,
            }
            for index, amplitude in initial_components
        ],
        "targets": targets,
        "lchs_shift": lchs.shift,
        "lchs_branches": branch_reports,
        "expected_circuit_count": expected_circuits,
    }
    (output_dir / "resource_report.json").write_text(
        json.dumps(resource_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    f = system.species_slices["free"]
    m = system.species_slices["mobile"]
    classical_summary = {
        "time_start": time_start,
        "time_end": time_end,
        "free_nuclear_mean_nonlinear": nuclear_mean(nonlinear[f], system.grid),
        "free_nuclear_mean_carleman": nuclear_mean(first_order[f], system.grid),
        "nuclear_flux_nonlinear": nuclear_flux(nonlinear[f], nonlinear[m], system),
        "nuclear_flux_carleman": nuclear_flux(first_order[f], first_order[m], system),
        "surface_protocol": {
            "nuclear_file": "nuclear_face.csv",
            "membrane_input_file": "membrane_input.csv",
            "side_initial_input_file": "side_initial_input.csv",
            "initial_components_file": "initial_state_components.npz",
            "sidewall_file": "sidewall_surface.csv",
            "coefficients_file": "patch_coefficients.json",
            "nuclear_output_file": "nuclear_output.json",
            "full_state_file": "final_state.npz",
            "resolved_surface_inputs_file": "resolved_surface_inputs.json",
        },
    }
    (output_dir / "classical_summary.json").write_text(
        json.dumps(classical_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if args.dry_run or args.classical_only:
        neighbor_records = export_all_neighbor_states(
            output_dir, system, interface_report, interfaces
        )
        print("\nDry/classical run complete; no QPU task submitted.")
        print(f"Neighbour-state files: {len(neighbor_records)}")
        print(f"Outputs: {output_dir.resolve()}")
        return 0

    if not targets:
        raise ValueError("No interface targets were generated for QPU execution")

    print("\n=== 5. Build shallow surface-transition circuits ===")
    records: list[CircuitRecord] = []
    for branch_index, terms in enumerate(branch_terms):
        for target_position, target in enumerate(targets):
            for initial_position, (initial_index, _) in enumerate(initial_components):
                for component in ("real", "imag"):
                    program = build_transition_program(
                        data_qubits=data_qubits,
                        initial_index=initial_index,
                        target_index=target["lifted_index"],
                        terms=terms,
                        evolution_time=duration,
                        component=component,
                        h=hardware,
                    )
                    records.append(
                        CircuitRecord(
                            program=program,
                            lchs_index=branch_index,
                            target_position=target_position,
                            initial_position=initial_position,
                            component=component,
                            target_lifted_index=target["lifted_index"],
                            initial_lifted_index=initial_index,
                        )
                    )
    print(f"Circuits: {len(records)}; parallel jobs: {hardware.parallel_jobs}")

    print("\n=== 6. Connect to OriginQ real hardware ===")
    _, backend, backend_name, backend_status = create_cloud_backend(hardware)
    print(f"Selected backend: {backend_name}")
    (output_dir / "backend_status.json").write_text(
        json.dumps(backend_status, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if hardware.run_preflight_bell and not args.skip_preflight:
        bell = run_preflight_bell(backend, hardware)
        print(f"Preflight Bell probabilities: {bell}")

    print("\n=== 7. Submit QPU surface circuits ===")
    expectations = submit_programs_parallel(
        backend, records, hardware, output_dir
    )

    transition: dict[tuple[int, int, int], complex] = {}
    for record, expectation in zip(records, expectations, strict=True):
        key = (record.lchs_index, record.target_position, record.initial_position)
        current = transition.get(key, 0.0j)
        if record.component == "real":
            current = complex(expectation, current.imag)
        else:
            current = complex(current.real, expectation)
        transition[key] = current

    z0_norm = float(np.linalg.norm(z0))
    qpu_targets: list[dict[str, Any]] = []
    for target_position, target in enumerate(targets):
        lchs_sum = 0.0j
        for branch_index, coefficient in enumerate(lchs.coefficients):
            branch_amplitude = 0.0j
            for initial_position, (_, initial_amplitude) in enumerate(initial_components):
                branch_amplitude += initial_amplitude * transition[
                    (branch_index, target_position, initial_position)
                ]
            lchs_sum += coefficient * branch_amplitude
        lifted_value = math.exp(lchs.shift * duration) * z0_norm * lchs_sum
        reference = float(nonlinear[target["physical_index"]])
        carleman = float(first_order[target["physical_index"]])
        qpu_targets.append(
            {
                **target,
                "qpu_real": float(lifted_value.real),
                "qpu_imag": float(lifted_value.imag),
                "nonlinear_reference": reference,
                "selected_carleman": carleman,
                "absolute_error_vs_nonlinear": abs(
                    float(lifted_value.real) - reference
                ),
            }
        )

    # QPU point values are converted into low-order nuclear-face and side-patch
    # coefficients.  These coefficients, rather than the entire classical field,
    # are the preferred inter-cone handoff payload.
    interface_report = add_qpu_coefficients(
        interface_report, system, qpu_targets, interfaces
    )
    (output_dir / "patch_coefficients.json").write_text(
        json.dumps(interface_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "nuclear_output.json").write_text(
        json.dumps(
            {
                "time_start": time_start,
                "time_end": time_end,
                "role": "biological_result_output",
                "face": "z=0 nuclear envelope",
                "nuclear_face": interface_report.get("nuclear_face", {}),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    neighbor_records = export_all_neighbor_states(
        output_dir, system, interface_report, interfaces
    )

    cloud_report = {
        "backend": backend_name,
        "shots": hardware.shots,
        "time_start": time_start,
        "time_end": time_end,
        "targets": qpu_targets,
        "initial_component_coverage": initial_coverage,
        "carleman_relative_error": carleman_error,
        "patch_coefficients_file": "patch_coefficients.json",
        "nuclear_output_file": "nuclear_output.json",
        "qpu_nuclear_output": interface_report.get("nuclear_face", {}).get("sources", {}).get("qpu", {}),
        "neighbor_input_manifest": "neighbor_inputs/neighbor_input_manifest.json",
    }
    (output_dir / "cloud_report.json").write_text(
        json.dumps(cloud_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    for csv_name in ("qpu_nuclear_side_concentrations.csv", "qpu_target_concentrations.csv"):
        with (output_dir / csv_name).open(
            "w", newline="", encoding="utf-8-sig"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(qpu_targets[0].keys()))
            writer.writeheader()
            writer.writerows(qpu_targets)

    print("\n=== 8. Complete ===")
    print(f"QPU nuclear/side points: {len(qpu_targets)}")
    print(f"Neighbour-state files: {len(neighbor_records)}")
    print(f"Outputs: {output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"\nProgram failed: {error}", file=sys.stderr)
        raise
