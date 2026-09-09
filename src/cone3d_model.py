"""
True 3-D conical-frustum intracellular transport model.

Biological states
-----------------
C_f : freely diffusing material
C_s : material bound to stationary binding proteins
C_m : material bound to mobile proteins and transported toward the nucleus

Mechanisms retained
-------------------
* reversible binding/unbinding for stationary and mobile binders
* adjustable mobile-binder fraction
* motor transport from plasma membrane (z=1) toward nucleus (z=0)
* meridional cytoplasmic circulation plus optional azimuthal swirl
* open lateral exchange with surrounding cytoplasm
* permeable 3-D ellipsoidal organelles that slow diffusion and velocity
* nuclear-envelope absorption
* local selected second-order Carleman lift

All quantities are dimensionless.  The compact finite-volume mesh is defined
in (z, rho, phi), where rho=r/R(z).  The central disk is one control volume
per z layer; it is not duplicated around phi.  The physical domain is the genuine 3-D volume
0<=z<=1, 0<=r<=R(z), 0<=phi<2*pi.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict, replace
from pathlib import Path
from typing import Any, Iterable
import csv
import hashlib
import json
import math

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
from scipy.sparse import coo_matrix, csr_matrix, eye
from scipy.sparse.linalg import expm_multiply

Array = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EllipsoidSpec:
    """A fully 3-D permeable organelle.

    axes are full principal-axis lengths [a_full, b_full, c_full].
    center_cyl is [z, r, phi_degrees].
    orientation_deg is [azimuth_about_global_z, tilt_from_global_z, roll].
    """

    organelle_id: str
    axes: tuple[float, float, float]
    center_cyl: tuple[float, float, float]
    orientation_deg: tuple[float, float, float]
    density: float = 0.85
    diffusion_resistance: float = 8.0
    velocity_resistance: float = 5.0
    edge_smoothing: float = 0.06


@dataclass(frozen=True)
class ModelParameters:
    # Compact true-3D mesh.  The central radial cell is collapsed to one
    # volume per z layer instead of an artificial ring of duplicate points.
    # azimuth_counts_by_ring gives the number of sectors in each radial shell.
    # The first value must be 1.  Default [1,4,6,6] with nz=6 gives
    # 6*(1+4+6+6)=102 volume cells and a 511-dimensional selected lift,
    # which still fits 9 data qubits + 1 Hadamard ancilla.
    nz: int = 6
    nr: int = 4
    nphi: int = 6
    azimuth_counts_by_ring: tuple[int, ...] = (1, 4, 6, 6)

    # Conical-frustum geometry.
    radius_nucleus: float = 0.35
    radius_membrane: float = 0.65

    # Diffusion and degradation.
    diffusion_free: float = 0.040
    mobile_diffusion_fraction: float = 0.08
    decay_free: float = 0.10
    decay_stationary: float = 0.04
    decay_mobile: float = 0.05

    # Nuclear capture.
    nuclear_permeability_free: float = 0.18
    nuclear_permeability_mobile: float = 0.10

    # Binding protein pool and reversible kinetics.
    binding_total: float = 0.80
    mobile_binding_fraction: float = 0.35
    kon_stationary: float = 0.90
    koff_stationary: float = 0.24
    kon_mobile: float = 0.70
    koff_mobile: float = 0.18

    # Active transport and cytoplasmic circulation.
    motor_speed: float = 0.55
    circulation_speed: float = 0.22
    circulation_swirl_speed: float = 0.08

    # Open lateral exchange.  The surrounding concentration can be localised
    # in both z and phi.
    side_exchange_rate: float = 0.30
    side_background_base: float = 0.08
    side_background_peak: float = 0.40
    side_input_center_z: float = 0.62
    side_input_sigma_z: float = 0.24
    side_input_axisymmetric: bool = False
    side_input_center_phi_deg: float = 20.0
    side_input_sigma_phi_deg: float = 55.0

    # Initial conditions.  Each state has an independently adjustable input.
    initial_free_amplitude: float = 0.65
    initial_stationary_occupancy: float = 0.05
    initial_mobile_occupancy: float = 0.10
    initial_center_z: float = 0.92
    initial_sigma_z: float = 0.12
    initial_center_rho: float = 0.22
    initial_sigma_rho: float = 0.30
    initial_axisymmetric: bool = False
    initial_center_phi_deg: float = 0.0
    initial_sigma_phi_deg: float = 45.0

    final_time: float = 0.25

    # Fully 3-D organelles.  Multiple organelles are accepted.
    organelles: tuple[EllipsoidSpec, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class Grid3D:
    z_edges: Array
    rho_edges: Array
    z: Array
    rho: Array
    rho_metric: Array
    phi: Array
    phi_width: Array
    r: Array
    x: Array
    y: Array
    volume: Array
    outer_side_area: Array
    lower_z_area: Array
    upper_z_area: Array
    z_index: NDArray[np.int64]
    ring_index: NDArray[np.int64]
    sector_index: NDArray[np.int64]
    ring_counts: tuple[int, ...]
    ring_offsets: tuple[int, ...]
    cells_per_layer: int
    core_mask: NDArray[np.bool_]
    shape: tuple[int, int, int]


@dataclass(frozen=True)
class ModelSystem:
    parameters: ModelParameters
    grid: Grid3D
    linear_matrix: csr_matrix
    source_vector: Array
    initial_state: Array
    organelle_density: Array
    diffusion_factor: Array
    velocity_factor: Array
    side_bulk: Array
    nonlinear_pairs: tuple[tuple[int, int], ...]
    nonlinear_coefficients: csr_matrix
    species_slices: dict[str, slice]


# ---------------------------------------------------------------------------
# Parsing and geometry
# ---------------------------------------------------------------------------


def _stable_angle_from_id(organelle_id: str) -> float:
    digest = hashlib.sha256(organelle_id.encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big")
    return 360.0 * value / float(2**64)


def _as_ellipsoid(item: Any, defaults: dict[str, float] | None = None) -> EllipsoidSpec:
    """Parse either the new 3-D dictionary format or the old five-field list.

    New recommended form:
      {
        "id": "ORG-001",
        "axes": [0.24, 0.11, 0.08],
        "center_cyl": [0.35, 0.12, 40.0],
        "orientation_deg": [40.0, 25.0, 0.0],
        "density": 0.85,
        "diffusion_resistance": 8.0,
        "velocity_resistance": 5.0,
        "edge_smoothing": 0.06
      }

    Backward-compatible old form:
      [long_axis, short_axis, [center_z, center_r], tilt_deg, id]
    It is promoted to a spheroid; its azimuth is generated deterministically
    from the ID.
    """
    defaults = defaults or {}
    if isinstance(item, dict):
        oid = str(item.get("id", item.get("organelle_id", "ORG")))
        axes_raw = item.get("axes")
        if axes_raw is None or len(axes_raw) != 3:
            raise ValueError(f"Organelle {oid}: axes must contain three full lengths.")
        center = item.get("center_cyl")
        if center is None or len(center) != 3:
            raise ValueError(f"Organelle {oid}: center_cyl must be [z,r,phi_deg].")
        orientation = item.get("orientation_deg", [center[2], 0.0, 0.0])
        if len(orientation) != 3:
            raise ValueError(
                f"Organelle {oid}: orientation_deg must be [azimuth,tilt,roll]."
            )
        return EllipsoidSpec(
            organelle_id=oid,
            axes=tuple(float(v) for v in axes_raw),
            center_cyl=tuple(float(v) for v in center),
            orientation_deg=tuple(float(v) for v in orientation),
            density=float(item.get("density", defaults.get("density", 0.85))),
            diffusion_resistance=float(
                item.get(
                    "diffusion_resistance",
                    defaults.get("diffusion_resistance", 8.0),
                )
            ),
            velocity_resistance=float(
                item.get(
                    "velocity_resistance", defaults.get("velocity_resistance", 5.0)
                )
            ),
            edge_smoothing=float(
                item.get("edge_smoothing", defaults.get("edge_smoothing", 0.06))
            ),
        )

    if isinstance(item, (list, tuple)) and len(item) == 5:
        long_axis, short_axis, center_zr, tilt_deg, oid = item
        oid = str(oid)
        if len(center_zr) != 2:
            raise ValueError(f"Legacy organelle {oid}: center must be [z,r].")
        phi_deg = _stable_angle_from_id(oid)
        return EllipsoidSpec(
            organelle_id=oid,
            axes=(float(long_axis), float(short_axis), float(short_axis)),
            center_cyl=(float(center_zr[0]), float(center_zr[1]), phi_deg),
            orientation_deg=(phi_deg, float(tilt_deg), 0.0),
            density=float(defaults.get("density", 0.85)),
            diffusion_resistance=float(defaults.get("diffusion_resistance", 8.0)),
            velocity_resistance=float(defaults.get("velocity_resistance", 5.0)),
            edge_smoothing=float(defaults.get("edge_smoothing", 0.06)),
        )
    raise ValueError("Each organelle must be a dictionary or a legacy five-field list.")


def parameters_from_dict(raw: dict[str, Any]) -> ModelParameters:
    data = dict(raw.get("model", raw))
    organelles_raw = data.pop("organelles", [])
    defaults = {
        "density": float(data.pop("organelle_density_value", 0.85)),
        "diffusion_resistance": float(
            data.pop("organelle_diffusion_resistance", 8.0)
        ),
        "velocity_resistance": float(data.pop("organelle_velocity_resistance", 5.0)),
        "edge_smoothing": float(data.pop("organelle_edge_smoothing", 0.06)),
    }
    if "azimuth_counts_by_ring" in data:
        data["azimuth_counts_by_ring"] = tuple(int(v) for v in data["azimuth_counts_by_ring"])
    valid = set(ModelParameters.__dataclass_fields__) - {"organelles"}
    unknown = sorted(set(data) - valid)
    if unknown:
        raise ValueError(f"Unknown model parameters: {unknown}")
    organelles = tuple(_as_ellipsoid(item, defaults) for item in organelles_raw)
    return ModelParameters(**data, organelles=organelles)


def load_parameters(path: str | Path) -> tuple[ModelParameters, dict[str, Any]]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return parameters_from_dict(raw), raw


def radius_at(z: Array | float, p: ModelParameters) -> Array | float:
    return p.radius_nucleus + (p.radius_membrane - p.radius_nucleus) * np.asarray(z)


def periodic_angle_difference(angle: Array, center: float) -> Array:
    return np.angle(np.exp(1j * (angle - center)))


def _resolved_ring_counts(p: ModelParameters) -> tuple[int, ...]:
    if p.azimuth_counts_by_ring:
        counts = tuple(int(v) for v in p.azimuth_counts_by_ring)
    else:
        inner = max(3, p.nphi // 2)
        counts = (1,) + tuple(inner if ir == 1 else p.nphi for ir in range(1, p.nr))
    if len(counts) != p.nr:
        raise ValueError("azimuth_counts_by_ring must have exactly nr entries.")
    if counts[0] != 1:
        raise ValueError("The first radial ring must contain exactly one collapsed core cell.")
    if any(v < 1 for v in counts):
        raise ValueError("All azimuth sector counts must be positive.")
    return counts


def build_grid(p: ModelParameters) -> Grid3D:
    if p.nz < 2 or p.nr < 2:
        raise ValueError("nz and nr must both be at least 2.")
    counts = _resolved_ring_counts(p)
    z_edges = np.linspace(0.0, 1.0, p.nz + 1)
    rho_edges = np.linspace(0.0, 1.0, p.nr + 1)
    offsets = [0]
    for count in counts[:-1]:
        offsets.append(offsets[-1] + count)
    cells_per_layer = sum(counts)

    slope = p.radius_membrane - p.radius_nucleus
    z_values=[]; rho_values=[]; rho_metric_values=[]; phi_values=[]; phi_widths=[]
    r_values=[]; x_values=[]; y_values=[]; volumes=[]; side_areas=[]
    lower_areas=[]; upper_areas=[]; z_indices=[]; ring_indices=[]; sector_indices=[]

    def int_r2(z0: float, z1: float) -> float:
        rn = p.radius_nucleus
        s = slope
        return (
            rn * rn * (z1-z0)
            + rn*s*(z1*z1-z0*z0)
            + (s*s/3.0)*(z1**3-z0**3)
        )

    for iz in range(p.nz):
        z0,z1=z_edges[iz],z_edges[iz+1]
        zmid=0.5*(z0+z1); dz=z1-z0
        rmid=float(radius_at(zmid,p))
        for ir,count in enumerate(counts):
            rho0,rho1=rho_edges[ir],rho_edges[ir+1]
            # Area-weighted radial centroid for transport distances.  The
            # collapsed core is displayed at r=0 but retains its full disk volume.
            rho_metric=(2.0/3.0)*(rho1**3-rho0**3)/max(rho1**2-rho0**2,1e-15)
            for ip in range(count):
                dphi=2.0*math.pi/count
                phi=0.0 if count==1 else (ip+0.5)*dphi
                display_r=0.0 if ir==0 else rho_metric*rmid
                vol=0.5*dphi*(rho1**2-rho0**2)*int_r2(z0,z1)
                area0=0.5*dphi*(rho1**2-rho0**2)*float(radius_at(z0,p))**2
                area1=0.5*dphi*(rho1**2-rho0**2)*float(radius_at(z1,p))**2
                side=(rmid*math.sqrt(1.0+slope*slope)*dz*dphi) if ir==p.nr-1 else 0.0
                z_values.append(zmid); rho_values.append(0.0 if ir==0 else rho_metric)
                rho_metric_values.append(rho_metric); phi_values.append(phi); phi_widths.append(dphi)
                r_values.append(display_r); x_values.append(display_r*math.cos(phi)); y_values.append(display_r*math.sin(phi))
                volumes.append(vol); side_areas.append(side)
                lower_areas.append(area0 if iz==0 else 0.0)
                upper_areas.append(area1 if iz==p.nz-1 else 0.0)
                z_indices.append(iz); ring_indices.append(ir); sector_indices.append(ip)

    arr=lambda values,dtype=float: np.asarray(values,dtype=dtype)
    ring_counts=tuple(counts); ring_offsets=tuple(offsets)
    return Grid3D(
        z_edges=z_edges, rho_edges=rho_edges,
        z=arr(z_values), rho=arr(rho_values), rho_metric=arr(rho_metric_values),
        phi=arr(phi_values), phi_width=arr(phi_widths),
        r=arr(r_values), x=arr(x_values), y=arr(y_values),
        volume=arr(volumes), outer_side_area=arr(side_areas),
        lower_z_area=arr(lower_areas), upper_z_area=arr(upper_areas),
        z_index=arr(z_indices,np.int64), ring_index=arr(ring_indices,np.int64),
        sector_index=arr(sector_indices,np.int64), ring_counts=ring_counts,
        ring_offsets=ring_offsets, cells_per_layer=cells_per_layer,
        core_mask=arr([ir==0 for ir in ring_indices],np.bool_),
        shape=(p.nz,p.nr,max(counts)),
    )


def grid_index(grid: Grid3D, iz: int, ir: int, ip: int) -> int:
    count=grid.ring_counts[ir]
    if not (0 <= iz < grid.shape[0] and 0 <= ir < len(grid.ring_counts) and 0 <= ip < count):
        raise IndexError((iz,ir,ip))
    return iz*grid.cells_per_layer + grid.ring_offsets[ir] + ip


def angular_overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1,b1)-max(a0,b0))


def _rotation_matrix(azimuth_deg: float, tilt_deg: float, roll_deg: float) -> Array:
    """Local-to-global Z-Y-X Euler rotation."""
    az = math.radians(azimuth_deg)
    ti = math.radians(tilt_deg)
    ro = math.radians(roll_deg)
    cz, sz = math.cos(az), math.sin(az)
    cy, sy = math.cos(ti), math.sin(ti)
    cx, sx = math.cos(ro), math.sin(ro)
    rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    return rz @ ry @ rx


def organelle_fields(
    grid: Grid3D, p: ModelParameters
) -> tuple[Array, Array, Array, list[dict[str, Any]]]:
    n = grid.z.size
    survival = np.ones(n, dtype=float)
    diffusion_load = np.zeros(n, dtype=float)
    velocity_load = np.zeros(n, dtype=float)
    resolved: list[dict[str, Any]] = []
    points = np.column_stack([grid.x, grid.y, grid.z])

    for spec in p.organelles:
        axes = 0.5 * np.asarray(spec.axes, dtype=float)
        if np.any(axes <= 0.0):
            raise ValueError(f"Organelle {spec.organelle_id}: axes must be positive.")
        zc, rc, phideg = spec.center_cyl
        phi = math.radians(phideg)
        center = np.array([rc * math.cos(phi), rc * math.sin(phi), zc], dtype=float)
        rotation = _rotation_matrix(*spec.orientation_deg)
        local = (points - center) @ rotation
        q = np.sum((local / axes) ** 2, axis=1)
        smoothing = max(spec.edge_smoothing, 1e-6)
        rootq = np.sqrt(np.maximum(q, 0.0))
        occupancy = 1.0 / (1.0 + np.exp(np.clip((rootq - 1.0) / smoothing, -60, 60)))
        local_density = np.clip(spec.density * occupancy, 0.0, 1.0)
        survival *= 1.0 - local_density
        diffusion_load += spec.diffusion_resistance * local_density
        velocity_load += spec.velocity_resistance * local_density
        resolved.append(
            {
                "id": spec.organelle_id,
                "axes_full": list(spec.axes),
                "center_cyl": list(spec.center_cyl),
                "center_xyz": center.tolist(),
                "orientation_deg": list(spec.orientation_deg),
                "density": spec.density,
                "diffusion_resistance": spec.diffusion_resistance,
                "velocity_resistance": spec.velocity_resistance,
                "edge_smoothing": spec.edge_smoothing,
                "maximum_sampled_occupancy": float(np.max(occupancy)),
            }
        )
    density = np.clip(1.0 - survival, 0.0, 1.0)
    return density, 1.0 / (1.0 + diffusion_load), 1.0 / (1.0 + velocity_load), resolved


# ---------------------------------------------------------------------------
# Flow and finite-volume operators
# ---------------------------------------------------------------------------


def circulation_velocity(
    z: float, r: float, phi: float, p: ModelParameters
) -> tuple[float, float, float]:
    """Return cylindrical (v_z, v_r, v_phi).

    The meridional component is generated from a streamfunction and vanishes
    on the conical sidewall.  The swirl component is optional and also vanishes
    at the axis, sidewall and end planes.
    """
    radius = float(radius_at(z, p))
    if radius <= 0.0:
        return 0.0, 0.0, 0.0
    rho = float(np.clip(r / radius, 0.0, 1.0))
    u = p.circulation_speed
    slope = p.radius_membrane - p.radius_nucleus
    sinz = math.sin(math.pi * z)
    cosz = math.cos(math.pi * z)
    one = 1.0 - rho * rho
    vz = u * sinz * one * (1.0 - 3.0 * rho * rho)
    vr = -0.5 * u * r * (
        math.pi * one * one * cosz
        + 4.0 * rho * rho * one * slope / max(radius, 1e-12) * sinz
    )
    vphi = (
        p.circulation_swirl_speed
        * 4.0
        * rho
        * (1.0 - rho)
        * sinz
    )
    return vz, vr, vphi


def side_bulk_concentration(z: Array, phi: Array, p: ModelParameters) -> Array:
    zpart = np.exp(-((z - p.side_input_center_z) ** 2) / (2.0 * p.side_input_sigma_z**2))
    if p.side_input_axisymmetric:
        phipart = np.ones_like(zpart)
    else:
        center = math.radians(p.side_input_center_phi_deg)
        sigma = math.radians(max(p.side_input_sigma_phi_deg, 1e-6))
        dphi = periodic_angle_difference(phi, center)
        phipart = np.exp(-(dphi**2) / (2.0 * sigma**2))
    return p.side_background_base + p.side_background_peak * zpart * phipart


def _harmonic(a: float, b: float) -> float:
    if a <= 0.0 or b <= 0.0:
        return 0.0
    return 2.0 * a * b / (a + b)


def _add_pair_diffusion(
    rows: list[int], cols: list[int], vals: list[float],
    a: int, b: int, conductance: float, volume: Array,
) -> None:
    if conductance == 0.0:
        return
    rows.extend([a,a,b,b]); cols.extend([a,b,b,a])
    vals.extend([-conductance/volume[a],conductance/volume[a],-conductance/volume[b],conductance/volume[b]])


def _add_pair_advection(
    rows: list[int], cols: list[int], vals: list[float],
    a: int, b: int, flux_a_to_b: float, volume: Array,
) -> None:
    if flux_a_to_b >= 0.0:
        rows.extend([a,b]); cols.extend([a,a]); vals.extend([-flux_a_to_b/volume[a],flux_a_to_b/volume[b]])
    else:
        magnitude=-flux_a_to_b
        rows.extend([a,b]); cols.extend([b,b]); vals.extend([magnitude/volume[a],-magnitude/volume[b]])


def build_scalar_transport_operator(
    grid: Grid3D,
    p: ModelParameters,
    diffusion_cell: Array,
    velocity_factor: Array,
    *,
    include_circulation: bool,
    include_motor: bool,
    nuclear_permeability: float,
    lateral_exchange: bool,
) -> tuple[csr_matrix, Array]:
    """Conservative finite-volume transport on the compact 3-D mesh.

    The axis core has one control volume per z layer.  Radial faces are coupled
    by exact angular-overlap fractions, so neighbouring rings may use different
    azimuthal sector counts without creating duplicate centre points.
    """
    nz=p.nz; nr=p.nr; n=grid.z.size
    rows=[]; cols=[]; vals=[]; source=np.zeros(n,float)
    dz=1.0/nz; slope=p.radius_membrane-p.radius_nucleus

    def vel_at(idx:int,z:float|None=None,r:float|None=None,phi:float|None=None):
        zz=grid.z[idx] if z is None else z
        rr=(grid.rho_metric[idx]*float(radius_at(zz,p))) if r is None else r
        pp=grid.phi[idx] if phi is None else phi
        vz=vr=vp=0.0
        if include_circulation:
            vz,vr,vp=circulation_velocity(float(zz),float(rr),float(pp),p)
        if include_motor:
            vz-=p.motor_speed
        factor=velocity_factor[idx]
        return vz*factor,vr*factor,vp*factor

    # z faces: topology is identical in every layer.
    for iz in range(nz-1):
        zface=grid.z_edges[iz+1]; radius_face=float(radius_at(zface,p))
        for ir,count in enumerate(grid.ring_counts):
            rho0,rho1=grid.rho_edges[ir],grid.rho_edges[ir+1]
            for ip in range(count):
                a=grid_index(grid,iz,ir,ip); b=grid_index(grid,iz+1,ir,ip)
                area=0.5*grid.phi_width[a]*(rho1**2-rho0**2)*radius_face**2
                dface=_harmonic(diffusion_cell[a],diffusion_cell[b])
                _add_pair_diffusion(rows,cols,vals,a,b,dface*area/dz,grid.volume)
                va=vel_at(a); vb=vel_at(b)
                _add_pair_advection(rows,cols,vals,a,b,0.5*(va[0]+vb[0])*area,grid.volume)

    # Radial faces with angular overlap between possibly unequal sector counts.
    for iz in range(nz):
        zmid=0.5*(grid.z_edges[iz]+grid.z_edges[iz+1]); radius_mid=float(radius_at(zmid,p))
        for ir in range(nr-1):
            inner_count=grid.ring_counts[ir]; outer_count=grid.ring_counts[ir+1]
            rho_face=grid.rho_edges[ir+1]; rface=rho_face*radius_mid
            normal_norm=math.sqrt(1.0+(rho_face*slope)**2)
            rho_in=(0.0 if ir==0 else (2.0/3.0)*(grid.rho_edges[ir+1]**3-grid.rho_edges[ir]**3)/max(grid.rho_edges[ir+1]**2-grid.rho_edges[ir]**2,1e-15))
            j=ir+1
            rho_out=(2.0/3.0)*(grid.rho_edges[j+1]**3-grid.rho_edges[j]**3)/max(grid.rho_edges[j+1]**2-grid.rho_edges[j]**2,1e-15)
            distance=max(radius_mid*(rho_out-rho_in),1e-10)
            for ia in range(inner_count):
                a0=ia*2.0*math.pi/inner_count; a1=(ia+1)*2.0*math.pi/inner_count
                for ib in range(outer_count):
                    b0=ib*2.0*math.pi/outer_count; b1=(ib+1)*2.0*math.pi/outer_count
                    overlap=angular_overlap(a0,a1,b0,b1)
                    if overlap<=1e-14: continue
                    a=grid_index(grid,iz,ir,ia); b=grid_index(grid,iz,ir+1,ib)
                    area=rface*normal_norm*dz*overlap
                    dface=_harmonic(diffusion_cell[a],diffusion_cell[b])
                    _add_pair_diffusion(rows,cols,vals,a,b,dface*area/distance,grid.volume)
                    va=vel_at(a); vb=vel_at(b)
                    vz=0.5*(va[0]+vb[0]); vr=0.5*(va[1]+vb[1])
                    vn=(vr-rho_face*slope*vz)/normal_norm
                    _add_pair_advection(rows,cols,vals,a,b,vn*area,grid.volume)

    # Periodic phi faces.  No artificial azimuthal subdivision exists in core.
    for iz in range(nz):
        zmid=0.5*(grid.z_edges[iz]+grid.z_edges[iz+1]); radius_mid=float(radius_at(zmid,p))
        for ir,count in enumerate(grid.ring_counts):
            if count<=1: continue
            rho0,rho1=grid.rho_edges[ir],grid.rho_edges[ir+1]
            rho_mid=(2.0/3.0)*(rho1**3-rho0**3)/max(rho1**2-rho0**2,1e-15)
            rmid=max(rho_mid*radius_mid,1e-10); dphi=2.0*math.pi/count
            face_area=radius_mid*dz*(rho1-rho0); distance=rmid*dphi
            for ip in range(count):
                jp=(ip+1)%count
                a=grid_index(grid,iz,ir,ip); b=grid_index(grid,iz,ir,jp)
                dface=_harmonic(diffusion_cell[a],diffusion_cell[b])
                _add_pair_diffusion(rows,cols,vals,a,b,dface*face_area/distance,grid.volume)
                va=vel_at(a); vb=vel_at(b)
                _add_pair_advection(rows,cols,vals,a,b,0.5*(va[2]+vb[2])*face_area,grid.volume)

    # Nuclear and cell-membrane ends.
    for idx in np.flatnonzero(grid.z_index==0):
        area=grid.lower_z_area[idx]
        if nuclear_permeability>0.0:
            rows.append(int(idx)); cols.append(int(idx)); vals.append(-nuclear_permeability*area/grid.volume[idx])
        vz,_,_=vel_at(int(idx),z=0.0)
        if -vz>0.0:
            rows.append(int(idx)); cols.append(int(idx)); vals.append(vz*area/grid.volume[idx])
    for idx in np.flatnonzero(grid.z_index==nz-1):
        area=grid.upper_z_area[idx]; vz,_,_=vel_at(int(idx),z=1.0)
        if vz>0.0:
            rows.append(int(idx)); cols.append(int(idx)); vals.append(-vz*area/grid.volume[idx])

    if lateral_exchange:
        bulk=side_bulk_concentration(grid.z,grid.phi,p)
        for idx in np.flatnonzero(grid.ring_index==nr-1):
            area=grid.outer_side_area[idx]; coefficient=p.side_exchange_rate*area/grid.volume[idx]
            rows.append(int(idx)); cols.append(int(idx)); vals.append(-coefficient)
            source[idx]+=coefficient*bulk[idx]

    matrix=coo_matrix((vals,(rows,cols)),shape=(n,n)).tocsr(); matrix.sum_duplicates()
    return matrix,source


# ---------------------------------------------------------------------------
# Biological system and selected Carleman lift
# ---------------------------------------------------------------------------


def build_initial_conditions(grid: Grid3D, p: ModelParameters) -> tuple[Array, Array, Array]:
    zpart = np.exp(-((grid.z - p.initial_center_z) ** 2) / (2.0 * p.initial_sigma_z**2))
    rpart = np.exp(-((grid.rho - p.initial_center_rho) ** 2) / (2.0 * p.initial_sigma_rho**2))
    if p.initial_axisymmetric:
        phipart = np.ones_like(zpart)
    else:
        center = math.radians(p.initial_center_phi_deg)
        sigma = math.radians(max(p.initial_sigma_phi_deg, 1e-6))
        dphi = periodic_angle_difference(grid.phi, center)
        phipart = np.exp(-(dphi**2) / (2.0 * sigma**2))
    pulse = zpart * rpart * phipart
    free = p.initial_free_amplitude * pulse
    stationary = p.initial_stationary_occupancy * pulse
    mobile = p.initial_mobile_occupancy * pulse
    return free, stationary, mobile


def build_model_system(p: ModelParameters) -> tuple[ModelSystem, list[dict[str, Any]]]:
    grid = build_grid(p)
    n = grid.z.size
    density, diffusion_factor, velocity_factor, resolved = organelle_fields(grid, p)

    free_diffusion = p.diffusion_free * diffusion_factor
    mobile_diffusion = (
        p.diffusion_free * p.mobile_diffusion_fraction * diffusion_factor
    )
    free_transport, free_source = build_scalar_transport_operator(
        grid, p, free_diffusion, velocity_factor,
        include_circulation=True, include_motor=False,
        nuclear_permeability=p.nuclear_permeability_free,
        lateral_exchange=True,
    )
    mobile_transport, _ = build_scalar_transport_operator(
        grid, p, mobile_diffusion, velocity_factor,
        include_circulation=True, include_motor=True,
        nuclear_permeability=p.nuclear_permeability_mobile,
        lateral_exchange=False,
    )

    f = slice(0, n)
    s = slice(n, 2 * n)
    m = slice(2 * n, 3 * n)
    total = 3 * n
    bs = (1.0 - p.mobile_binding_fraction) * p.binding_total
    bm = p.mobile_binding_fraction * p.binding_total

    # Assemble the linear species block matrix.
    blocks_rows: list[int] = []
    blocks_cols: list[int] = []
    blocks_vals: list[float] = []

    def append_sparse(block: csr_matrix, row_offset: int, col_offset: int) -> None:
        coo = block.tocoo()
        blocks_rows.extend((coo.row + row_offset).tolist())
        blocks_cols.extend((coo.col + col_offset).tolist())
        blocks_vals.extend(coo.data.tolist())

    append_sparse(free_transport, 0, 0)
    append_sparse(mobile_transport, 2 * n, 2 * n)
    for i in range(n):
        # Free state.
        blocks_rows += [i, i, i]
        blocks_cols += [i, n + i, 2 * n + i]
        blocks_vals += [
            -(p.decay_free + p.kon_stationary * bs + p.kon_mobile * bm),
            p.koff_stationary,
            p.koff_mobile,
        ]
        # Stationary bound state.
        blocks_rows += [n + i, n + i]
        blocks_cols += [i, n + i]
        blocks_vals += [p.kon_stationary * bs, -(p.koff_stationary + p.decay_stationary)]
        # Mobile bound state.
        blocks_rows += [2 * n + i, 2 * n + i]
        blocks_cols += [i, 2 * n + i]
        blocks_vals += [p.kon_mobile * bm, -(p.koff_mobile + p.decay_mobile)]

    linear = coo_matrix(
        (blocks_vals, (blocks_rows, blocks_cols)), shape=(total, total)
    ).tocsr()
    source = np.zeros(total, dtype=float)
    source[f] = free_source

    free0, stationary0, mobile0 = build_initial_conditions(grid, p)
    initial = np.concatenate([free0, stationary0, mobile0])

    # Nonlinear local monomials: f_i*s_i and f_i*m_i.
    pairs: list[tuple[int, int]] = []
    qrows: list[int] = []
    qcols: list[int] = []
    qvals: list[float] = []
    for i in range(n):
        col_fs = len(pairs)
        pairs.append((i, n + i))
        qrows.extend([i, n + i])
        qcols.extend([col_fs, col_fs])
        qvals.extend([p.kon_stationary, -p.kon_stationary])
        col_fm = len(pairs)
        pairs.append((i, 2 * n + i))
        qrows.extend([i, 2 * n + i])
        qcols.extend([col_fm, col_fm])
        qvals.extend([p.kon_mobile, -p.kon_mobile])
    qmatrix = coo_matrix(
        (qvals, (qrows, qcols)), shape=(total, len(pairs))
    ).tocsr()

    system = ModelSystem(
        parameters=p,
        grid=grid,
        linear_matrix=linear,
        source_vector=source,
        initial_state=initial,
        organelle_density=density,
        diffusion_factor=diffusion_factor,
        velocity_factor=velocity_factor,
        side_bulk=side_bulk_concentration(grid.z, grid.phi, p),
        nonlinear_pairs=tuple(pairs),
        nonlinear_coefficients=qmatrix,
        species_slices={"free": f, "stationary": s, "mobile": m},
    )
    return system, resolved


def nonlinear_rhs(_: float, y: Array, system: ModelSystem) -> Array:
    pair_values = np.fromiter(
        (y[a] * y[b] for a, b in system.nonlinear_pairs),
        dtype=float,
        count=len(system.nonlinear_pairs),
    )
    return (
        system.linear_matrix @ y
        + system.source_vector
        + system.nonlinear_coefficients @ pair_values
    )


def solve_nonlinear_reference(
    system: ModelSystem,
    time_start: float = 0.0,
    time_end: float | None = None,
) -> Array:
    """Solve one absolute-time window.

    The present coefficients are time independent, so the numerical result only
    depends on ``time_end-time_start``.  Keeping absolute times in the interface
    makes repeated neighbouring-cone windows traceable.
    """
    p = system.parameters
    end = time_start + p.final_time if time_end is None else float(time_end)
    if end <= time_start:
        raise ValueError(f"time_end ({end}) must exceed time_start ({time_start}).")
    solution = solve_ivp(
        nonlinear_rhs,
        (float(time_start), end),
        system.initial_state,
        args=(system,),
        method="DOP853",
        rtol=2e-9,
        atol=2e-11,
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    return solution.y[:, -1]


def build_selected_carleman(system: ModelSystem) -> tuple[csr_matrix, Array, dict[str, Any]]:
    """Build the local selected second-order Carleman system.

    Basis:
      [1, all first-order species, f_i*s_i, f_i*m_i]

    The derivative of each selected product retains affine and linear terms
    whose monomials are also present in the selected basis.  Cubic terms and
    unselected nonlocal products are the second-order sparse closure.
    """
    a = system.linear_matrix.tocsr()
    b = system.source_vector
    q = system.nonlinear_coefficients.tocsr()
    ydim = a.shape[0]
    pairs = list(system.nonlinear_pairs)
    pair_to_column = {tuple(sorted(pair)): idx for idx, pair in enumerate(pairs)}
    lifted_dim = 1 + ydim + len(pairs)
    rows: list[int] = []
    cols: list[int] = []
    vals: list[complex] = []

    # First-order rows.
    acoo = a.tocoo()
    for row, col, value in zip(acoo.row, acoo.col, acoo.data, strict=True):
        rows.append(1 + int(row)); cols.append(1 + int(col)); vals.append(value)
    for i, value in enumerate(b):
        if value != 0.0:
            rows.append(1 + i); cols.append(0); vals.append(value)
    qcoo = q.tocoo()
    for row, col, value in zip(qcoo.row, qcoo.col, qcoo.data, strict=True):
        rows.append(1 + int(row)); cols.append(1 + ydim + int(col)); vals.append(value)

    # Product rows from d(y_a*y_b)/dt, truncated/projected at degree 2.
    for product_idx, (var_a, var_b) in enumerate(pairs):
        target_row = 1 + ydim + product_idx
        if b[var_a] != 0.0:
            rows.append(target_row); cols.append(1 + var_b); vals.append(b[var_a])
        if b[var_b] != 0.0:
            rows.append(target_row); cols.append(1 + var_a); vals.append(b[var_b])

        start, stop = a.indptr[var_a], a.indptr[var_a + 1]
        for j, value in zip(a.indices[start:stop], a.data[start:stop], strict=True):
            key = tuple(sorted((int(j), var_b)))
            selected = pair_to_column.get(key)
            if selected is not None:
                rows.append(target_row)
                cols.append(1 + ydim + selected)
                vals.append(value)
        start, stop = a.indptr[var_b], a.indptr[var_b + 1]
        for j, value in zip(a.indices[start:stop], a.data[start:stop], strict=True):
            key = tuple(sorted((var_a, int(j))))
            selected = pair_to_column.get(key)
            if selected is not None:
                rows.append(target_row)
                cols.append(1 + ydim + selected)
                vals.append(value)

    lifted = coo_matrix(
        (np.asarray(vals, dtype=np.complex128), (rows, cols)),
        shape=(lifted_dim, lifted_dim),
    ).tocsr()
    lifted.sum_duplicates()

    y0 = system.initial_state
    z0 = np.empty(lifted_dim, dtype=np.complex128)
    z0[0] = 1.0
    z0[1 : 1 + ydim] = y0
    for idx, (va, vb) in enumerate(pairs):
        z0[1 + ydim + idx] = y0[va] * y0[vb]
    metadata = {
        "constant_index": 0,
        "first_order_start": 1,
        "first_order_stop": 1 + ydim,
        "product_start": 1 + ydim,
        "product_stop": lifted_dim,
        "lifted_dimension": lifted_dim,
        "physical_dimension": ydim,
        "spatial_cells": system.grid.z.size,
    }
    return lifted, z0, metadata


def solve_selected_carleman(matrix: csr_matrix, z0: ComplexArray, time: float) -> ComplexArray:
    return expm_multiply(matrix * time, z0)


def relative_error(reference: NDArray, estimate: NDArray) -> float:
    denominator = float(np.linalg.norm(reference))
    if denominator == 0.0:
        return float(np.linalg.norm(estimate))
    return float(np.linalg.norm(reference - estimate) / denominator)


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def nuclear_layer_indices(grid: Grid3D) -> Array:
    return np.flatnonzero(grid.z_index == 0).astype(int)


def nuclear_mean(field: Array, grid: Grid3D) -> float:
    indices = nuclear_layer_indices(grid)
    weights = grid.lower_z_area[indices]
    return float(np.sum(weights * field[indices]) / np.sum(weights))


def nuclear_flux(field_free: Array, field_mobile: Array, system: ModelSystem) -> float:
    p, grid = system.parameters, system.grid
    indices = nuclear_layer_indices(grid)
    weights = grid.lower_z_area[indices]
    return float(
        np.sum(
            weights
            * (
                p.nuclear_permeability_free * field_free[indices]
                + p.nuclear_permeability_mobile * field_mobile[indices]
            )
        )
    )


def write_classical_csv(
    path: Path,
    system: ModelSystem,
    nonlinear: Array,
    carleman_first_order: Array,
) -> None:
    n = system.grid.z.size
    f, s, m = system.species_slices.values()
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "cell", "z_index", "radial_ring", "sector_index", "sector_count",
                "z", "rho", "rho_metric", "phi_deg", "phi_width_deg", "r", "x", "y", "volume",
                "organelle_density", "diffusion_factor", "velocity_factor",
                "side_bulk",
                "free_initial", "stationary_initial", "mobile_initial",
                "free_nonlinear", "stationary_nonlinear", "mobile_nonlinear",
                "free_carleman", "stationary_carleman", "mobile_carleman",
            ]
        )
        for i in range(n):
            writer.writerow(
                [
                    i, int(system.grid.z_index[i]), int(system.grid.ring_index[i]),
                    int(system.grid.sector_index[i]), int(system.grid.ring_counts[int(system.grid.ring_index[i])]),
                    system.grid.z[i], system.grid.rho[i], system.grid.rho_metric[i],
                    math.degrees(system.grid.phi[i]), math.degrees(system.grid.phi_width[i]), system.grid.r[i],
                    system.grid.x[i], system.grid.y[i], system.grid.volume[i],
                    system.organelle_density[i], system.diffusion_factor[i],
                    system.velocity_factor[i], system.side_bulk[i],
                    system.initial_state[f][i], system.initial_state[s][i], system.initial_state[m][i],
                    nonlinear[f][i], nonlinear[s][i], nonlinear[m][i],
                    carleman_first_order[f][i], carleman_first_order[s][i], carleman_first_order[m][i],
                ]
            )


def parameter_summary(p: ModelParameters) -> dict[str, Any]:
    result = asdict(p)
    result["organelles"] = [asdict(item) for item in p.organelles]
    counts = _resolved_ring_counts(p)
    result["resolved_azimuth_counts_by_ring"] = list(counts)
    result["cells_per_z_layer"] = int(sum(counts))
    result["total_spatial_cells"] = int(p.nz * sum(counts))
    return result


# ---------------------------------------------------------------------------
# External state and neighbouring-cone interface protocol
# ---------------------------------------------------------------------------


def grid_signature(grid: Grid3D) -> str:
    """Stable signature used to reject states created on an incompatible mesh."""
    payload = {
        "shape": list(grid.shape),
        "ring_counts": list(grid.ring_counts),
        "z_edges": np.round(grid.z_edges, 14).tolist(),
        "rho_edges": np.round(grid.rho_edges, 14).tolist(),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def compact_field_to_dense(
    field: Array,
    grid: Grid3D,
    fill_value: float = np.nan,
) -> Array:
    """Map a compact variable-sector field to (nz,nr,max_nphi).

    Invalid sector slots are filled with NaN.  The collapsed central disk has
    only sector 0, so it is never duplicated around the axis.
    """
    values = np.asarray(field, dtype=float).reshape(-1)
    if values.size != grid.z.size:
        raise ValueError(f"compact field length {values.size} != {grid.z.size}")
    dense = np.full(grid.shape, fill_value, dtype=float)
    for index, value in enumerate(values):
        dense[
            int(grid.z_index[index]),
            int(grid.ring_index[index]),
            int(grid.sector_index[index]),
        ] = value
    return dense


def dense_field_to_compact(field: Array, grid: Grid3D, name: str = "field") -> Array:
    """Read either a compact vector or a padded dense field."""
    values = np.asarray(field, dtype=float)
    if values.ndim == 1:
        if values.size != grid.z.size:
            raise ValueError(f"{name}: flat length {values.size} != {grid.z.size}")
        return values.copy()
    if tuple(values.shape) != tuple(grid.shape):
        raise ValueError(f"{name}: {values.shape} != compact dense shape {grid.shape}")
    compact = np.empty(grid.z.size, dtype=float)
    for index in range(grid.z.size):
        compact[index] = values[
            int(grid.z_index[index]),
            int(grid.ring_index[index]),
            int(grid.sector_index[index]),
        ]
    if not np.all(np.isfinite(compact)):
        raise ValueError(f"{name}: valid compact cells contain NaN or infinite values")
    return compact


def replace_initial_state(
    system: ModelSystem,
    free: Array,
    stationary: Array,
    mobile: Array,
) -> ModelSystem:
    n = system.grid.z.size
    f = dense_field_to_compact(free, system.grid, "free")
    s = dense_field_to_compact(stationary, system.grid, "stationary")
    m = dense_field_to_compact(mobile, system.grid, "mobile")
    if not (f.size == s.size == m.size == n):
        raise ValueError("External species state sizes do not match the current grid.")
    return replace(system, initial_state=np.concatenate([f, s, m]))


def load_external_state(path: str | Path, system: ModelSystem) -> ModelSystem:
    """Load ``final_state.npz`` from this or a neighbouring cone/window."""
    state_path = Path(path)
    with np.load(state_path, allow_pickle=False) as data:
        stored_signature = str(data["grid_signature"].item()) if "grid_signature" in data else None
        current_signature = grid_signature(system.grid)
        if stored_signature and stored_signature != current_signature:
            raise ValueError(
                "Input-state grid signature does not match this cone. "
                "Use a Schwarz/interface mapper before passing states between unlike grids."
            )
        def get_species(name: str) -> Array:
            flat_key = f"{name}_flat"
            if flat_key in data:
                return np.asarray(data[flat_key], dtype=float)
            if name in data:
                return np.asarray(data[name], dtype=float)
            raise KeyError(f"Input state lacks {name!r} and {flat_key!r}.")
        return replace_initial_state(
            system,
            get_species("free"),
            get_species("stationary"),
            get_species("mobile"),
        )


def save_final_state(
    path: str | Path,
    system: ModelSystem,
    final_state: Array,
    *,
    time_start: float,
    time_end: float,
    source: str = "classical_nonlinear",
) -> None:
    """Write the full three-species field using the small file protocol."""
    n = system.grid.z.size
    values = np.asarray(final_state, dtype=float).reshape(-1)
    if values.size != 3 * n:
        raise ValueError(f"final state length {values.size} != {3*n}")
    f, s, m = system.species_slices.values()
    free = values[f]
    stationary = values[s]
    mobile = values[m]
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        # Names requested by SINGLE_CONE_INTEGRATION_PATCH.md.  These arrays
        # are padded because the compact centre and inner rings have fewer phi cells.
        free=compact_field_to_dense(free, system.grid),
        stationary=compact_field_to_dense(stationary, system.grid),
        mobile=compact_field_to_dense(mobile, system.grid),
        # Lossless compact arrays used preferentially by this program.
        free_flat=free,
        stationary_flat=stationary,
        mobile_flat=mobile,
        z=system.grid.z,
        rho=system.grid.rho,
        phi=system.grid.phi,
        z_index=system.grid.z_index,
        ring_index=system.grid.ring_index,
        sector_index=system.grid.sector_index,
        ring_counts=np.asarray(system.grid.ring_counts, dtype=np.int64),
        grid_signature=np.asarray(grid_signature(system.grid)),
        time_start=np.asarray(float(time_start)),
        time_end=np.asarray(float(time_end)),
        source=np.asarray(str(source)),
    )


def outer_boundary_indices(grid: Grid3D) -> Array:
    return np.flatnonzero(grid.ring_index == len(grid.ring_counts) - 1).astype(int)


def _angle_in_window(phi: Array, center: float, width: float) -> NDArray[np.bool_]:
    difference = periodic_angle_difference(phi, center)
    return np.abs(difference) <= 0.5 * width + 1e-12


def interface_configuration(raw: dict[str, Any], grid: Grid3D) -> dict[str, Any]:
    """Resolve the optional top-level ``interface`` configuration."""
    cfg = dict(raw.get("interface", {}))
    outer_count = int(grid.ring_counts[-1])
    neighbours = cfg.get("neighbors") or cfg.get("neighbours")
    if not neighbours:
        width = 360.0 / outer_count
        neighbours = [
            {
                "id": f"NEIGHBOR-{sector:02d}",
                "phi_center_deg": (sector + 0.5) * width,
                "phi_width_deg": width,
            }
            for sector in range(outer_count)
        ]
    return {
        "enabled": bool(cfg.get("enabled", True)),
        "legendre_order": max(0, int(cfg.get("legendre_order", 1))),
        "fourier_order": max(0, int(cfg.get("fourier_order", 0))),
        "species": list(cfg.get("species", ["free", "stationary", "mobile"])),
        "ridge": float(cfg.get("ridge", 1e-10)),
        "neighbors": neighbours,
    }


def _interface_basis(
    z: Array,
    phi: Array,
    *,
    phi_center: float,
    legendre_order: int,
    fourier_order: int,
) -> tuple[Array, list[str]]:
    from numpy.polynomial.legendre import legvander

    zeta = np.clip(2.0 * np.asarray(z, dtype=float) - 1.0, -1.0, 1.0)
    leg = legvander(zeta, legendre_order)
    local_phi = periodic_angle_difference(np.asarray(phi, dtype=float), phi_center)
    columns: list[Array] = []
    labels: list[str] = []
    for ell in range(legendre_order + 1):
        columns.append(leg[:, ell])
        labels.append(f"P{ell}_cos0")
        for mode in range(1, fourier_order + 1):
            columns.append(leg[:, ell] * np.cos(mode * local_phi))
            labels.append(f"P{ell}_cos{mode}")
            columns.append(leg[:, ell] * np.sin(mode * local_phi))
            labels.append(f"P{ell}_sin{mode}")
    return np.column_stack(columns), labels


def build_interface_functionals(
    system: ModelSystem,
    raw: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build weighted Legendre-Fourier coefficient functionals on side patches.

    Each returned row ``weights`` satisfies approximately
    ``coefficient = weights @ boundary_values``.  The weighted least-squares
    construction remains valid on the compact, nonuniform angular mesh.
    """
    config = interface_configuration(raw, system.grid)
    if not config["enabled"]:
        return []
    grid = system.grid
    outer = outer_boundary_indices(grid)
    functionals: list[dict[str, Any]] = []
    for neighbor in config["neighbors"]:
        neighbor_id = str(neighbor.get("id", neighbor.get("neighbor_id", "NEIGHBOR")))
        center = math.radians(float(neighbor.get("phi_center_deg", 0.0)))
        width = math.radians(float(neighbor.get("phi_width_deg", 360.0)))
        mask = _angle_in_window(grid.phi[outer], center, width)
        indices = outer[mask]
        if indices.size == 0:
            raise ValueError(f"Interface patch {neighbor_id} contains no outer-side cells.")
        basis, labels = _interface_basis(
            grid.z[indices], grid.phi[indices], phi_center=center,
            legendre_order=config["legendre_order"],
            fourier_order=config["fourier_order"],
        )
        # Never ask for more independent modes than available cells.
        if basis.shape[1] > indices.size:
            basis = basis[:, : indices.size]
            labels = labels[: indices.size]
        area = np.maximum(grid.outer_side_area[indices], 1e-15)
        gram = basis.T @ (area[:, None] * basis)
        ridge = config["ridge"] * max(float(np.trace(gram)), 1.0)
        functionals_matrix = np.linalg.solve(
            gram + ridge * np.eye(gram.shape[0]),
            basis.T * area[None, :],
        )
        for row, label in zip(functionals_matrix, labels, strict=True):
            functionals.append(
                {
                    "neighbor_id": neighbor_id,
                    "phi_center_deg": math.degrees(center) % 360.0,
                    "phi_width_deg": math.degrees(width),
                    "mode": label,
                    "cell_indices": indices.astype(int),
                    "weights": np.asarray(row, dtype=float),
                    "side_areas": area,
                }
            )
    return functionals


def evaluate_interface_coefficients(
    system: ModelSystem,
    field_state: Array,
    raw: dict[str, Any],
    *,
    source: str,
    time_start: float,
    time_end: float,
) -> dict[str, Any]:
    """Project the final sidewall state into compressed neighbour-interface modes."""
    n = system.grid.z.size
    state = np.asarray(field_state, dtype=float)
    functionals = build_interface_functionals(system, raw)
    config = interface_configuration(raw, system.grid)
    species_offsets = {"free": 0, "stationary": n, "mobile": 2 * n}
    results: list[dict[str, Any]] = []
    for functional in functionals:
        indices = functional["cell_indices"]
        weights = functional["weights"]
        item = {
            "neighbor_id": functional["neighbor_id"],
            "mode": functional["mode"],
            "phi_center_deg": functional["phi_center_deg"],
            "phi_width_deg": functional["phi_width_deg"],
            "cell_indices": indices.tolist(),
            "coefficients": {},
        }
        for species in config["species"]:
            if species not in species_offsets:
                raise ValueError(f"Unknown interface species: {species}")
            offset = species_offsets[species]
            item["coefficients"][species] = float(weights @ state[offset + indices])
        free_values = state[indices]
        flux_density = system.parameters.side_exchange_rate * (
            free_values - system.side_bulk[indices]
        )
        item["free_outward_flux_total"] = float(
            np.sum(functional["side_areas"] * flux_density)
        )
        results.append(item)
    return {
        "protocol": "originq-cone-interface-v1",
        "source": source,
        "preferred_source": source,
        "grid_signature": grid_signature(system.grid),
        "time_start": float(time_start),
        "time_end": float(time_end),
        "legendre_order": config["legendre_order"],
        "fourier_order": config["fourier_order"],
        "species": config["species"],
        "patches": results,
    }


def save_interface_coefficients(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
