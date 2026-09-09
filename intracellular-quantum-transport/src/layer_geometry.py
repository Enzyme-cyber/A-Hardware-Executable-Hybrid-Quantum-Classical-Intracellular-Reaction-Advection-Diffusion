"""Exact three-layer tiled geometry for compact 3-D cone subdomains.

Coordinate conventions
----------------------
* Global geometry: cones are placed on three latitude rings of a sphere.
* Every cone axis points from the plasma membrane toward the cell centre.
* Local side angle phi=0 points along increasing global azimuth (RIGHT).
* Local phi=+pi/2 points north/upward (TOP).
* Local phi=pi points along decreasing global azimuth (LEFT).
* Local phi=-pi/2 points south/downward (BOTTOM).

The physical contact direction is computed from the actual 3-D displacement and
projected into each cone's local tangent/vertical frame.  It is then mapped to a
reciprocal pair of the configured compact-cone side patches.  This prevents a donor and
receiver from accidentally using unrelated sides after layer staggering.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable
import math

import numpy as np


ROLE_NAMES = ("target", "upper", "lower")
CARDINAL_CENTERS = {
    "right": 0.0,
    "top": 0.5 * math.pi,
    "left": math.pi,
    "bottom": 1.5 * math.pi,
}
OPPOSITE_CARDINAL = {
    "right": "left",
    "left": "right",
    "top": "bottom",
    "bottom": "top",
}


@dataclass(frozen=True)
class GeometryParameters:
    layers: int = 3
    cell_circumference: float = 24.0
    step_length: float = 3.0
    coverage_fraction: float = 1.0
    minimum_cones_per_layer: int = 3
    layer_offsets_fraction: tuple[float, float, float] = (
        0.0,
        1.0 / 3.0,
        2.0 / 3.0,
    )
    layer_polar_angles_deg: tuple[float, float, float] = (90.0, 70.0, 110.0)
    sphere_radius: float | None = None
    start_angle_deg: float = 0.0
    side_patch_count: int = 6
    same_layer_weight: float = 1.0
    cross_layer_weight: float = 0.55

    # Scheduling / triangular dependency graph.
    # One sector is solved as upper -> lower -> target, so the target receives
    # the current upper/lower interfaces plus the already-computed target from
    # the preceding sector on the active sweep arm.
    schedule_mode: str = "bidirectional_half_ring"
    within_sector_order: tuple[str, str, str] = ("upper", "lower", "target")
    require_even_cone_count: bool = True
    start_sector: int = 0


@dataclass(frozen=True)
class ConeNode:
    cone_id: str
    layer: int
    role: str
    sector: int
    order_index: int
    angle: float
    polar_angle: float
    position: tuple[float, float, float]
    axis: tuple[float, float, float]
    tangent: tuple[float, float, float]
    vertical: tuple[float, float, float]


@dataclass(frozen=True)
class InterfaceLink:
    link_id: str
    node_a: str
    node_b: str
    edge_type: str
    weight: float
    distance: float
    a_local_phi: float
    b_local_phi: float
    a_side: str
    b_side: str
    a_patch_index: int
    b_patch_index: int
    a_patch_id: str
    b_patch_id: str
    actual_opposition_error_deg: float

    def oriented(self, donor_id: str, receiver_id: str) -> "DirectedInterface":
        if donor_id == self.node_a and receiver_id == self.node_b:
            return DirectedInterface(
                link_id=self.link_id,
                donor_id=donor_id,
                receiver_id=receiver_id,
                edge_type=self.edge_type,
                weight=self.weight,
                distance=self.distance,
                donor_local_phi=self.a_local_phi,
                receiver_local_phi=self.b_local_phi,
                donor_side=self.a_side,
                receiver_side=self.b_side,
                donor_patch_index=self.a_patch_index,
                receiver_patch_index=self.b_patch_index,
                donor_patch_id=self.a_patch_id,
                receiver_patch_id=self.b_patch_id,
            )
        if donor_id == self.node_b and receiver_id == self.node_a:
            return DirectedInterface(
                link_id=self.link_id,
                donor_id=donor_id,
                receiver_id=receiver_id,
                edge_type=self.edge_type,
                weight=self.weight,
                distance=self.distance,
                donor_local_phi=self.b_local_phi,
                receiver_local_phi=self.a_local_phi,
                donor_side=self.b_side,
                receiver_side=self.a_side,
                donor_patch_index=self.b_patch_index,
                receiver_patch_index=self.a_patch_index,
                donor_patch_id=self.b_patch_id,
                receiver_patch_id=self.a_patch_id,
            )
        raise ValueError(f"{donor_id}->{receiver_id} is not link {self.link_id}")


@dataclass(frozen=True)
class DirectedInterface:
    link_id: str
    donor_id: str
    receiver_id: str
    edge_type: str
    weight: float
    distance: float
    donor_local_phi: float
    receiver_local_phi: float
    donor_side: str
    receiver_side: str
    donor_patch_index: int
    receiver_patch_index: int
    donor_patch_id: str
    receiver_patch_id: str


@dataclass(frozen=True)
class ScheduleEntry:
    order: int
    cone_id: str
    role: str
    sector: int
    sweep_arm: str
    sweep_direction: int
    predecessor_sectors: tuple[int, ...]
    is_start: bool = False
    is_meeting: bool = False


@dataclass(frozen=True)
class LayerGeometry:
    parameters: GeometryParameters
    cone_count_per_layer: int
    sphere_radius: float
    angular_step: float
    nodes: tuple[ConeNode, ...]
    links: tuple[InterfaceLink, ...]
    schedule: tuple[str, ...]
    schedule_entries: tuple[ScheduleEntry, ...]

    @property
    def nodes_by_id(self) -> dict[str, ConeNode]:
        return {node.cone_id: node for node in self.nodes}

    @property
    def links_by_node(self) -> dict[str, list[InterfaceLink]]:
        result = {node.cone_id: [] for node in self.nodes}
        for link in self.links:
            result[link.node_a].append(link)
            result[link.node_b].append(link)
        return result

    @property
    def schedule_by_cone(self) -> dict[str, ScheduleEntry]:
        return {entry.cone_id: entry for entry in self.schedule_entries}

    @property
    def schedule_by_sector_role(self) -> dict[tuple[int, str], ScheduleEntry]:
        return {(entry.sector, entry.role): entry for entry in self.schedule_entries}

    def incoming_links(self, receiver_id: str) -> list[DirectedInterface]:
        return [
            link.oriented(
                link.node_b if link.node_a == receiver_id else link.node_a,
                receiver_id,
            )
            for link in self.links_by_node[receiver_id]
        ]


def geometry_from_dict(raw: dict[str, Any]) -> GeometryParameters:
    data = dict(raw.get("geometry", raw.get("ring", {})))
    # Friendly aliases from older packages.
    if "layer_polar_angles" in data and "layer_polar_angles_deg" not in data:
        data["layer_polar_angles_deg"] = data.pop("layer_polar_angles")
    for key in ("layer_offsets_fraction", "layer_polar_angles_deg"):
        if key in data:
            data[key] = tuple(float(value) for value in data[key])
    if "within_sector_order" in data:
        data["within_sector_order"] = tuple(str(value) for value in data["within_sector_order"])
    valid = set(GeometryParameters.__dataclass_fields__)
    unknown = sorted(set(data) - valid)
    if unknown:
        raise ValueError(f"Unknown geometry parameters: {unknown}")
    result = GeometryParameters(**data)
    if result.layers != 3:
        raise ValueError("This exact tiled geometry requires layers=3")
    if len(result.layer_offsets_fraction) != 3:
        raise ValueError("layer_offsets_fraction must contain target/upper/lower values")
    if len(result.layer_polar_angles_deg) != 3:
        raise ValueError("layer_polar_angles_deg must contain target/upper/lower values")
    if result.cell_circumference <= 0.0 or result.step_length <= 0.0:
        raise ValueError("cell_circumference and step_length must be positive")
    if not (0.0 < result.coverage_fraction <= 1.0):
        raise ValueError("coverage_fraction must be in (0,1]")
    if result.side_patch_count < 4 or result.side_patch_count % 2:
        raise ValueError("side_patch_count must be an even integer >= 4")
    if result.schedule_mode != "bidirectional_half_ring":
        raise ValueError("schedule_mode must be bidirectional_half_ring")
    if tuple(result.within_sector_order) != ("upper", "lower", "target"):
        raise ValueError("within_sector_order must be [upper, lower, target]")
    return result


def wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def positive_angle(angle: float) -> float:
    return angle % (2.0 * math.pi)


def angular_distance(a: float, b: float) -> float:
    return abs(wrap_angle(a - b))


def _unit(values: Iterable[float]) -> np.ndarray:
    vector = np.asarray(tuple(values), dtype=float)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-14:
        raise ValueError("Cannot normalise zero vector")
    return vector / norm


def _local_frame(theta: float, alpha: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Spherical outward radial unit vector.
    outward = np.array(
        [
            math.sin(theta) * math.cos(alpha),
            math.sin(theta) * math.sin(alpha),
            math.cos(theta),
        ],
        dtype=float,
    )
    # Cone axis is membrane -> cell centre/nucleus.
    axis = -outward
    # Increasing azimuth is local RIGHT.
    tangent = np.array([-math.sin(alpha), math.cos(alpha), 0.0], dtype=float)
    # Decreasing polar angle (northward) is local TOP.
    e_theta = np.array(
        [
            math.cos(theta) * math.cos(alpha),
            math.cos(theta) * math.sin(alpha),
            -math.sin(theta),
        ],
        dtype=float,
    )
    vertical = -e_theta
    return _unit(outward), _unit(axis), _unit(tangent), _unit(vertical)


def local_interface_phi(source: ConeNode, target: ConeNode) -> float:
    displacement = np.asarray(target.position, dtype=float) - np.asarray(
        source.position, dtype=float
    )
    axis = np.asarray(source.axis, dtype=float)
    transverse = displacement - float(np.dot(displacement, axis)) * axis
    t = float(np.dot(transverse, np.asarray(source.tangent, dtype=float)))
    v = float(np.dot(transverse, np.asarray(source.vertical, dtype=float)))
    if abs(t) + abs(v) <= 1e-14:
        return 0.0
    return positive_angle(math.atan2(v, t))


def semantic_side(phi: float) -> str:
    return min(CARDINAL_CENTERS, key=lambda name: angular_distance(phi, CARDINAL_CENTERS[name]))


def patch_center(index: int, patch_count: int) -> float:
    return (index + 0.5) * (2.0 * math.pi / patch_count)


def reciprocal_patch_pair(
    phi_a: float,
    phi_b: float,
    patch_count: int,
) -> tuple[int, int]:
    """Choose an exactly opposite patch pair minimizing both angular errors."""
    half = patch_count // 2
    candidates: list[tuple[float, int, int]] = []
    for a_index in range(patch_count):
        b_index = (a_index + half) % patch_count
        error = angular_distance(phi_a, patch_center(a_index, patch_count))
        error += angular_distance(phi_b, patch_center(b_index, patch_count))
        candidates.append((error, a_index, b_index))
    _, a_index, b_index = min(candidates, key=lambda item: item[0])
    return a_index, b_index


def _make_link(
    node_a: ConeNode,
    node_b: ConeNode,
    edge_type: str,
    weight: float,
    patch_count: int,
    serial: int,
) -> InterfaceLink:
    phi_a = local_interface_phi(node_a, node_b)
    phi_b = local_interface_phi(node_b, node_a)
    a_patch, b_patch = reciprocal_patch_pair(phi_a, phi_b, patch_count)
    opposition_error = abs(
        math.degrees(abs(wrap_angle(phi_b - phi_a)) - math.pi)
    )
    return InterfaceLink(
        link_id=f"E{serial:04d}",
        node_a=node_a.cone_id,
        node_b=node_b.cone_id,
        edge_type=edge_type,
        weight=float(weight),
        distance=float(
            np.linalg.norm(
                np.asarray(node_b.position, dtype=float)
                - np.asarray(node_a.position, dtype=float)
            )
        ),
        a_local_phi=phi_a,
        b_local_phi=phi_b,
        a_side=semantic_side(phi_a),
        b_side=semantic_side(phi_b),
        a_patch_index=a_patch,
        b_patch_index=b_patch,
        a_patch_id=f"SIDE_{a_patch:02d}",
        b_patch_id=f"SIDE_{b_patch:02d}",
        actual_opposition_error_deg=float(opposition_error),
    )


def _nearest_sector(nodes: list[ConeNode], global_angle: float) -> ConeNode:
    return min(nodes, key=lambda node: angular_distance(node.angle, global_angle))


def bidirectional_sector_plan(count: int, start_sector: int = 0) -> list[dict[str, Any]]:
    """Return two sweep arms that meet exactly at the antipodal sector.

    For N=8 and start=0 the sector order is::

        0, 1, 2, 3, 7, 6, 5, 4

    Sector 4 is deliberately solved last, after both arms have reached its two
    neighbours.  This avoids concentrating the cyclic seam error at sector 0.
    """
    if count < 4:
        raise ValueError("bidirectional schedule requires at least four sectors")
    if count % 2:
        raise ValueError(
            "An exact antipodal meeting sector requires an even cone count per layer"
        )
    start = int(start_sector) % count
    opposite = (start + count // 2) % count
    clockwise = [((start + step) % count) for step in range(1, count // 2)]
    counterclockwise = [((start - step) % count) for step in range(1, count // 2)]
    plan: list[dict[str, Any]] = [
        {
            "sector": start,
            "sweep_arm": "start",
            "sweep_direction": 0,
            "predecessor_sectors": (),
            "is_start": True,
            "is_meeting": False,
        }
    ]
    previous = start
    for sector in clockwise:
        plan.append(
            {
                "sector": sector,
                "sweep_arm": "clockwise",
                "sweep_direction": +1,
                "predecessor_sectors": (previous,),
                "is_start": False,
                "is_meeting": False,
            }
        )
        previous = sector
    previous = start
    for sector in counterclockwise:
        plan.append(
            {
                "sector": sector,
                "sweep_arm": "counterclockwise",
                "sweep_direction": -1,
                "predecessor_sectors": (previous,),
                "is_start": False,
                "is_meeting": False,
            }
        )
        previous = sector
    cw_last = clockwise[-1] if clockwise else start
    ccw_last = counterclockwise[-1] if counterclockwise else start
    plan.append(
        {
            "sector": opposite,
            "sweep_arm": "meeting",
            "sweep_direction": 0,
            "predecessor_sectors": (cw_last, ccw_last),
            "is_start": False,
            "is_meeting": True,
        }
    )
    return plan


def build_layer_geometry(parameters: GeometryParameters) -> LayerGeometry:
    p = parameters
    full_ring = math.isclose(p.coverage_fraction, 1.0, abs_tol=1e-12)
    count = max(
        p.minimum_cones_per_layer,
        int(math.ceil(p.cell_circumference * p.coverage_fraction / p.step_length)),
    )
    if p.require_even_cone_count and count % 2:
        raise ValueError(
            f"cone_count_per_layer={count} is odd; change circumference/step so the two half-ring sweeps meet at one antipodal sector"
        )
    span = 2.0 * math.pi * p.coverage_fraction
    angular_step = (2.0 * math.pi / count) if full_ring else span / max(count - 1, 1)
    target_theta = math.radians(p.layer_polar_angles_deg[0])
    denominator = 2.0 * math.pi * max(math.sin(target_theta), 1e-12)
    sphere_radius = (
        float(p.sphere_radius)
        if p.sphere_radius is not None
        else p.cell_circumference / denominator
    )

    nodes_by_layer: list[list[ConeNode]] = []
    all_nodes: list[ConeNode] = []
    start = math.radians(p.start_angle_deg)
    for layer in range(3):
        theta = math.radians(p.layer_polar_angles_deg[layer])
        offset = p.layer_offsets_fraction[layer] * angular_step
        layer_nodes: list[ConeNode] = []
        for sector in range(count):
            alpha = positive_angle(start + offset + sector * angular_step)
            outward, axis, tangent, vertical = _local_frame(theta, alpha)
            position = sphere_radius * outward
            order_index = sector * 3 + layer
            node = ConeNode(
                cone_id=f"L{layer+1}-C{sector+1:03d}",
                layer=layer,
                role=ROLE_NAMES[layer],
                sector=sector,
                order_index=order_index,
                angle=alpha,
                polar_angle=theta,
                position=tuple(float(v) for v in position),
                axis=tuple(float(v) for v in axis),
                tangent=tuple(float(v) for v in tangent),
                vertical=tuple(float(v) for v in vertical),
            )
            layer_nodes.append(node)
            all_nodes.append(node)
        nodes_by_layer.append(layer_nodes)

    links: list[InterfaceLink] = []
    seen_pairs: set[tuple[str, str]] = set()

    def add(a: ConeNode, b: ConeNode, edge_type: str, weight: float) -> None:
        pair = tuple(sorted((a.cone_id, b.cone_id)))
        if pair in seen_pairs:
            return
        seen_pairs.add(pair)
        links.append(
            _make_link(a, b, edge_type, weight, p.side_patch_count, len(links) + 1)
        )

    # Same-layer cyclic contacts: increasing azimuth is right, decreasing is left.
    for layer_nodes in nodes_by_layer:
        for sector, node in enumerate(layer_nodes):
            if sector + 1 < count:
                other = layer_nodes[sector + 1]
            elif full_ring:
                other = layer_nodes[0]
            else:
                continue
            add(node, other, "same_layer", p.same_layer_weight)

    # Exact branch geometry: the target layer is the bridge.  Each target cone
    # couples to the nearest staggered upper cone and nearest staggered lower cone.
    # Keep the resolved pair so the scheduler can solve both cross-layer donors
    # immediately before their target cone, even when the lower-layer index is
    # shifted by the 2/3-sector staggering.
    cross_neighbor_by_target: dict[tuple[str, str], ConeNode] = {}
    for target in nodes_by_layer[0]:
        upper = _nearest_sector(nodes_by_layer[1], target.angle)
        lower = _nearest_sector(nodes_by_layer[2], target.angle)
        cross_neighbor_by_target[(target.cone_id, "upper")] = upper
        cross_neighbor_by_target[(target.cone_id, "lower")] = lower
        add(target, upper, "target_upper", p.cross_layer_weight)
        add(target, lower, "target_lower", p.cross_layer_weight)

    # In the staggered three-row mesh every upper/lower cone must belong to one
    # target-centred triangular cell.  This is true for the default 0,1/3,2/3
    # offsets and is checked explicitly rather than silently duplicating a cone.
    for role, layer in (("upper", 1), ("lower", 2)):
        resolved = [
            cross_neighbor_by_target[(target.cone_id, role)].cone_id
            for target in nodes_by_layer[0]
        ]
        if len(set(resolved)) != count:
            raise ValueError(
                f"{role} nearest-neighbour mapping is not one-to-one; adjust layer offsets before using triangular scheduling"
            )

    role_to_layer = {"target": 0, "upper": 1, "lower": 2}
    sector_plan = bidirectional_sector_plan(count, p.start_sector)
    schedule_entries: list[ScheduleEntry] = []
    for sector_item in sector_plan:
        sector = int(sector_item["sector"])
        target_node = nodes_by_layer[0][sector]
        for role in p.within_sector_order:
            if role == "target":
                node = target_node
            else:
                node = cross_neighbor_by_target[(target_node.cone_id, role)]
            schedule_entries.append(
                ScheduleEntry(
                    order=len(schedule_entries) + 1,
                    cone_id=node.cone_id,
                    role=role,
                    sector=sector,
                    sweep_arm=str(sector_item["sweep_arm"]),
                    sweep_direction=int(sector_item["sweep_direction"]),
                    predecessor_sectors=tuple(
                        int(value) for value in sector_item["predecessor_sectors"]
                    ),
                    is_start=bool(sector_item["is_start"]),
                    is_meeting=bool(sector_item["is_meeting"]),
                )
            )
    schedule = tuple(entry.cone_id for entry in schedule_entries)
    return LayerGeometry(
        parameters=p,
        cone_count_per_layer=count,
        sphere_radius=sphere_radius,
        angular_step=angular_step,
        nodes=tuple(sorted(all_nodes, key=lambda node: node.order_index)),
        links=tuple(links),
        schedule=schedule,
        schedule_entries=tuple(schedule_entries),
    )


def geometry_summary(geometry: LayerGeometry) -> dict[str, Any]:
    by_type: dict[str, int] = {}
    for link in geometry.links:
        by_type[link.edge_type] = by_type.get(link.edge_type, 0) + 1
    return {
        "cone_count_per_layer": geometry.cone_count_per_layer,
        "total_cones": len(geometry.nodes),
        "sphere_radius": geometry.sphere_radius,
        "angular_step_deg": math.degrees(geometry.angular_step),
        "layer_offsets_fraction": list(geometry.parameters.layer_offsets_fraction),
        "layer_polar_angles_deg": list(geometry.parameters.layer_polar_angles_deg),
        "side_patch_count": geometry.parameters.side_patch_count,
        "link_count": len(geometry.links),
        "links_by_type": by_type,
        "schedule_mode": geometry.parameters.schedule_mode,
        "within_sector_order": list(geometry.parameters.within_sector_order),
        "sector_sweep": [
            {
                "sector": entry.sector + 1,
                "arm": entry.sweep_arm,
                "direction": entry.sweep_direction,
                "predecessors": [value + 1 for value in entry.predecessor_sectors],
                "meeting": entry.is_meeting,
            }
            for entry in geometry.schedule_entries
            if entry.role == "target"
        ],
        "schedule": "start; clockwise half-arm; counterclockwise half-arm; antipodal meeting. Within each sector: upper -> lower -> target",
    }
