"""Validate triangular donor dependencies and antipodal two-arm scheduling."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from layer_geometry import build_layer_geometry, geometry_from_dict


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate triangular Layer geometry and two-arm schedule")
    parser.add_argument("--config", default="configs/smoke/layer_geometry_smoke.json")
    args = parser.parse_args()

    config = Path(args.config).resolve()
    raw = json.loads(config.read_text(encoding="utf-8"))
    geometry = build_layer_geometry(geometry_from_dict(raw))
    n = geometry.cone_count_per_layer
    failures: list[str] = []

    if n % 2:
        failures.append(f"cone count must be even, got {n}")
    if len(geometry.schedule) != len(geometry.nodes):
        failures.append("schedule length differs from node count")
    if len(set(geometry.schedule)) != len(geometry.nodes):
        failures.append("schedule contains duplicate/missing cones")

    target_entries = [e for e in geometry.schedule_entries if e.role == "target"]
    if len(target_entries) != n:
        failures.append(f"expected {n} target entries, got {len(target_entries)}")

    starts = [e for e in target_entries if e.is_start]
    meetings = [e for e in target_entries if e.is_meeting]
    if len(starts) != 1:
        failures.append(f"expected one start sector, got {len(starts)}")
    if len(meetings) != 1:
        failures.append(f"expected one antipodal meeting sector, got {len(meetings)}")
    if starts and meetings:
        expected = (starts[0].sector + n // 2) % n
        if meetings[0].sector != expected:
            failures.append(
                f"meeting sector {meetings[0].sector + 1} is not antipodal to start {starts[0].sector + 1}"
            )
        if len(meetings[0].predecessor_sectors) != 2:
            failures.append("meeting target must accept both half-ring predecessors")

    nodes = geometry.nodes_by_id
    for target_entry in target_entries:
        same_cell = [e for e in geometry.schedule_entries if e.sector == target_entry.sector]
        roles = [e.role for e in same_cell]
        if roles != ["upper", "lower", "target"]:
            failures.append(
                f"sector {target_entry.sector + 1}: execution order is {roles}, expected upper/lower/target"
            )
        if len(same_cell) == 3:
            target = nodes[target_entry.cone_id]
            for cross_role, edge_type in (("upper", "target_upper"), ("lower", "target_lower")):
                donor = next(e for e in same_cell if e.role == cross_role)
                pair = {target.cone_id, donor.cone_id}
                if not any({link.node_a, link.node_b} == pair and link.edge_type == edge_type for link in geometry.links):
                    failures.append(
                        f"sector {target_entry.sector + 1}: {cross_role} donor {donor.cone_id} is not geometrically linked to {target.cone_id}"
                    )
        if not target_entry.is_start and not target_entry.predecessor_sectors:
            failures.append(f"sector {target_entry.sector + 1}: target lacks same-layer predecessor")
        for predecessor_sector in target_entry.predecessor_sectors:
            predecessor_id = f"L1-C{predecessor_sector + 1:03d}"
            pair = {target_entry.cone_id, predecessor_id}
            if not any(
                {link.node_a, link.node_b} == pair and link.edge_type == "same_layer"
                for link in geometry.links
            ):
                failures.append(
                    f"sector {target_entry.sector + 1}: predecessor {predecessor_id} is not a same-layer neighbour"
                )

    half = geometry.parameters.side_patch_count // 2
    for link in geometry.links:
        if (link.a_patch_index + half) % geometry.parameters.side_patch_count != link.b_patch_index:
            failures.append(f"{link.link_id}: patches are not reciprocal")

    sector_order = [entry.sector + 1 for entry in target_entries]
    print("status:", "PASS" if not failures else "FAIL")
    print("cones_per_layer:", n)
    print("total_cones:", len(geometry.nodes))
    print("target_sector_order:", " -> ".join(map(str, sector_order)))
    print("within_cell_order: upper -> lower -> target")
    if starts:
        print("start_sector:", starts[0].sector + 1)
    if meetings:
        print("antipodal_meeting_sector:", meetings[0].sector + 1)
        print("meeting_predecessors:", [v + 1 for v in meetings[0].predecessor_sectors])
    print("patch_reciprocity_failures:", sum("patches are not reciprocal" in item for item in failures))
    if failures:
        for item in failures:
            print("-", item)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
