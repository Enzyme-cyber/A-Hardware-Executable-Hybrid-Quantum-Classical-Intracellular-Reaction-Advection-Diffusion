"""Draw the staggered triangular cells and bidirectional half-ring schedule."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt

from layer_geometry import build_layer_geometry, geometry_from_dict

ROLE_Y = {"lower": 0.0, "target": 1.0, "upper": 2.0}


def unwrap_near(reference: float, angle: float) -> float:
    return reference + math.atan2(math.sin(angle - reference), math.cos(angle - reference))


def main() -> int:
    parser = argparse.ArgumentParser(description="Preview triangular cells and two-arm schedule")
    parser.add_argument("--config", default="configs/smoke/layer_geometry_smoke.json")
    parser.add_argument("--output", default="bidirectional_triangular_schedule.png")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    raw = json.loads(Path(args.config).read_text(encoding="utf-8"))
    geometry = build_layer_geometry(geometry_from_dict(raw))
    nodes = geometry.nodes_by_id

    fig, ax = plt.subplots(figsize=(15, 6.5), constrained_layout=True)
    # Geometry links in an unwrapped periodic view.
    for link in geometry.links:
        a, b = nodes[link.node_a], nodes[link.node_b]
        bx = unwrap_near(a.angle, b.angle)
        ax.plot(
            [math.degrees(a.angle), math.degrees(bx)],
            [ROLE_Y[a.role], ROLE_Y[b.role]],
            linewidth=0.8,
            alpha=0.22,
        )

    for role in ("lower", "target", "upper"):
        group = [node for node in geometry.nodes if node.role == role]
        ax.scatter(
            [math.degrees(node.angle) for node in group],
            [ROLE_Y[role]] * len(group),
            s=150,
            edgecolors="black",
            linewidths=0.6,
            label=role,
            zorder=3,
        )
        for node in group:
            ax.text(math.degrees(node.angle), ROLE_Y[role] + 0.10, node.cone_id, ha="center", fontsize=7)

    target_entries = [entry for entry in geometry.schedule_entries if entry.role == "target"]
    # Draw the two propagation arms on the target row.
    for previous, current in zip(target_entries, target_entries[1:]):
        # Do not draw a false jump between the two independent arms.
        if previous.sweep_arm == "clockwise" and current.sweep_arm == "counterclockwise":
            continue
        x0 = math.degrees(nodes[previous.cone_id].angle)
        x1 = math.degrees(unwrap_near(nodes[previous.cone_id].angle, nodes[current.cone_id].angle))
        ax.annotate(
            "",
            xy=(x1, ROLE_Y["target"] + 0.02),
            xytext=(x0, ROLE_Y["target"] + 0.02),
            arrowprops={"arrowstyle": "-|>", "linewidth": 2.0, "alpha": 0.8},
            zorder=4,
        )

    # Add start-to-counterclockwise first arm and both arrows into the meeting target.
    by_sector = {entry.sector: entry for entry in target_entries}
    start = next(entry for entry in target_entries if entry.is_start)
    cc_first = next((entry for entry in target_entries if entry.sweep_arm == "counterclockwise"), None)
    if cc_first:
        x0 = math.degrees(nodes[start.cone_id].angle)
        x1 = math.degrees(unwrap_near(nodes[start.cone_id].angle, nodes[cc_first.cone_id].angle))
        ax.annotate("", xy=(x1, 1.02), xytext=(x0, 1.02), arrowprops={"arrowstyle":"-|>","linewidth":2.0,"alpha":0.8})
    meeting = next(entry for entry in target_entries if entry.is_meeting)
    for pred_sector in meeting.predecessor_sectors:
        pred = by_sector[pred_sector]
        x0 = math.degrees(nodes[pred.cone_id].angle)
        x1 = math.degrees(unwrap_near(nodes[pred.cone_id].angle, nodes[meeting.cone_id].angle))
        ax.annotate("", xy=(x1, 1.02), xytext=(x0, 1.02), arrowprops={"arrowstyle":"-|>","linewidth":2.0,"alpha":0.8})

    for entry in target_entries:
        x = math.degrees(nodes[entry.cone_id].angle)
        label = f"cell {entry.sector + 1}\n{entry.sweep_arm}"
        if entry.is_meeting:
            label += "\nMEET"
        ax.text(x, 0.66, label, ha="center", fontsize=8)

    ax.set_yticks([0, 1, 2], ["lower", "target / centre", "upper"])
    ax.set_xlabel("global azimuth (degrees; periodic)")
    ax.set_xlim(-20, 380)
    ax.set_ylim(-0.35, 2.45)
    ax.grid(axis="x", alpha=0.22)
    ax.set_title(
        "Triangular Layer dependency: upper + lower + preceding target; two half-ring sweeps meet antipodally"
    )
    ax.legend(loc="upper right")
    fig.savefig(args.output, dpi=190)
    if args.show:
        plt.show()
    plt.close(fig)
    print(Path(args.output).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
