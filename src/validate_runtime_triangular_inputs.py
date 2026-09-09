"""Audit runtime logs for current upper/lower + directional target handoffs."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def split_field(value: str) -> list[str]:
    return [item for item in (value or "").split(";") if item]


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate triangular runtime inputs")
    parser.add_argument("--output-dir", default="layer_bidirectional_triangular_smoke_output")
    parser.add_argument(
        "--require-meeting",
        action="store_true",
        help="Also require a completed antipodal target with two same-layer predecessors.",
    )
    args = parser.parse_args()
    root = Path(args.output_dir).resolve()
    path = root / "run_summary.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    failures: list[str] = []
    checked = 0
    meeting_checked = False
    for row in rows:
        if row.get("role") != "target" or row.get("status") != "complete":
            continue
        neighbors = split_field(row.get("incoming_neighbors", ""))
        epochs = split_field(row.get("incoming_donor_epochs", ""))
        if row.get("sweep_arm") == "start":
            if len(neighbors) < 2:
                failures.append(f"{row['cone_id']}: start target did not receive upper+lower")
            continue
        checked += 1
        layer_counts = {
            "upper": sum(name.startswith("L2-") for name in neighbors),
            "lower": sum(name.startswith("L3-") for name in neighbors),
            "target": sum(name.startswith("L1-") for name in neighbors),
        }
        expected_target = 2 if row.get("is_antipodal_meeting") == "1" else 1
        if layer_counts["upper"] < 1 or layer_counts["lower"] < 1 or layer_counts["target"] < expected_target:
            failures.append(
                f"{row['cone_id']}: donors={neighbors}; expected upper>=1, lower>=1, target>={expected_target}"
            )
        if epochs and any(value != "current_iteration" for value in epochs):
            failures.append(f"{row['cone_id']}: expected current-iteration donors, got {epochs}")
        if row.get("is_antipodal_meeting") == "1":
            meeting_checked = True

    if checked == 0:
        failures.append("no completed non-start target was found")
    if args.require_meeting and not meeting_checked:
        failures.append("no completed antipodal meeting target was found")

    # Verify side handoffs stay explicitly separate from membrane input.
    handoff = root / "handoff_log.csv"
    if handoff.exists():
        with handoff.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("explicitly_not_membrane_input") not in {"True", "1", "true"}:
                    failures.append(f"{row.get('link_id')}: side handoff is not marked as non-membrane input")

    print("status:", "PASS" if not failures else "FAIL")
    print("completed_non_start_targets_checked:", checked)
    print("antipodal_meeting_checked:", meeting_checked)
    if failures:
        for item in failures:
            print("-", item)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
