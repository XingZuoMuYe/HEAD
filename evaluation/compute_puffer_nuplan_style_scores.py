"""Compute a nuPlan-inspired score from PufferDrive per-map JSONL files.

This is additive to existing evaluation outputs. It does not replace the
native score and does not require rerunning rollouts.
"""

import argparse
import csv
import json
import math
import re
import statistics
from pathlib import Path


FORMULA_VERSION = "puffer_nuplan_style_score_v1"
WEIGHTS = {
    "completion_rate": 5.0,
    "ttc": 5.0,
    "comfort": 2.0,
}
TTC_COMFORT_FIELDS = {
    "strict": {
        "ttc": "ttc_within_bound_rate",
        "comfort": "comfortable_rate",
    },
    "frame": {
        "ttc": "ttc_safe_frame_rate",
        "comfort": "comfort_frame_rate",
    },
}
BASE_RATE_FIELDS = ("completion_rate", "collision_rate", "offroad_rate")


def parse_experiment(value):
    """Parse a LABEL=PATH command-line value."""
    label, separator, path = value.partition("=")
    if not separator or not label.strip() or not path.strip():
        raise argparse.ArgumentTypeError(
            "--experiment must use LABEL=/path/to/per_map.jsonl"
        )
    return label.strip(), Path(path).expanduser()


def _rate(row, field, source, line_number):
    if field not in row:
        raise ValueError(f"{source}:{line_number} is missing {field}")
    try:
        value = float(row[field])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{source}:{line_number} has a non-numeric {field}: {row[field]!r}"
        ) from exc
    if not math.isfinite(value):
        raise ValueError(f"{source}:{line_number} has a non-finite {field}")
    tolerance = 1e-6
    if value < -tolerance or value > 1.0 + tolerance:
        raise ValueError(
            f"{source}:{line_number} has {field}={value}, expected [0, 1]"
        )
    return min(1.0, max(0.0, value))


def score_scenario(
    row,
    source="<row>",
    line_number=1,
    ttc_comfort_mode="strict",
):
    """Return one compact per-scenario score record."""
    if ttc_comfort_mode not in TTC_COMFORT_FIELDS:
        raise ValueError(
            f"Unknown TTC/Comfort mode {ttc_comfort_mode!r}; "
            f"expected one of {sorted(TTC_COMFORT_FIELDS)}"
        )
    selected_fields = TTC_COMFORT_FIELDS[ttc_comfort_mode]
    rate_fields = (*BASE_RATE_FIELDS, *selected_fields.values())
    rates = {
        field: _rate(row, field, source, line_number)
        for field in rate_fields
    }
    selected_ttc_rate = rates[selected_fields["ttc"]]
    selected_comfort_rate = rates[selected_fields["comfort"]]
    quality_score = (
        WEIGHTS["completion_rate"] * rates["completion_rate"]
        + WEIGHTS["ttc"] * selected_ttc_rate
        + WEIGHTS["comfort"] * selected_comfort_rate
    ) / sum(WEIGHTS.values())
    safety_factor = (
        (1.0 - rates["collision_rate"])
        * (1.0 - rates["offroad_rate"])
    )
    return {
        "map_id": row.get("map_id"),
        "scenario_id": row.get("scenario_id"),
        "controlled_agents": row.get("n"),
        "native_puffer_score": row.get(
            "native_puffer_score",
            row.get("score"),
        ),
        **rates,
        "ttc_comfort_mode": ttc_comfort_mode,
        "selected_ttc_field": selected_fields["ttc"],
        "selected_comfort_field": selected_fields["comfort"],
        "selected_ttc_rate": selected_ttc_rate,
        "selected_comfort_rate": selected_comfort_rate,
        "quality_score": quality_score,
        "safety_factor": safety_factor,
        FORMULA_VERSION: quality_score * safety_factor,
    }

def load_and_score(path, ttc_comfort_mode="strict"):
    """Read and score every non-empty JSONL row."""
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = []
    keys = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            row = score_scenario(
                raw,
                str(path),
                line_number,
                ttc_comfort_mode=ttc_comfort_mode,
            )
            key = (row["map_id"], row["scenario_id"])
            if key in keys:
                raise ValueError(f"Duplicate scenario key {key!r} in {path}")
            keys.add(key)
            rows.append(row)
    if not rows:
        raise ValueError(f"No scenario rows found in {path}")
    return rows, keys


def _percentile(values, fraction):
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    position = fraction * (len(values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return values[lower]
    ratio = position - lower
    return values[lower] * (1.0 - ratio) + values[upper] * ratio


def summarize(label, source, rows):
    """Summarize using an equal-weight mean across scenarios."""
    modes = {row["ttc_comfort_mode"] for row in rows}
    if len(modes) != 1:
        raise ValueError(f"Rows contain mixed TTC/Comfort modes: {sorted(modes)}")
    ttc_comfort_mode = modes.pop()
    selected_fields = TTC_COMFORT_FIELDS[ttc_comfort_mode]
    scores = [row[FORMULA_VERSION] for row in rows]
    summary = {
        "label": label,
        "source": str(Path(source).resolve()),
        "formula_version": FORMULA_VERSION,
        "ttc_comfort_mode": ttc_comfort_mode,
        "ttc_rate_field": selected_fields["ttc"],
        "comfort_rate_field": selected_fields["comfort"],
        "formula": (
            f"((5*completion_rate + 5*{selected_fields['ttc']} + "
            f"2*{selected_fields['comfort']}) / 12) * "
            "(1-collision_rate) * (1-offroad_rate)"
        ),
        "aggregation": "equal arithmetic mean over per-map scenario scores",
        "scenario_count": len(rows),
        "score": statistics.fmean(scores),
        "score_std": statistics.pstdev(scores),
        "score_median": statistics.median(scores),
        "score_p05": _percentile(scores, 0.05),
        "score_p95": _percentile(scores, 0.95),
        "zero_score_rate": statistics.fmean(score <= 1e-12 for score in scores),
    }
    for field in (
        "quality_score",
        "safety_factor",
        *BASE_RATE_FIELDS,
        "selected_ttc_rate",
        "selected_comfort_rate",
        *selected_fields.values(),
    ):
        summary[f"mean_{field}"] = statistics.fmean(
            float(row[field]) for row in rows
        )
    return summary

def _safe_label(label):
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("_.")
    return value or "experiment"


def _write_json(path, value):
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")

def _write_csv(path, summaries):
    fields = (
        "label", "ttc_comfort_mode", "ttc_rate_field", "comfort_rate_field",
        "scenario_count", "score", "score_std", "score_median",
        "score_p05", "score_p95", "zero_score_rate", "mean_quality_score",
        "mean_safety_factor", "mean_completion_rate",
        "mean_selected_ttc_rate", "mean_selected_comfort_rate",
        "mean_ttc_within_bound_rate", "mean_comfortable_rate",
        "mean_ttc_safe_frame_rate", "mean_comfort_frame_rate",
        "mean_collision_rate", "mean_offroad_rate", "source",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summaries)

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        action="append",
        type=parse_experiment,
        required=True,
        metavar="LABEL=PER_MAP_JSONL",
        help="Repeat once per experiment.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--ttc-comfort-mode",
        choices=tuple(TTC_COMFORT_FIELDS),
        default="strict",
        help=(
            "Choose trajectory-level strict rates or valid-frame rates for "
            "the TTC and Comfort quality terms (default: strict)."
        ),
    )
    parser.add_argument(
        "--allow-scenario-mismatch",
        action="store_true",
        help="Allow experiments to contain different map/scenario key sets.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    labels = [label for label, _ in args.experiment]
    if len(set(labels)) != len(labels):
        raise ValueError("Experiment labels must be unique")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    reference_label = None
    reference_keys = None
    for label, path in args.experiment:
        rows, keys = load_and_score(
            path,
            ttc_comfort_mode=args.ttc_comfort_mode,
        )
        if reference_keys is None:
            reference_label, reference_keys = label, keys
        elif keys != reference_keys and not args.allow_scenario_mismatch:
            missing = len(reference_keys - keys)
            extra = len(keys - reference_keys)
            raise ValueError(
                f"Scenario set for {label!r} differs from {reference_label!r}: "
                f"missing={missing}, extra={extra}. Use --allow-scenario-mismatch "
                "only when this comparison is intentional."
            )

        summary = summarize(label, path, rows)
        summaries.append(summary)
        _write_jsonl(
            args.output_dir / f"{_safe_label(label)}_per_scenario_scores.jsonl",
            rows,
        )

    baseline = summaries[0]["score"]
    for summary in summaries:
        summary["delta_from_first"] = summary["score"] - baseline

    comparison = {
        "formula_version": FORMULA_VERSION,
        "ttc_comfort_mode": args.ttc_comfort_mode,
        "ttc_rate_field": TTC_COMFORT_FIELDS[args.ttc_comfort_mode]["ttc"],
        "comfort_rate_field": TTC_COMFORT_FIELDS[args.ttc_comfort_mode][
            "comfort"
        ],
        "weights": {
            "completion_rate": WEIGHTS["completion_rate"],
            TTC_COMFORT_FIELDS[args.ttc_comfort_mode]["ttc"]: WEIGHTS["ttc"],
            TTC_COMFORT_FIELDS[args.ttc_comfort_mode]["comfort"]: WEIGHTS[
                "comfort"
            ],
        },
        "safety_factor": "(1-collision_rate) * (1-offroad_rate)",
        "scenario_aggregation": "equal arithmetic mean",
        "experiments": summaries,
    }
    _write_json(args.output_dir / "nuplan_style_score_comparison.json", comparison)
    _write_csv(args.output_dir / "nuplan_style_score_comparison.csv", summaries)

    print(
        f"TTC/Comfort mode: {args.ttc_comfort_mode} "
        f"({TTC_COMFORT_FIELDS[args.ttc_comfort_mode]['ttc']}, "
        f"{TTC_COMFORT_FIELDS[args.ttc_comfort_mode]['comfort']})"
    )
    print(
        f"{'Experiment':24s} {'Maps':>7s} {'Score':>9s} {'Quality':>9s} "
        f"{'Safety':>9s} {'Delta':>9s}"
    )
    print("-" * 72)
    for summary in summaries:
        print(
            f"{summary['label'][:24]:24s} "
            f"{summary['scenario_count']:7,d} "
            f"{summary['score']:9.4f} "
            f"{summary['mean_quality_score']:9.4f} "
            f"{summary['mean_safety_factor']:9.4f} "
            f"{summary['delta_from_first']:+9.4f}"
        )
    print(f"Wrote results to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
