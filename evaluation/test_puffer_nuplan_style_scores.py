import json
import tempfile
import unittest
from pathlib import Path

from evaluation.compute_puffer_nuplan_style_scores import (
    FORMULA_VERSION,
    load_and_score,
    score_scenario,
    summarize,
)


class PufferNuPlanStyleScoreTest(unittest.TestCase):
    def test_scenario_formula_uses_strict_metrics_and_soft_safety(self):
        row = score_scenario(
            {
                "map_id": 7,
                "scenario_id": "scene",
                "n": 4,
                "score": 0.9,
                "completion_rate": 0.8,
                "ttc_within_bound_rate": 0.6,
                "comfortable_rate": 0.5,
                "ttc_safe_frame_rate": 0.0,
                "comfort_frame_rate": 0.0,
                "collision_rate": 0.25,
                "offroad_rate": 0.2,
            }
        )

        expected_quality = (5 * 0.8 + 5 * 0.6 + 2 * 0.5) / 12
        expected_safety = (1 - 0.25) * (1 - 0.2)
        self.assertAlmostEqual(row["quality_score"], expected_quality)
        self.assertAlmostEqual(row["safety_factor"], expected_safety)
        self.assertAlmostEqual(
            row[FORMULA_VERSION], expected_quality * expected_safety
        )
        self.assertEqual(row["ttc_comfort_mode"], "strict")
        self.assertEqual(row["selected_ttc_field"], "ttc_within_bound_rate")
        self.assertEqual(row["selected_comfort_field"], "comfortable_rate")

    def test_scenario_formula_can_use_frame_metrics(self):
        row = score_scenario(
            {
                "map_id": 7,
                "scenario_id": "scene",
                "completion_rate": 0.8,
                "ttc_within_bound_rate": 0.0,
                "comfortable_rate": 0.0,
                "ttc_safe_frame_rate": 0.9,
                "comfort_frame_rate": 0.7,
                "collision_rate": 0.25,
                "offroad_rate": 0.2,
            },
            ttc_comfort_mode="frame",
        )

        expected_quality = (5 * 0.8 + 5 * 0.9 + 2 * 0.7) / 12
        expected_safety = (1 - 0.25) * (1 - 0.2)
        self.assertAlmostEqual(row["quality_score"], expected_quality)
        self.assertAlmostEqual(
            row[FORMULA_VERSION], expected_quality * expected_safety
        )
        self.assertEqual(row["ttc_comfort_mode"], "frame")
        self.assertEqual(row["selected_ttc_field"], "ttc_safe_frame_rate")
        self.assertEqual(row["selected_comfort_field"], "comfort_frame_rate")

    def test_existing_native_score_field_takes_precedence(self):
        row = score_scenario(
            {
                "score": 0.25,
                "native_puffer_score": 0.9,
                "completion_rate": 1,
                "ttc_within_bound_rate": 1,
                "comfortable_rate": 1,
                "collision_rate": 0,
                "offroad_rate": 0,
            }
        )

        self.assertEqual(row["native_puffer_score"], 0.9)
        self.assertEqual(row[FORMULA_VERSION], 1.0)

    def test_frame_mode_requires_frame_metrics(self):
        with self.assertRaisesRegex(ValueError, "ttc_safe_frame_rate"):
            score_scenario(
                {
                    "completion_rate": 1,
                    "ttc_within_bound_rate": 1,
                    "comfortable_rate": 1,
                    "collision_rate": 0,
                    "offroad_rate": 0,
                },
                ttc_comfort_mode="frame",
            )

    def test_summary_is_equal_weighted_across_scenarios(self):
        rows = [
            score_scenario(
                {
                    "map_id": 0,
                    "scenario_id": "a",
                    "n": 1,
                    "completion_rate": 1,
                    "ttc_within_bound_rate": 1,
                    "comfortable_rate": 1,
                    "collision_rate": 0,
                    "offroad_rate": 0,
                }
            ),
            score_scenario(
                {
                    "map_id": 1,
                    "scenario_id": "b",
                    "n": 100,
                    "completion_rate": 0,
                    "ttc_within_bound_rate": 0,
                    "comfortable_rate": 0,
                    "collision_rate": 0,
                    "offroad_rate": 0,
                }
            ),
        ]

        summary = summarize("test", "input.jsonl", rows)

        self.assertEqual(summary["scenario_count"], 2)
        self.assertAlmostEqual(summary["score"], 0.5)
        self.assertAlmostEqual(summary["mean_quality_score"], 0.5)

    def test_jsonl_loader_rejects_duplicate_scenarios(self):
        raw = {
            "map_id": 0,
            "scenario_id": "same",
            "completion_rate": 1,
            "ttc_within_bound_rate": 1,
            "comfortable_rate": 1,
            "collision_rate": 0,
            "offroad_rate": 0,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            path.write_text(
                json.dumps(raw) + "\n" + json.dumps(raw) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Duplicate scenario key"):
                load_and_score(path)


if __name__ == "__main__":
    unittest.main()
