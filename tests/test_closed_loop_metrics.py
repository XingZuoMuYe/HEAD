import json
from types import SimpleNamespace

from head.manager.closed_loop_metrics import ClosedLoopMetricsRecorder


def _config(tmp_path, **evaluation):
    return SimpleNamespace(
        args=SimpleNamespace(
            task="real_scenario-v0",
            workflow=SimpleNamespace(policy="imitation"),
            scenario=SimpleNamespace(map="real"),
            evaluation=dict(evaluation),
            artifacts=SimpleNamespace(
                root=str(tmp_path),
                evaluation="eval",
            ),
        )
    )


def test_closed_loop_metrics_writes_episode_and_summary(tmp_path):
    recorder = ClosedLoopMetricsRecorder(_config(tmp_path, updated_metrics=False))
    recorder.start_episode()
    recorder.step({"ok": True}, [0.0], 0, object())
    recorder.finish_episode(episode_index=1, reward=3.0, length=1, env=object())

    payload = json.loads(recorder.save().read_text(encoding="utf-8"))

    assert payload["summary"]["mean_reward"] == 3.0
    assert payload["summary"]["success_rate"] == 1.0
    assert payload["episodes"][0]["success"] is True
    assert payload["episodes"][0]["collision"] is False


def test_collision_is_a_hard_failure(tmp_path):
    recorder = ClosedLoopMetricsRecorder(_config(tmp_path, updated_metrics=False))
    recorder.start_episode()
    recorder.step({"ok": True, "crash_vehicle": True}, [0.0], 0, object())
    recorder.finish_episode(episode_index=1, reward=0.0, length=1, env=object())

    payload = json.loads(recorder.save().read_text(encoding="utf-8"))
    assert payload["episodes"][0]["collision"] is True
    assert payload["episodes"][0]["success"] is False
    assert payload["summary"]["success_rate"] == 0.0


def test_out_of_road_is_a_hard_failure(tmp_path):
    recorder = ClosedLoopMetricsRecorder(_config(tmp_path, updated_metrics=False))
    recorder.start_episode()
    recorder.step({"out_of_road": True}, [0.0], 0, object())
    recorder.finish_episode(episode_index=1, reward=0.0, length=1, env=object())

    payload = json.loads(recorder.save().read_text(encoding="utf-8"))
    assert payload["episodes"][0]["out_of_road"] is True
    assert payload["episodes"][0]["success"] is False


def test_evaluation_v2_is_the_default_metric_source(tmp_path):
    """The removed UniTraj recorder left evaluation as the only metric source."""
    recorder = ClosedLoopMetricsRecorder(_config(tmp_path))
    assert recorder.v2_enabled is True
    assert recorder.available is True


def test_metrics_schema_keeps_legacy_keys_null(tmp_path):
    """Readers of metrics.json still find the keys, always empty."""
    recorder = ClosedLoopMetricsRecorder(_config(tmp_path, updated_metrics=False))
    recorder.start_episode()
    recorder.finish_episode(episode_index=1, reward=0.0, length=1, env=object())

    payload = json.loads(recorder.save().read_text(encoding="utf-8"))
    assert payload["legacy_source"] is None
    assert payload["error"] is None
    assert "scores" not in payload["episodes"][0]
    assert "scene_score" not in payload["episodes"][0]
