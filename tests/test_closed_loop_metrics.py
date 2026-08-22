import json
from types import SimpleNamespace

from head.manager.closed_loop_metrics import ClosedLoopMetricsRecorder


class FakeEvaluator:
    def __init__(self):
        self.round_scores = {
            "no_collision": [],
            "progress": [],
        }
        self.total_scores = []

    def step(self, info, observation, step_index, env):
        assert info["ok"] is True
        assert step_index >= 0

    def reset(self, total_steps, env):
        self.round_scores["no_collision"].append(1.0)
        self.round_scores["progress"].append(0.5)
        self.total_scores.append(0.75)
        return 0.75, {"no_collision": 1.0, "progress": 0.5}, 1


def _config(tmp_path):
    return SimpleNamespace(
        args=SimpleNamespace(
            task="real_scenario-v0",
            workflow=SimpleNamespace(policy="imitation"),
            scenario=SimpleNamespace(map="real"),
            artifacts=SimpleNamespace(
                root=str(tmp_path),
                evaluation="eval",
            ),
        )
    )


def test_closed_loop_metrics_writes_episode_and_summary(tmp_path):
    recorder = ClosedLoopMetricsRecorder(_config(tmp_path), evaluator_cls=FakeEvaluator)
    recorder.start_episode()
    recorder.step({"ok": True}, [0.0], 0, object())
    recorder.finish_episode(episode_index=1, reward=3.0, length=1, env=object())

    path = recorder.save()
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["available"] is True
    assert payload["summary"]["success_rate"] == 1.0
    assert payload["summary"]["total_score"] == 0.75
    assert payload["episodes"][0]["scene_score"] == 0.75


def test_closed_loop_metrics_records_unavailable_dependency(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "head.manager.closed_loop_metrics.ensure_unitraj_path",
        lambda source: (_ for _ in ()).throw(ModuleNotFoundError("missing")),
    )
    recorder = ClosedLoopMetricsRecorder(_config(tmp_path), evaluator_cls=None)
    assert recorder.available is False
    recorder.finish_episode(episode_index=1, reward=0.0, length=0, env=None)
    path = recorder.save()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["available"] is False
    assert "missing" in payload["error"]
