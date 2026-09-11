"""Opt-in real-weight regressions; weights are never part of the repository.

Set HEAD_PLUTO_CHECKPOINT / HEAD_WAYFORMER_CHECKPOINT to trusted local files.
Uses in-memory copies of the bundled scenarios, including frames beyond 80.
"""
import copy
import os
from pathlib import Path
import pickle
import zipfile

import numpy as np
import pytest
from omegaconf import OmegaConf

from head.model import ModelInput, create_adapter


@pytest.mark.parametrize("model", ["pluto", "wayformer"])
def test_real_weights_predict_bundled_scenarios(model):
    checkpoint = os.environ.get(f"HEAD_{model.upper()}_CHECKPOINT")
    if not checkpoint:
        pytest.skip(f"Set HEAD_{model.upper()}_CHECKPOINT for real-weight regression")
    adapter = create_adapter(OmegaConf.create({"model": model}), checkpoint, device="cpu")
    root = Path(__file__).parents[1]
    for dataset in ["waymo", "geely"]:
        with zipfile.ZipFile(root / f"head/scenario_datasets/{dataset}.zip") as archive:
            member = next(n for n in archive.namelist() if "/sd_" in n and n.endswith(".pkl"))
            scenario = pickle.loads(archive.read(member))
        original_ts = np.asarray(scenario["metadata"]["ts"]).copy()
        steps = [s for s in [21, 80, 115] if s < len(original_ts)]
        for step in steps:
            result = adapter.compute_trajectory(ModelInput(scenario, step))
            assert result.samples.shape[0] >= 2
            assert np.isfinite(result.samples).all()
            np.testing.assert_equal(scenario["metadata"]["ts"], original_ts)
        adapter.reset()


@pytest.mark.parametrize("model", ["pluto", "wayformer"])
def test_declared_future_context_scope(model):
    checkpoint = os.environ.get(f"HEAD_{model.upper()}_CHECKPOINT")
    if not checkpoint:
        pytest.skip(f"Set HEAD_{model.upper()}_CHECKPOINT for input-scope audit")
    adapter = create_adapter(OmegaConf.create({"model": model}), checkpoint, device="cpu")
    root = Path(__file__).parents[1]
    with zipfile.ZipFile(root / "head/scenario_datasets/waymo.zip") as archive:
        member = next(n for n in archive.namelist() if "/sd_" in n)
        scenario = pickle.loads(archive.read(member))
    before = adapter.compute_trajectory(ModelInput(copy.deepcopy(scenario), 21)).samples
    altered = copy.deepcopy(scenario)
    for track in altered["tracks"].values():
        state = track["state"]
        state["position"][22:, :2] += 10000
        state["velocity"][22:] *= -3
        state["heading"][22:] += 1
    adapter.reset()
    after = adapter.compute_trajectory(ModelInput(altered, 21)).samples
    if adapter.input_scope == "history_only":
        np.testing.assert_array_equal(before, after)
    else:
        # Document an existing limitation, not a guarantee of causal inference.
        assert adapter.input_scope == "legacy_logged_route"
        assert not np.allclose(before, after)
