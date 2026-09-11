"""Non-mutating normalization of ScenarioDescription scalar state arrays."""
import numpy as np


def normalize_scalar_states(scenario):
    """Accept both [T] and [T,1] scalars without changing simulator-owned data."""
    result = dict(scenario)
    result["tracks"] = {}
    for key, track in scenario.get("tracks", {}).items():
        track_copy = dict(track)
        state = dict(track.get("state", {}))
        for field in ("heading", "length", "width", "height", "valid"):
            if field in state:
                values = np.asarray(state[field])
                if values.ndim == 2 and values.shape[1] == 1:
                    state[field] = values[:, 0]
        track_copy["state"] = state
        result["tracks"][key] = track_copy
    return result
