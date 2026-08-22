"""Central path resolution for weights and generated artifacts."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_project_path(value) -> Path:
    """Resolve an absolute path or a path relative to the repository root."""
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def artifacts_root(args) -> Path:
    return resolve_project_path(args.artifacts.root)


def artifact_path(args, value) -> Path:
    """Resolve a configured artifact subpath relative to artifacts.root."""
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (artifacts_root(args) / path).resolve()


def evolution_paths(args) -> dict:
    """Return deterministic directories for one evolution strategy and task."""
    strategy = str(args.workflow.evolution.strategy)
    learner = str(args.workflow.evolution.learner)
    task = str(args.task).split("-", 1)[0]
    map_name = str(args.scenario.map)
    map_aliases = {
        "X": "interaction",
        "O": "roundabout",
        "C": "circle_road",
        "r": "inRamp",
        "SSSSSSSSSSSSSS": "straight_road",
    }
    map_name = map_aliases.get(map_name, map_name)
    if map_name == "straight_road" and not getattr(args.scenario, "pedestrians", False):
        map_name = "straight_road_no_pedestrian"

    relative_run = Path(strategy) / learner / task / map_name
    return {
        "weights": artifact_path(args, args.artifacts.weights.evolution) / relative_run,
        "logs": artifact_path(args, args.artifacts.logs) / relative_run,
        "evaluation": artifact_path(args, args.artifacts.evaluation) / relative_run,
        "task": task,
        "map": map_name,
    }


def poly_checkpoint_dir(args) -> Path:
    """Resolve Poly's configured or automatic SAC run directory."""
    configured = args.workflow.policies.Poly.get("checkpoint", None)
    if configured in (None, "", "auto"):
        return evolution_paths(args)["weights"] / str(args.train_name)
    return resolve_project_path(configured)


def poly_checkpoint_roots(args):
    """Return configured/automatic roots, including the pre-refactor layout."""
    configured = args.workflow.policies.Poly.get("checkpoint", None)
    if configured not in (None, "", "auto"):
        return [resolve_project_path(configured)]

    paths = evolution_paths(args)
    train_name = str(args.train_name)
    roots = [paths["weights"] / train_name]
    # The legacy tree is only a compatibility fallback for the default
    # artifacts root. A custom root must be isolated from repository artifacts.
    if str(args.artifacts.root) == "artifacts":
        roots.append(
            PROJECT_ROOT / "artifacts" / "models" / "RLBoost_SAC" / "checkpoints"
            / paths["task"] / paths["map"] / train_name
        )
    return roots


def resolve_poly_checkpoint(args):
    """Find the newest complete SAC checkpoint directory under known roots."""
    checkpoints = []
    for root in poly_checkpoint_roots(args):
        if (root / "sac_policy").is_file():
            checkpoints.append(root)
        if root.is_dir():
            checkpoints.extend(
                path.parent for path in root.rglob("sac_policy") if path.is_file()
            )
    if not checkpoints:
        return None
    return max(set(checkpoints), key=lambda path: path.stat().st_mtime)


def has_poly_checkpoint(args) -> bool:
    return resolve_poly_checkpoint(args) is not None


def imitation_weights_dir(args) -> Path:
    model = str(args.workflow.policies.imitation.model)
    return artifact_path(args, args.artifacts.weights.imitation) / model


def imitation_checkpoint_candidates(args):
    """Return new and legacy locations for an imitation checkpoint."""
    imitation = args.workflow.policies.imitation
    configured_value = imitation.get("checkpoint", None)
    if not configured_value:
        return []
    configured = Path(str(configured_value)).expanduser()
    if configured.is_absolute():
        return [configured]

    source = imitation.get("source", None)
    candidates = [
        PROJECT_ROOT / configured,
        imitation_weights_dir(args) / configured,
        artifact_path(args, args.artifacts.weights.imitation) / configured,
    ]
    if source:
        source_path = Path(str(source)).expanduser()
        source_root = source_path if source_path.is_absolute() else resolve_project_path(source_path)
        candidates.append(source_root / configured)
        candidates.append(source_root / "checkpoints" / configured)
    candidates.extend([
        PROJECT_ROOT / "head" / "policy" / "imitation_policy" / "checkpoints" / configured.name,
    ])
    return candidates


def resolve_imitation_checkpoint(args) -> Path:
    candidates = imitation_checkpoint_candidates(args)
    checkpoint = next((path.resolve() for path in candidates if path.is_file()), None)
    if checkpoint is None:
        checked = ", ".join(str(path) for path in candidates)
        raise FileNotFoundError(f"Imitation checkpoint not found. Checked: {checked}")
    return checkpoint
