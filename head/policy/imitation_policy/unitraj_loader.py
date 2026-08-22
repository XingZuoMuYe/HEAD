"""Helpers for loading an optional external UniTraj checkout."""

from pathlib import Path
import sys
import types


def _install_namespace(root, name):
    """Avoid eager optional imports in UniTraj's package ``__init__`` files."""
    module_name = f"unitraj.{name}"
    module = sys.modules.get(module_name)
    if module is None:
        module = types.ModuleType(module_name)
        module.__path__ = [str(root / "unitraj" / name)]
        module.__package__ = module_name
        sys.modules[module_name] = module


def ensure_unitraj_path(source=None):
    """Add a UniTraj repository to ``sys.path`` and return its root.

    The benchmark is intentionally optional: rule-based and evolutionary modes
    must remain importable without its heavyweight dependencies.
    """
    if source:
        root = Path(source).expanduser()
        if not root.is_absolute():
            root = Path(__file__).resolve().parents[3] / root
        root = root.resolve()
        if not (root / "unitraj").is_dir():
            raise FileNotFoundError(
                f"UniTraj source must contain a 'unitraj' package: {root}"
            )
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        _install_namespace(root, "models")
        _install_namespace(root, "datasets")
        return root

    package_root = Path(__file__).resolve().parents[3]
    candidates = [Path(entry) if entry else Path.cwd() for entry in sys.path]
    candidates.extend([
        package_root.parent / "UniTraj_benchmark_sample",
        Path.cwd().parent / "UniTraj_benchmark_sample",
    ])
    for root in candidates:
        root = root.expanduser().resolve()
        if (root / "unitraj").is_dir():
            if str(root) not in sys.path:
                sys.path.insert(0, str(root))
            _install_namespace(root, "models")
            _install_namespace(root, "datasets")
            return root
    raise ModuleNotFoundError(
        "UniTraj is required for imitation closed-loop mode. Set "
        "workflow.policies.imitation.source to the UniTraj repository root."
    )
