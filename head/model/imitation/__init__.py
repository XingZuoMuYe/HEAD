"""Pluggable algorithms; importing this package loads no network."""
from .base import ModelInput, BaseAdapter, Trajectory
from .loader import create_adapter, get_adapter_class
__all__ = ["ModelInput", "BaseAdapter", "Trajectory", "create_adapter", "get_adapter_class"]
