"""Compatibility export for the shared pi05 metadata reader."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bindings/python"))
from vla_cpp.pi05_gguf import GGUFReader
