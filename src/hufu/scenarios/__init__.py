"""Ready-made data-quality scenarios built on the HuFu rule layer."""

from hufu.scenarios.memory import (
    MemoryConflict,
    default_ruleset,
    detect_memory_conflicts,
)

__all__ = [
    "MemoryConflict",
    "default_ruleset",
    "detect_memory_conflicts",
]
