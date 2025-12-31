"""Utilities for managing and modifying agent memory state."""


def clear_memory(memory) -> None:
    """Helper to reset memory if needed."""
    if hasattr(memory, "clear"):
        memory.clear()
