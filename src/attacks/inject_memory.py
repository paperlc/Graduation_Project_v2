"""Memory injection attack utilities."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage


def inject_memory(agent: Any, text: str) -> None:
    """
    Directly write into the agent's memory buffer to simulate an injection attack.

    Works with SimpleChatMemory (chat_memory alias) or a custom chat_memory that supports add_user_message.
    """
    if hasattr(agent, "memory"):
        mem = agent.memory
        if hasattr(mem, "chat_memory") and hasattr(mem.chat_memory, "add_user_message"):
            mem.chat_memory.add_user_message(text)
            return
        if hasattr(mem, "add_user_message"):
            mem.add_user_message(text)
            return
        if hasattr(mem, "_messages"):  # fallback: append raw human message
            mem._messages.append(HumanMessage(content=text))
            return

    raise AttributeError("Agent does not expose injectable memory.")
