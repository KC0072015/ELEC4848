"""
src/chat_history.py: Lightweight conversation history buffer.
"""

from typing import List, Tuple

class ChatHistory:
    """Stores recent user/assistant turns and formats them for prompts."""

    def __init__(self, max_turns: int = 6):
        self.max_turns = max_turns
        self.turns: List[Tuple[str, str]] = []

    def add_turn(self, user_message: str, assistant_message: str) -> None:
        self.turns.append((user_message, assistant_message))
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns:]

    def format_for_prompt(self) -> str:
        if not self.turns:
            return "None"
        return "\n\n".join(
            [f"User: {u}\nAssistant: {a}" for u, a in self.turns[-self.max_turns:]]
        )
