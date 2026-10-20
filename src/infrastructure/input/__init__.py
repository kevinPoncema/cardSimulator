"""Controladores de entrada para jugadores humanos y agentes autónomos (OCP/LSP)."""

from src.infrastructure.input.dummy_ai_controller import DummyAiController
from src.infrastructure.input.human_keyboard_controller import HumanKeyboardController

__all__ = ["DummyAiController", "HumanKeyboardController"]
