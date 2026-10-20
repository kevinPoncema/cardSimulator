"""Exportación de interfaces del dominio (DIP e ISP)."""

from src.domain.interfaces.controllable import IControllable
from src.domain.interfaces.input_controller import IInputController
from src.domain.interfaces.physics_engine import IPhysicsEngine
from src.domain.interfaces.renderer import IRenderer
from src.domain.interfaces.sensor import ISensor
from src.domain.interfaces.telemetry_provider import ITelemetryProvider

__all__ = [
    "IControllable",
    "IInputController",
    "IPhysicsEngine",
    "IRenderer",
    "ISensor",
    "ITelemetryProvider",
]
