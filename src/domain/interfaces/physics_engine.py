"""Interfaz IPhysicsEngine para desacoplar el motor de físicas 2D del dominio."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.domain.entities.car import Car
    from src.domain.entities.track import Track
    from src.domain.value_objects.angle import Angle
    from src.domain.value_objects.vector2d import Vector2D


class IPhysicsEngine(ABC):
    """Contrato de inversión de dependencias para el motor de simulación física."""

    @abstractmethod
    def register_track(self, track: Track) -> None:
        """Registra las barreras estáticas y zonas de fricción de la pista."""

    @abstractmethod
    def register_car(self, car: Car) -> None:
        """Registra un vehículo dinámico en el espacio físico."""

    @abstractmethod
    def step(self, dt: float) -> None:
        """Avanza la simulación física un intervalo de tiempo dt (en segundos) y sincroniza entidades."""

    def reset_car_pose(self, car_id: str, position: Vector2D, angle: Angle) -> None:
        """Reposiciona un vehículo en la pista con velocidad cero."""
