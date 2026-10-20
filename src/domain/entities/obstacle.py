"""Entidades de dominio para obstáculos físicos y zonas de peligro en la pista."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from src.domain.value_objects.vector2d import Vector2D


class ObstacleType(str, Enum):
    """Tipos de obstáculos interactivos en el circuito."""

    CONE = "cone"             # Cono de tráfico naranja físico (se golpea y desplaza)
    BARRIER = "barrier"       # Barrera de hormigón rígida / neumáticos
    POTHOLE = "pothole"       # Bache gigante / zona rota (reduce drásticamente la velocidad)
    OIL_SLICK = "oil_slick"   # Mancha de aceite deslizante (pérdida total de agarre lateral)


@dataclass(frozen=True)
class Obstacle:
    """Obstáculo colocado en el mundo de carreras.

    Atributos:
        obstacle_id: Identificador único.
        kind: Tipo de obstáculo (ObstacleType).
        position: Coordenadas mundiales iniciales.
        radius: Radio para detección y colisión circular (conos, baches, aceite).
        width: Anchura (para barreras rectangulares).
        height: Altura (para barreras rectangulares).
        angle_deg: Orientación en grados para barreras rectangulares.
    """

    obstacle_id: str
    kind: ObstacleType
    position: Vector2D
    radius: float = 14.0
    width: float = 36.0
    height: float = 14.0
    angle_deg: float = 0.0

    @property
    def is_physical(self) -> bool:
        """Indica si el obstáculo tiene cuerpo sólido detectable por LIDAR y colisión."""
        return self.kind in (ObstacleType.CONE, ObstacleType.BARRIER)

    @property
    def is_hazard_surface(self) -> bool:
        """Indica si el obstáculo modifica las propiedades de tracción/agarre del suelo."""
        return self.kind in (ObstacleType.POTHOLE, ObstacleType.OIL_SLICK)
