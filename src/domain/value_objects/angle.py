"""Objeto de Valor Angle para conversiones angulares y vectores de dirección."""

from __future__ import annotations

import math
from dataclasses import dataclass

from src.domain.value_objects.vector2d import Vector2D


@dataclass(frozen=True)
class Angle:
    """Representa un ángulo en radianes con utilidades de transformación 2D."""

    radians: float = 0.0

    @classmethod
    def from_degrees(cls, degrees: float) -> Angle:
        """Crea un Angle a partir de grados sexagesimales."""
        return cls(radians=math.radians(degrees))

    @property
    def degrees(self) -> float:
        """Retorna el ángulo en grados sexagesimales."""
        return math.degrees(self.radians)

    def normalized(self) -> Angle:
        """Normaliza el ángulo al intervalo (-pi, pi]."""
        norm = (self.radians + math.pi) % (2.0 * math.pi) - math.pi
        return Angle(radians=norm)

    def to_unit_vector(self) -> Vector2D:
        """Retorna el vector unitario (cos(theta), sin(theta))."""
        return Vector2D(math.cos(self.radians), math.sin(self.radians))

    def to_right_vector(self) -> Vector2D:
        """Retorna el vector unitario perpendicular derecho (-sin(theta), cos(theta))."""
        return Vector2D(-math.sin(self.radians), math.cos(self.radians))

    def add_degrees(self, delta_degrees: float) -> Angle:
        """Suma un desplazamiento en grados al ángulo actual."""
        return Angle(radians=self.radians + math.radians(delta_degrees))
