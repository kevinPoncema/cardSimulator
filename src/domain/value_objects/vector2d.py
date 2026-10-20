"""Objeto de Valor Vector2D para cálculos geométricos y cinemáticos 2D."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Vector2D:
    """Representa un vector o punto inmutable en el espacio bidimensional."""

    x: float = 0.0
    y: float = 0.0

    def __add__(self, other: Vector2D) -> Vector2D:
        return Vector2D(self.x + other.x, self.y + other.y)

    def __sub__(self, other: Vector2D) -> Vector2D:
        return Vector2D(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: float) -> Vector2D:
        return Vector2D(self.x * float(scalar), self.y * float(scalar))

    def __rmul__(self, scalar: float) -> Vector2D:
        return self.__mul__(scalar)

    def __truediv__(self, scalar: float) -> Vector2D:
        if abs(scalar) < 1e-12:
            raise ZeroDivisionError("No se puede dividir un Vector2D por cero.")
        return Vector2D(self.x / float(scalar), self.y / float(scalar))

    def magnitude(self) -> float:
        """Retorna la norma euclidiana del vector."""
        return math.hypot(self.x, self.y)

    def distance_to(self, other: Vector2D) -> float:
        """Calcula la distancia euclidiana hacia otro punto Vector2D."""
        return math.hypot(other.x - self.x, other.y - self.y)

    def normalized(self) -> Vector2D:
        """Retorna el vector unitario en la misma dirección, o (0, 0) si es nulo."""
        mag = self.magnitude()
        if mag < 1e-12:
            return Vector2D(0.0, 0.0)
        return Vector2D(self.x / mag, self.y / mag)

    def dot(self, other: Vector2D) -> float:
        """Producto escalar (dot product) entre dos vectores 2D."""
        return self.x * other.x + self.y * other.y

    def rotate(self, radians: float) -> Vector2D:
        """Rota el vector un ángulo dado en radianes."""
        cos_a = math.cos(radians)
        sin_a = math.sin(radians)
        return Vector2D(
            self.x * cos_a - self.y * sin_a,
            self.x * sin_a + self.y * cos_a,
        )

    def to_tuple(self) -> tuple[float, float]:
        """Convierte el vector a una tupla (x, y) de flotantes."""
        return (float(self.x), float(self.y))

    def to_int_tuple(self) -> tuple[int, int]:
        """Convierte el vector a una tupla (int(x), int(y)) para píxeles."""
        return (int(round(self.x)), int(round(self.y)))
