"""Entidad de dominio Coin para monedas coleccionables generadas sobre la pista."""

from __future__ import annotations

from dataclasses import dataclass

from src.domain.value_objects.vector2d import Vector2D


@dataclass
class Coin:
    """Moneda coleccionable ubicada en una zona alcanzable del circuito.

    Atributos:
        coin_id: Identificador único de la moneda.
        position: Coordenadas globales (x, y) dentro de la calzada transitable.
        radius: Radio de recolección visual y física (px).
        value: Puntos que otorga al ser recolectada por un vehículo.
        active: Indica si la moneda está disponible en el circuito.
    """

    coin_id: str
    position: Vector2D
    radius: float = 14.0
    value: int = 1
    active: bool = True
