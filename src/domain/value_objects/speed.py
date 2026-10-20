"""Objeto de Valor Speed para encapsular y normalizar velocidades del vehículo."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Speed:
    """Representa una velocidad escalar acotada con soporte de normalización."""

    value: float = 0.0
    max_forward: float = 420.0
    max_reverse: float = 160.0

    def __post_init__(self) -> None:
        if self.max_forward <= 0:
            raise ValueError("max_forward debe ser mayor que 0.")
        if self.max_reverse <= 0:
            raise ValueError("max_reverse debe ser mayor que 0.")

    def clamped(self) -> Speed:
        """Retorna una nueva instancia con el valor acotado en [-max_reverse, max_forward]."""
        clamped_val = max(-self.max_reverse, min(self.max_forward, self.value))
        return Speed(
            value=clamped_val,
            max_forward=self.max_forward,
            max_reverse=self.max_reverse,
        )

    def normalized(self) -> float:
        """Normaliza la velocidad respecto a max_forward en el rango [-1.0, 1.0]."""
        norm = self.value / self.max_forward
        return max(-1.0, min(1.0, norm))
