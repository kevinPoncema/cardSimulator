"""Objeto de Valor CarAction que encapsula la intención de conducción por frame."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CarAction:
    """Comando inmutable de conducción retornado por cualquier IInputController.

    Atributos:
        throttle: Aceleración en [0.0, 1.0].
        brake: Frenado / reversa en [0.0, 1.0].
        steer: Dirección en [-1.0, 1.0] (-1.0 = izquierda, +1.0 = derecha).
        handbrake: Freno de mano activo (True aumenta el derrape lateral).
    """

    throttle: float = 0.0
    brake: float = 0.0
    steer: float = 0.0
    handbrake: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "throttle", max(0.0, min(1.0, float(self.throttle))))
        object.__setattr__(self, "brake", max(0.0, min(1.0, float(self.brake))))
        object.__setattr__(self, "steer", max(-1.0, min(1.0, float(self.steer))))
        object.__setattr__(self, "handbrake", bool(self.handbrake))

    @classmethod
    def idle(cls) -> CarAction:
        """Retorna una acción neutra sin entrada de control."""
        return cls(throttle=0.0, brake=0.0, steer=0.0, handbrake=False)
