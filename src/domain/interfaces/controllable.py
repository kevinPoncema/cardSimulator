"""Interfaz segregada IControllable para entidades que reciben comandos de conducción."""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.domain.value_objects.car_action import CarAction


class IControllable(ABC):
    """Contrato para manipular la intención de conducción de un vehículo (ISP)."""

    @abstractmethod
    def apply_action(self, action: CarAction) -> None:
        """Aplica un comando de conducción (acelerador, freno, giro, freno de mano)."""
