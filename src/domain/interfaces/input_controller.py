"""Interfaz IInputController para controladores humanos o de inteligencia artificial."""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.domain.value_objects.car_action import CarAction


class IInputController(ABC):
    """Contrato para obtener acciones de conducción sin efectos colaterales (OCP / LSP)."""

    @abstractmethod
    def get_action(self) -> CarAction:
        """Retorna la acción de conducción para el frame actual sin efectos colaterales."""

    def poll_debug_toggle(self) -> bool:
        """Retorna True si el usuario solicitó alternar el modo debug (F3)."""
        return False

    def poll_map_selection(self) -> str | None:
        """Retorna el identificador de mapa seleccionado ('next' o índice '1'..'4'), o None."""
        return None

    def poll_reset_car(self) -> bool:
        """Retorna True si el usuario solicitó reposicionar el coche en la salida (R)."""
        return False

    def poll_quit(self) -> bool:
        """Retorna True si el usuario solicitó cerrar la simulación."""
        return False
