"""Interfaz IRenderer para desacoplar la presentación visual de la lógica de carrera."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.domain.entities.car import Car
    from src.domain.entities.sensor_reading import SensorReading
    from src.domain.entities.track import Track


class IRenderer(ABC):
    """Contrato para el motor de renderizado 2D y overlay de depuración."""

    @abstractmethod
    def render(
        self,
        track: Track,
        cars: list[Car],
        readings: dict[str, SensorReading],
        observations: dict[str, np.ndarray],
        debug_mode: bool = False,
        fps: float = 60.0,
        dynamic_obstacles: list | None = None,
    ) -> None:
        """Renderiza un fotograma completo de la simulación."""

    @abstractmethod
    def close(self) -> None:
        """Libera los recursos gráficos al finalizar la simulación."""
