"""Interfaz segregada ITelemetryProvider para consulta de observación y sensores."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.domain.entities.sensor_reading import SensorReading


class ITelemetryProvider(ABC):
    """Contrato de solo lectura para consultar el vector de observación y telemetría LIDAR."""

    @abstractmethod
    def get_observation(self, car_id: str | None = None) -> np.ndarray:
        """Retorna el vector de estado normalizado S de 7 dimensiones como np.ndarray."""

    @abstractmethod
    def get_latest_reading(self, car_id: str | None = None) -> SensorReading | None:
        """Retorna la última lectura sensorial LIDAR registrada para el vehículo."""
