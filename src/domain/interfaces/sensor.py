"""Interfaz ISensor para sistemas de percepción espacial (como LIDAR 2D)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.domain.entities.car import Car
    from src.domain.entities.sensor_reading import SensorReading


class ISensor(ABC):
    """Contrato para sensores que emiten rayos o consultan el entorno alrededor de un vehículo."""

    @abstractmethod
    def scan(self, car: Car) -> SensorReading:
        """Ejecuta el barrido sensorial desde el chasis del vehículo y retorna la lectura."""
