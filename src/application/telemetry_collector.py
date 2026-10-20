"""Colector y normalizador de telemetría LIDAR y estado cinemático para agentes IA/RL."""

from __future__ import annotations

import numpy as np

from src.domain.entities.car import Car
from src.domain.entities.sensor_reading import SensorReading
from src.domain.interfaces.telemetry_provider import ITelemetryProvider


class TelemetryCollector(ITelemetryProvider):
    """Construye y almacena el vector de estado normalizado S de 7 dimensiones.

    Vector de observación:
        S = [d_0/R_max, d_45/R_max, d_-45/R_max, d_90/R_max, d_-90/R_max, v/v_max, omega/omega_max]
    """

    def __init__(self) -> None:
        self._observations: dict[str, np.ndarray] = {}
        self._readings: dict[str, SensorReading] = {}
        self._wrong_ways: dict[str, bool] = {}
        self._heading_alignments: dict[str, float] = {}
        self._turn_dirs: dict[str, float] = {}
        self._default_car_id: str | None = None

    @staticmethod
    def build_observation_vector(car: Car, reading: SensorReading) -> np.ndarray:
        """Calcula el vector de observación S normalizado a partir de Car y SensorReading."""
        normalized_distances = reading.get_normalized_distances()
        if len(normalized_distances) != 5:
            raise ValueError(
                f"Se esperaban exactamente 5 rayos LIDAR, se recibieron {len(normalized_distances)}."
            )

        # Asegurar acotamiento estricto [0.0, 1.0] para distancias LIDAR
        clamped_rays = [float(np.clip(d, 0.0, 1.0)) for d in normalized_distances]

        # Velocidad lineal longitudinal normalizada en [-1.0, 1.0]
        norm_v = float(np.clip(car.normalized_forward_speed(), -1.0, 1.0))

        # Velocidad angular normalizada en [-1.0, 1.0]
        norm_omega = float(np.clip(car.normalized_angular_velocity(), -1.0, 1.0))

        state_vector = np.array(
            [
                clamped_rays[0],  # d_0 / R_max
                clamped_rays[1],  # d_45 / R_max
                clamped_rays[2],  # d_-45 / R_max
                clamped_rays[3],  # d_90 / R_max
                clamped_rays[4],  # d_-90 / R_max
                norm_v,           # v / v_max
                norm_omega,       # omega / omega_max
            ],
            dtype=np.float32,
        )
        return state_vector

    def collect(self, car: Car, reading: SensorReading) -> np.ndarray:
        """Registra la lectura actual del vehículo y retorna su vector de observación normalizado."""
        if self._default_car_id is None:
            self._default_car_id = car.car_id
        obs = self.build_observation_vector(car, reading)
        self._readings[car.car_id] = reading
        self._observations[car.car_id] = obs
        self._wrong_ways[car.car_id] = bool(car.is_wrong_way)
        self._heading_alignments[car.car_id] = float(car.track_heading_alignment)
        self._turn_dirs[car.car_id] = float(getattr(car, "turn_to_track_dir", 1.0))
        return obs

    def is_wrong_way(self, car_id: str | None = None) -> bool:
        """Indica si el vehículo está circulando o encarado en sentido contrario."""
        target_id = car_id or self._default_car_id
        if target_id is None:
            return False
        return self._wrong_ways.get(target_id, False)

    def get_heading_alignment(self, car_id: str | None = None) -> float:
        """Retorna la proyección de alineación del vehículo respecto al sentido de carrera [-1.0, 1.0]."""
        target_id = car_id or self._default_car_id
        if target_id is None:
            return 1.0
        return self._heading_alignments.get(target_id, 1.0)

    def get_turn_to_track_dir(self, car_id: str | None = None) -> float:
        """Retorna la dirección angular de giro (+1 derecha, -1 izquierda) para encarar la pista."""
        target_id = car_id or self._default_car_id
        if target_id is None:
            return 1.0
        return self._turn_dirs.get(target_id, 1.0)

    def get_observation(self, car_id: str | None = None) -> np.ndarray:
        """Retorna el vector de estado normalizado S -> np.ndarray (Sección 6.2)."""
        target_id = car_id or self._default_car_id
        if target_id is None or target_id not in self._observations:
            return np.zeros(7, dtype=np.float32)
        return self._observations[target_id].copy()

    def get_latest_reading(self, car_id: str | None = None) -> SensorReading | None:
        """Retorna la última lectura SensorReading del vehículo indicado."""
        target_id = car_id or self._default_car_id
        if target_id is None:
            return None
        return self._readings.get(target_id)

    @property
    def all_readings(self) -> dict[str, SensorReading]:
        """Retorna el diccionario de lecturas actuales por car_id."""
        return dict(self._readings)

    @property
    def all_observations(self) -> dict[str, np.ndarray]:
        """Retorna el diccionario de vectores de observación actuales por car_id."""
        return {k: v.copy() for k, v in self._observations.items()}
