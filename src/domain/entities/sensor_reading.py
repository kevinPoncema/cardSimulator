"""Entidades y cálculos geométricos de dominio para el sensor LIDAR 2D."""

from __future__ import annotations

import math
from dataclasses import dataclass

from src.domain.value_objects.angle import Angle
from src.domain.value_objects.vector2d import Vector2D

# Distribución estándar de 5 rayos según Especificación Técnica (Sección 6.1)
LIDAR_RAY_SPECS: tuple[tuple[str, float], ...] = (
    ("d_0", 0.0),          # Frontal: 0°
    ("d_45_left", 45.0),   # Diagonal Izq.: +45°
    ("d_45_right", -45.0), # Diagonal Der.: -45°
    ("d_90_left", 90.0),   # Lateral Izq.: +90°
    ("d_90_right", -90.0), # Lateral Der.: -90°
)

DEFAULT_LIDAR_MAX_RANGE: float = 300.0


def local_ray_to_global_segment(
    origin: Vector2D,
    car_angle: Angle,
    relative_angle_deg: float,
    max_range: float = DEFAULT_LIDAR_MAX_RANGE,
) -> tuple[Vector2D, Vector2D, Angle]:
    """Transforma un rayo definido en el marco local del coche a coordenadas globales.

    Args:
        origin: Punto de emisión global (centro delantero del chasis).
        car_angle: Ángulo de orientación global del vehículo.
        relative_angle_deg: Ángulo relativo del rayo en grados (0, +45, -45, +90, -90).
        max_range: Alcance máximo del rayo R_max (por defecto 300.0 px).

    Returns:
        Tupla (origin, max_end_point, global_angle).
    """
    if max_range <= 0:
        raise ValueError("max_range debe ser mayor que 0.")
    rel_rad = math.radians(relative_angle_deg)
    local_ray_vector = Vector2D(
        max_range * math.cos(rel_rad),
        max_range * math.sin(rel_rad),
    )
    global_ray_vector = local_ray_vector.rotate(car_angle.radians)
    end_point = origin + global_ray_vector
    global_angle = Angle(radians=car_angle.radians + rel_rad)
    return origin, end_point, global_angle


@dataclass(frozen=True)
class SensorRay:
    """Representa el resultado individual de un rayo de sondeo LIDAR."""

    ray_id: str
    relative_angle_deg: float
    origin: Vector2D
    end_point: Vector2D
    distance: float
    max_range: float = DEFAULT_LIDAR_MAX_RANGE
    hit: bool = False

    @property
    def normalized_distance(self) -> float:
        """Retorna d_i / R_max acotado estrictamente en [0.0, 1.0]."""
        if self.max_range <= 0:
            return 0.0
        clamped_dist = max(0.0, min(self.max_range, self.distance))
        return clamped_dist / self.max_range


@dataclass(frozen=True)
class SensorReading:
    """Lectura completa del arreglo de 5 rayos LIDAR para un vehículo en un instante."""

    car_id: str
    rays: tuple[SensorRay, ...]
    max_range: float = DEFAULT_LIDAR_MAX_RANGE

    def get_normalized_distances(self) -> list[float]:
        """Retorna la lista de distancias normalizadas [d_0, d_45, d_-45, d_90, d_-90] / R_max."""
        return [ray.normalized_distance for ray in self.rays]

    def get_ray_by_id(self, ray_id: str) -> SensorRay:
        """Busca un rayo por su identificador ('d_0', 'd_45_left', etc.)."""
        for ray in self.rays:
            if ray.ray_id == ray_id:
                return ray
        raise KeyError(f"No existe el rayo con identificador '{ray_id}'.")
