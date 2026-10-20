"""Exportación de entidades de la capa de dominio."""

from src.domain.entities.car import Car, CarConfig
from src.domain.entities.sensor_reading import (
    DEFAULT_LIDAR_MAX_RANGE,
    LIDAR_RAY_SPECS,
    SensorRay,
    SensorReading,
    local_ray_to_global_segment,
)
from src.domain.entities.track import (
    PRESET_MAPS,
    SURFACE_CATALOG,
    SurfaceProperties,
    Track,
    TrackGate,
    TrackSegment,
    segments_intersect,
)

__all__ = [
    "Car",
    "CarConfig",
    "DEFAULT_LIDAR_MAX_RANGE",
    "LIDAR_RAY_SPECS",
    "PRESET_MAPS",
    "SURFACE_CATALOG",
    "SensorRay",
    "SensorReading",
    "SurfaceProperties",
    "Track",
    "TrackGate",
    "TrackSegment",
    "local_ray_to_global_segment",
    "segments_intersect",
]
