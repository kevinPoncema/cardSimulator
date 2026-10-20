"""Pruebas unitarias para las transformaciones LIDAR y normalización del vector de observación S."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.application.telemetry_collector import TelemetryCollector
from src.domain.entities.car import Car, CarConfig
from src.domain.entities.sensor_reading import (
    DEFAULT_LIDAR_MAX_RANGE,
    LIDAR_RAY_SPECS,
    SensorRay,
    SensorReading,
    local_ray_to_global_segment,
)
from src.domain.entities.track import Track, TrackSegment
from src.domain.value_objects.angle import Angle
from src.domain.value_objects.vector2d import Vector2D
from src.infrastructure.physics.pymunk_physics_engine import PymunkPhysicsEngine
from src.infrastructure.sensors.lidar_sensor_2d import LidarSensor2D


@pytest.mark.parametrize(
    ("car_deg", "rel_deg", "expected_end_x", "expected_end_y"),
    [
        # Coche apuntando a 0° (+X), origen en (100, 200), R_max = 300
        (0.0, 0.0, 400.0, 200.0),
        (0.0, 90.0, 100.0, 500.0),
        (0.0, -90.0, 100.0, -100.0),
        (0.0, 45.0, 100.0 + 300.0 * math.cos(math.pi / 4), 200.0 + 300.0 * math.sin(math.pi / 4)),
        (0.0, -45.0, 100.0 + 300.0 * math.cos(-math.pi / 4), 200.0 + 300.0 * math.sin(-math.pi / 4)),
        # Coche rotado a 90° (+Y): el rayo frontal (0°) apunta hacia +Y, el de -90° apunta hacia +X
        (90.0, 0.0, 100.0, 500.0),
        (90.0, -90.0, 400.0, 200.0),
        (90.0, 90.0, -200.0, 200.0),
        # Coche rotado a 180° (-X): el rayo frontal (0°) apunta hacia -X
        (180.0, 0.0, -200.0, 200.0),
    ],
)
def test_local_ray_to_global_segment_transformations(
    car_deg: float,
    rel_deg: float,
    expected_end_x: float,
    expected_end_y: float,
) -> None:
    """Verifica la transformación geométrica local-a-global de los rayos del sensor LIDAR."""
    origin = Vector2D(100.0, 200.0)
    car_angle = Angle.from_degrees(car_deg)
    max_range = 300.0

    start_pt, end_pt, global_angle = local_ray_to_global_segment(
        origin=origin,
        car_angle=car_angle,
        relative_angle_deg=rel_deg,
        max_range=max_range,
    )

    assert start_pt == origin
    assert math.isclose(end_pt.x, expected_end_x, abs_tol=1e-5)
    assert math.isclose(end_pt.y, expected_end_y, abs_tol=1e-5)
    assert math.isclose(start_pt.distance_to(end_pt), max_range, abs_tol=1e-5)
    assert math.isclose(global_angle.degrees, car_deg + rel_deg, abs_tol=1e-5)


def test_observation_vector_normalization_formula() -> None:
    """Verifica la fórmula de normalización del vector de observación S de 7 componentes."""
    max_range = 300.0
    cfg = CarConfig(max_forward_speed=400.0, max_angular_velocity=5.0)
    car = Car(
        car_id="rl_agent_1",
        position=Vector2D(0.0, 0.0),
        velocity=Vector2D(300.0, 0.0),  # v = 300 -> v / v_max = 0.75
        angle=Angle.from_degrees(0.0),
        angular_velocity=-2.5,          # omega = -2.5 -> omega / omega_max = -0.5
        config=cfg,
    )

    raw_distances = [300.0, 150.0, 75.0, 60.0, 0.0]
    rays = tuple(
        SensorRay(
            ray_id=spec[0],
            relative_angle_deg=spec[1],
            origin=car.front_center_position(),
            end_point=Vector2D(dist, 0.0),
            distance=dist,
            max_range=max_range,
            hit=(dist < max_range),
        )
        for spec, dist in zip(LIDAR_RAY_SPECS, raw_distances, strict=True)
    )
    reading = SensorReading(car_id=car.car_id, rays=rays, max_range=max_range)

    collector = TelemetryCollector()
    obs = collector.collect(car, reading)
    obs_from_getter = collector.get_observation()

    assert isinstance(obs, np.ndarray)
    assert obs.shape == (7,)
    assert obs.dtype == np.float32
    assert np.allclose(obs, obs_from_getter)

    expected = np.array(
        [
            1.0,    # d_0 / 300 = 300 / 300
            0.5,    # d_45 / 300 = 150 / 300
            0.25,   # d_-45 / 300 = 75 / 300
            0.20,   # d_90 / 300 = 60 / 300
            0.0,    # d_-90 / 300 = 0 / 300
            0.75,   # v / v_max = 300 / 400
            -0.50,  # omega / omega_max = -2.5 / 5.0
        ],
        dtype=np.float32,
    )
    assert np.allclose(obs, expected, atol=1e-5)


def test_observation_vector_clamping_at_extreme_values() -> None:
    """Verifica que el vector de observación S acote valores fuera de rango a [0, 1] y [-1, 1]."""
    max_range = 300.0
    cfg = CarConfig(max_forward_speed=200.0, max_angular_velocity=4.0)
    car = Car(
        car_id="extreme_car",
        position=Vector2D(0.0, 0.0),
        velocity=Vector2D(-500.0, 0.0),  # Excede reversa
        angle=Angle(0.0),
        angular_velocity=12.0,           # Excede omega_max
        config=cfg,
    )

    rays = tuple(
        SensorRay(
            ray_id=spec[0],
            relative_angle_deg=spec[1],
            origin=Vector2D(0.0, 0.0),
            end_point=Vector2D(450.0, 0.0),
            distance=450.0,  # Excede max_range
            max_range=max_range,
            hit=False,
        )
        for spec in LIDAR_RAY_SPECS
    )
    reading = SensorReading(car_id=car.car_id, rays=rays, max_range=max_range)
    obs = TelemetryCollector.build_observation_vector(car, reading)

    # Los primeros 5 elementos deben estar en [0.0, 1.0]
    assert np.all((obs[:5] >= 0.0) & (obs[:5] <= 1.0))
    assert np.allclose(obs[:5], 1.0)
    # Velocidad lineal y angular acotadas en [-1.0, 1.0]
    assert math.isclose(float(obs[5]), -1.0, abs_tol=1e-6)
    assert math.isclose(float(obs[6]), 1.0, abs_tol=1e-6)


def test_pymunk_lidar_sensor_detects_wall_distance_and_ignores_self() -> None:
    """Verifica que LidarSensor2D detecte una pared frontal a distancia exacta usando segment_query."""
    engine = PymunkPhysicsEngine()
    # Coche en (100, 100) con longitud 44 -> su frente está en x = 100 + 22 = 122, y = 100
    car = Car(
        car_id="sensor_car",
        position=Vector2D(100.0, 100.0),
        angle=Angle.from_degrees(0.0),
        config=CarConfig(length=44.0, width=24.0),
    )
    engine.register_car(car)

    # Colocar un muro vertical en x = 272 (exactamente a 150 px del frente del coche en x=122)
    wall = TrackSegment(start=Vector2D(272.0, 0.0), end=Vector2D(272.0, 300.0))
    track = Track(
        name="Test Wall Track",
        width=800.0,
        height=600.0,
        centerline=[Vector2D(0.0, 0.0), Vector2D(10.0, 0.0), Vector2D(5.0, 10.0)],
        outer_walls=[wall],
        inner_walls=[wall],
    )
    engine.register_track(track)

    sensor = LidarSensor2D(physics_engine=engine, max_range=DEFAULT_LIDAR_MAX_RANGE)
    reading = sensor.scan(car)

    d0_ray = reading.get_ray_by_id("d_0")
    assert d0_ray.hit is True
    # El segmento en Pymunk tiene grosor (radius=3.0), por lo que impacta en x ≈ 269 (dist ≈ 147 px)
    assert math.isclose(d0_ray.distance, 147.0, abs_tol=3.5)
    assert math.isclose(d0_ray.normalized_distance, d0_ray.distance / 300.0, abs_tol=1e-5)

    # Los rayos laterales (+90° y -90° en x=122) son paralelos al muro en x=272 y no deben impactar
    d90_left = reading.get_ray_by_id("d_90_left")
    d90_right = reading.get_ray_by_id("d_90_right")
    assert d90_left.hit is False
    assert math.isclose(d90_left.distance, 300.0)
    assert d90_right.hit is False
    assert math.isclose(d90_right.distance, 300.0)
