"""Pruebas unitarias para el sistema de Aprendizaje por Refuerzo, Recompensas y Red Neuronal."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from src.application.rl.car_racing_env import CarRacingRLEnv
from src.application.rl.reward_config import RewardCalculator, RewardConfig
from src.domain.entities.car import Car
from src.domain.entities.sensor_reading import LIDAR_RAY_SPECS, SensorRay, SensorReading
from src.domain.entities.track import Track
from src.domain.value_objects.angle import Angle
from src.domain.value_objects.vector2d import Vector2D
from src.infrastructure.ai.mlp_policy import MLPPolicy
from src.infrastructure.physics.pymunk_physics_engine import PymunkPhysicsEngine
from src.infrastructure.sensors.lidar_sensor_2d import LidarSensor2D


def _make_dummy_reading(car: Car, distance: float = 300.0) -> SensorReading:
    rays = tuple(
        SensorRay(
            ray_id=spec[0],
            relative_angle_deg=spec[1],
            origin=car.front_center_position(),
            end_point=Vector2D(distance, 0.0),
            distance=distance,
            max_range=300.0,
            hit=(distance < 300.0),
        )
        for spec in LIDAR_RAY_SPECS
    )
    return SensorReading(car_id=car.car_id, rays=rays, max_range=300.0)


def test_reward_config_loads_from_json() -> None:
    """Verifica que config/reward_config.json cargue todos los pesos correctamente."""
    cfg_path = Path(__file__).resolve().parent.parent / "config" / "reward_config.json"
    assert cfg_path.exists()
    cfg = RewardConfig.from_json_file(cfg_path)
    assert cfg.checkpoint_reward > 0
    assert cfg.lap_completion_reward > cfg.checkpoint_reward
    assert cfg.cone_collision_penalty < 0
    assert cfg.barrier_collision_penalty < 0
    assert cfg.reverse_finish_line_penalty < 0


def test_reward_calculator_penalizes_obstacles_and_wrong_way() -> None:
    """Verifica que chocar con obstáculos, ir en sentido contrario o cruzar meta al revés reste puntos."""
    cfg = RewardConfig()
    calc = RewardCalculator(cfg)
    track = Track.create_default_circuit("kart_classic")
    car = Car(car_id="rl_test", position=track.start_position, angle=track.start_angle)
    reading = _make_dummy_reading(car)

    # 1. Chocar con un cono y pisar un bache debe restar puntos
    car.active_collisions = {"cone", "pothole"}
    car.current_surface_name = "pothole"
    reward_obs, _, breakdown = calc.compute_step_reward(
        car=car,
        track=track,
        reading=reading,
        gate_crossings=[],
        lap_completed=False,
        checkpoint_advanced=False,
    )
    assert reward_obs < -35.0
    assert "cone_collision" in breakdown
    assert "pothole_penalty" in breakdown

    # 2. Cruzar la línea de meta (gate 0) en sentido contrario debe restar puntos y terminar episodio
    calc.reset()
    car.active_collisions.clear()
    car.current_surface_name = "asphalt"
    reward_rev, terminated_rev, breakdown_rev = calc.compute_step_reward(
        car=car,
        track=track,
        reading=reading,
        gate_crossings=[(0, False)],
        lap_completed=False,
        checkpoint_advanced=False,
    )
    assert reward_rev <= cfg.reverse_finish_line_penalty
    assert terminated_rev is True
    assert breakdown_rev["termination_reason"] == "reverse_finish_line"

    # 3. Ir en sentido contrario durante muchos pasos activa wrong_way_timeout
    calc.reset()
    car.is_wrong_way = True
    car.velocity = Vector2D(100.0, 0.0)
    terminated_ww = False
    for _ in range(cfg.max_wrong_way_steps + 2):
        r_ww, terminated_ww, bd_ww = calc.compute_step_reward(
            car=car,
            track=track,
            reading=reading,
            gate_crossings=[],
            lap_completed=False,
            checkpoint_advanced=False,
        )
        if terminated_ww:
            assert bd_ww["termination_reason"] == "wrong_way_timeout"
            assert r_ww <= cfg.wrong_way_timeout_penalty
            break
    assert terminated_ww is True


def test_reward_calculator_rewards_checkpoints_and_laps() -> None:
    """Verifica que cruzar checkpoints y completar vueltas válidas otorgue recompensas positivas altas."""
    cfg = RewardConfig()
    calc = RewardCalculator(cfg)
    track = Track.create_default_circuit("kart_classic")
    tangent = track.closest_centerline_tangent(track.start_position)
    car = Car(
        car_id="rl_good",
        position=track.start_position,
        velocity=tangent * 300.0,
        angle=Angle(radians=math.atan2(tangent.y, tangent.x)),
    )
    reading = _make_dummy_reading(car, distance=300.0)

    r_cp, term_cp, bd_cp = calc.compute_step_reward(
        car=car,
        track=track,
        reading=reading,
        gate_crossings=[(1, True)],
        lap_completed=False,
        checkpoint_advanced=True,
    )
    assert term_cp is False
    assert r_cp >= cfg.checkpoint_reward
    assert bd_cp["checkpoint"] == cfg.checkpoint_reward

    r_lap, term_lap, bd_lap = calc.compute_step_reward(
        car=car,
        track=track,
        reading=reading,
        gate_crossings=[(0, True)],
        lap_completed=True,
        checkpoint_advanced=True,
    )
    assert term_lap is False
    assert r_lap >= cfg.lap_completion_reward
    assert bd_lap["lap_completion"] == cfg.lap_completion_reward


def test_mlp_policy_and_rl_environment_integration(tmp_path: Path) -> None:
    """Verifica que MLPPolicy y CarRacingRLEnv ejecuten episodios y guarden/carguen pesos."""
    policy = MLPPolicy.create_warm_start(hidden_dims=(16, 16))
    save_file = tmp_path / "test_policy.npz"
    policy.save(save_file)
    loaded = MLPPolicy.load(save_file)
    assert np.allclose(policy.get_flat_params(), loaded.get_flat_params())

    engine = PymunkPhysicsEngine()
    sensor = LidarSensor2D(engine, max_range=300.0)
    track = Track.create_default_circuit("kart_classic")
    env = CarRacingRLEnv(physics_engine=engine, sensor=sensor, tracks=[track], max_steps_per_episode=60)

    obs, info = env.reset()
    assert obs.shape == (7,)
    assert info["score"] == 0

    for _ in range(30):
        action = loaded.predict_car_action(obs)
        obs, reward, terminated, truncated, step_info = env.step(action)
        assert obs.shape == (7,)
        assert isinstance(reward, float)
        if terminated or truncated:
            break
