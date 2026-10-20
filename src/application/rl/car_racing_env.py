"""Entorno de Aprendizaje por Refuerzo (API compatible con Gymnasium) para el simulador 2D."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.application.telemetry_collector import TelemetryCollector
from src.application.rl.reward_config import RewardCalculator, RewardConfig
from src.domain.entities.car import Car, CarConfig
from src.domain.entities.sensor_reading import DEFAULT_LIDAR_MAX_RANGE
from src.domain.entities.track import Track
from src.domain.interfaces.physics_engine import IPhysicsEngine
from src.domain.interfaces.renderer import IRenderer
from src.domain.interfaces.sensor import ISensor
from src.domain.value_objects.car_action import CarAction
from src.domain.value_objects.color import CAR_PALETTE


class CarRacingRLEnv:
    """Entorno estándar de Reinforcement Learning (`reset`, `step`, `render`, `close`).

    Espacio de Observación (7 dimensiones, float32):
        S = [d_0, d_45_left, d_45_right, d_90_left, d_90_right, v/v_max, omega/omega_max]

    Espacio de Acción (2 dimensiones continuas en [-1.0, 1.0], o CarAction directo):
        action[0]: Acelerador (>0) / Freno (<0) en [-1.0, 1.0]
        action[1]: Dirección (steer) en [-1.0, 1.0]
    """

    observation_dim: int = 7
    action_dim: int = 2

    def __init__(
        self,
        physics_engine: IPhysicsEngine,
        sensor: ISensor,
        tracks: list[Track] | None = None,
        reward_config: RewardConfig | None = None,
        renderer: IRenderer | None = None,
        max_steps_per_episode: int = 1200,
        dt: float = 1.0 / 60.0,
        lidar_max_range: float = DEFAULT_LIDAR_MAX_RANGE,
    ) -> None:
        self.physics_engine = physics_engine
        self.sensor = sensor
        self.tracks = tracks or [Track.create_default_circuit("kart_classic")]
        self.current_track_idx = 0
        self.track = self.tracks[0]

        self.reward_calculator = RewardCalculator(reward_config)
        self.telemetry = TelemetryCollector()
        self.renderer = renderer
        self.max_steps_per_episode = int(max_steps_per_episode)
        self.dt = float(dt)
        self.lidar_max_range = float(lidar_max_range)

        self.car = Car(
            car_id="rl_agent",
            position=self.track.start_position,
            angle=self.track.start_angle,
            color=CAR_PALETTE["electric_blue"],
            config=CarConfig(),
        )

        self.physics_engine.register_track(self.track)
        self.physics_engine.register_car(self.car)

        self.current_step: int = 0
        self.episode_reward: float = 0.0
        self.total_checkpoints_passed: int = 0

    @staticmethod
    def vector_to_car_action(action: np.ndarray | list[float] | CarAction) -> CarAction:
        """Convierte un vector continuo [accel_brake, steer] en un objeto de valor CarAction."""
        if isinstance(action, CarAction):
            return action
        arr = np.asarray(action, dtype=np.float32).flatten()
        longitudinal = float(np.clip(arr[0], -1.0, 1.0)) if arr.size >= 1 else 0.0
        steer = float(np.clip(arr[1], -1.0, 1.0)) if arr.size >= 2 else 0.0
        handbrake = bool(arr[2] > 0.5) if arr.size >= 3 else False

        throttle = max(0.0, longitudinal)
        brake = max(0.0, -longitudinal)
        return CarAction(
            throttle=throttle,
            brake=brake,
            steer=steer,
            handbrake=handbrake,
        )

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reinicia el episodio, alternando opcionalmente de circuito si hay varios."""
        if options and "track_index" in options:
            self.current_track_idx = int(options["track_index"]) % len(self.tracks)
        elif options and options.get("cycle_tracks", False) and len(self.tracks) > 1:
            self.current_track_idx = (self.current_track_idx + 1) % len(self.tracks)

        self.track = self.tracks[self.current_track_idx]
        self.track.spawn_random_coins()
        self.physics_engine.register_track(self.track)
        self.physics_engine.reset_obstacles()

        self.car.reset_lap_progress(reset_score=True)
        self.physics_engine.reset_car_pose(
            self.car.car_id,
            self.track.start_position,
            self.track.start_angle,
        )

        self.reward_calculator.reset()
        self.current_step = 0
        self.episode_reward = 0.0
        self.total_checkpoints_passed = 0

        reading = self.sensor.scan(self.car)
        obs = self.telemetry.collect(self.car, reading)
        info = {
            "track_name": self.track.name,
            "score": self.car.score,
            "laps_completed": self.car.laps_completed,
            "coins_collected": self.car.coins_collected,
            "checkpoints": self.car.checkpoints_passed_in_lap,
        }
        return obs, info

    def step(
        self, action: np.ndarray | list[float] | CarAction
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Avanza un paso en el entorno y retorna (obs, reward, terminated, truncated, info)."""
        self.current_step += 1
        car_action = self.vector_to_car_action(action)
        self.car.apply_action(car_action)

        prev_pos = self.car.position
        prev_laps = self.car.laps_completed
        prev_cp = self.car.checkpoints_passed_in_lap

        self.physics_engine.step(self.dt)

        # Evaluar cruce de puertas (checkpoints y línea de meta)
        num_gates = len(self.track.gates)
        crossings: list[tuple[int, bool]] = []
        lap_completed = False
        if num_gates >= 2:
            crossings = self.track.check_gate_crossings(prev_pos, self.car.position)
            for gate_id, is_forward in crossings:
                if self.car.record_gate_crossing(gate_id, is_forward, total_gates=num_gates):
                    lap_completed = True

        checkpoint_advanced = (self.car.checkpoints_passed_in_lap > prev_cp) or lap_completed
        if checkpoint_advanced:
            self.total_checkpoints_passed += 1

        # Evaluar recolección de monedas en la pista
        collected_coins = self.track.collect_touched_coins(
            self.car.position,
            collection_radius=self.car.config.width * 0.55,
            respawn=True,
        )
        for coin in collected_coins:
            self.car.collect_coin(coin.value)

        # Detectar sentido contrario (por velocidad negativa o morro apuntando al revés)
        tangent = self.track.closest_centerline_tangent(self.car.position)
        heading_dot = self.car.forward_unit_vector().dot(tangent)
        vel_dot = self.car.velocity.dot(tangent)
        self.car.track_heading_alignment = float(heading_dot)
        if self.car.velocity.magnitude() > 25.0:
            self.car.is_wrong_way = bool(vel_dot < -15.0 or (heading_dot < -0.3 and vel_dot < 10.0))
        else:
            self.car.is_wrong_way = bool(heading_dot < -0.25)

        reading = self.sensor.scan(self.car)
        obs = self.telemetry.collect(self.car, reading)

        reward, terminated, breakdown = self.reward_calculator.compute_step_reward(
            car=self.car,
            track=self.track,
            reading=reading,
            gate_crossings=crossings,
            lap_completed=(self.car.laps_completed > prev_laps),
            checkpoint_advanced=checkpoint_advanced,
            coins_collected_count=len(collected_coins),
        )

        self.episode_reward += reward
        truncated = self.current_step >= self.max_steps_per_episode

        info: dict[str, Any] = {
            "step": self.current_step,
            "episode_reward": self.episode_reward,
            "score": self.car.score,
            "laps_completed": self.car.laps_completed,
            "coins_collected": self.car.coins_collected,
            "checkpoints_in_lap": self.car.checkpoints_passed_in_lap,
            "total_checkpoints": self.total_checkpoints_passed,
            "surface": self.car.current_surface_name,
            "active_collisions": list(self.car.active_collisions),
            "is_wrong_way": self.car.is_wrong_way,
            "reward_breakdown": breakdown,
        }
        return obs, reward, terminated, truncated, info

    def render(self, debug_mode: bool = True, fps: float = 60.0) -> None:
        """Renderiza el estado actual si hay un IRenderer conectado."""
        if self.renderer is not None:
            self.renderer.render(
                track=self.track,
                cars=[self.car],
                readings=self.telemetry.all_readings,
                observations=self.telemetry.all_observations,
                debug_mode=debug_mode,
                fps=fps,
                dynamic_obstacles=self.physics_engine.get_physical_obstacles_state(),
            )

    def close(self) -> None:
        """Libera recursos gráficos si están activos."""
        if self.renderer is not None:
            self.renderer.close()
