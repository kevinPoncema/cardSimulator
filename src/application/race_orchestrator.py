"""Orquestador del bucle principal de simulación, validación de vueltas, monedas y carrera."""

from __future__ import annotations

import numpy as np

from src.application.telemetry_collector import TelemetryCollector
from src.domain.entities.car import Car
from src.domain.entities.track import Track
from src.domain.interfaces.input_controller import IInputController
from src.domain.interfaces.physics_engine import IPhysicsEngine
from src.domain.interfaces.renderer import IRenderer
from src.domain.interfaces.sensor import ISensor
from src.domain.value_objects.vector2d import Vector2D


class RaceOrchestrator:
    """Coordina las entidades de dominio y los adaptadores de infraestructura sin acoplamiento."""

    def __init__(
        self,
        track: Track,
        physics_engine: IPhysicsEngine,
        sensor: ISensor,
        renderer: IRenderer | None = None,
        telemetry_collector: TelemetryCollector | None = None,
        debug_mode: bool = True,
        available_tracks: dict[str, Track] | None = None,
        current_map_key: str = "kart_classic",
    ) -> None:
        self.track = track
        self.physics_engine = physics_engine
        self.sensor = sensor
        self.renderer = renderer
        self.telemetry = telemetry_collector or TelemetryCollector()
        self.debug_mode = debug_mode
        self.available_tracks = available_tracks or {current_map_key: track}
        self.current_map_key = current_map_key

        self.cars: list[Car] = []
        self.controllers: dict[str, IInputController] = {}
        self.is_running: bool = False

        self.physics_engine.register_track(self.track)

    def compute_grid_spawn_position(self, car_index: int) -> Vector2D:
        """Calcula una posición escalonada en la parrilla de salida para soportar múltiples coches."""
        if car_index == 0:
            return self.track.start_position
        fwd = self.track.start_angle.to_unit_vector()
        right = self.track.start_angle.to_right_vector()
        row = car_index // 2
        col_sign = 1.0 if (car_index % 2 == 1) else -1.0
        lateral_offset = col_sign * min(34.0, self.track.asphalt_half_width * 0.35)
        longitudinal_offset = -float(row) * 56.0
        return self.track.start_position + fwd * longitudinal_offset + right * lateral_offset

    def add_participant(self, car: Car, controller: IInputController) -> None:
        """Registra un vehículo y su controlador (humano o IA) en la sesión de carrera."""
        self.cars.append(car)
        self.controllers[car.car_id] = controller
        self.physics_engine.register_car(car)
        initial_reading = self.sensor.scan(car)
        self.telemetry.collect(car, initial_reading)

    def switch_track(self, map_key_or_next: str) -> None:
        """Cambia en caliente el circuito activo y reposiciona los vehículos en la línea de meta."""
        keys = list(self.available_tracks.keys())
        if not keys:
            return
        if map_key_or_next == "next":
            idx = keys.index(self.current_map_key) if self.current_map_key in keys else 0
            target_key = keys[(idx + 1) % len(keys)]
        elif map_key_or_next == "prev":
            idx = keys.index(self.current_map_key) if self.current_map_key in keys else 0
            target_key = keys[(idx - 1) % len(keys)]
        elif map_key_or_next in self.available_tracks:
            target_key = map_key_or_next
        else:
            target_key = map_key_or_next
            self.available_tracks[target_key] = Track.create_default_circuit(target_key)

        self.current_map_key = target_key
        self.track = self.available_tracks[target_key]
        self.physics_engine.register_track(self.track)
        self.reset_all_cars()

    def reset_all_cars(self) -> None:
        """Reposiciona todos los coches, regenera monedas y restaura obstáculos."""
        self.physics_engine.reset_obstacles()
        self.track.spawn_random_coins()
        for idx, car in enumerate(self.cars):
            spawn_pos = self.compute_grid_spawn_position(idx)
            car.reset_lap_progress(reset_score=False)
            self.physics_engine.reset_car_pose(
                car.car_id,
                spawn_pos,
                self.track.start_angle,
            )
            reading = self.sensor.scan(car)
            self.telemetry.collect(car, reading)

    def step_simulation(self, dt: float) -> bool:
        """Ejecuta un paso lógico y físico completo de duración dt segundos.

        Returns:
            False si algún controlador solicitó finalizar la simulación, True en caso contrario.
        """
        for car in self.cars:
            controller = self.controllers[car.car_id]
            if controller.poll_quit():
                self.is_running = False
                return False
            if controller.poll_debug_toggle():
                self.debug_mode = not self.debug_mode

            map_sel = controller.poll_map_selection()
            if map_sel is not None:
                self.switch_track(map_sel)

            if controller.poll_reset_car():
                self.reset_all_cars()

            action = controller.get_action()
            car.apply_action(action)

        # Guardar posiciones previas para detectar cruce exacto de puertas (checkpoints y meta)
        prev_positions = {car.car_id: car.position for car in self.cars}

        # Avanzar motor de físicas y sincronizar estados cinemáticos
        self.physics_engine.step(dt)

        # Validar cruce de puertas, recolección de monedas, sentido contrario y LIDAR
        num_gates = len(self.track.gates)
        for car in self.cars:
            prev_pos = prev_positions[car.car_id]
            if num_gates >= 2:
                crossings = self.track.check_gate_crossings(prev_pos, car.position)
                for gate_id, is_forward in crossings:
                    car.record_gate_crossing(gate_id, is_forward, total_gates=num_gates)

            # Recolección de monedas alcanzables sobre la pista
            collected_coins = self.track.collect_touched_coins(
                car.position,
                collection_radius=car.config.width * 0.55,
                respawn=True,
            )
            for coin in collected_coins:
                car.collect_coin(coin.value)

            # Detectar si el vehículo circula en sentido contrario a la pista
            if car.velocity.magnitude() > 35.0:
                tangent = self.track.closest_centerline_tangent(car.position)
                car.is_wrong_way = car.velocity.dot(tangent) < -25.0
            else:
                car.is_wrong_way = False

            reading = self.sensor.scan(car)
            self.telemetry.collect(car, reading)

        return True

    def render_frame(self, fps: float = 60.0) -> None:
        """Delega el dibujado del estado actual al IRenderer si está configurado."""
        if self.renderer is not None:
            dyn_obs = self.physics_engine.get_physical_obstacles_state()
            self.renderer.render(
                track=self.track,
                cars=self.cars,
                readings=self.telemetry.all_readings,
                observations=self.telemetry.all_observations,
                debug_mode=self.debug_mode,
                fps=fps,
                dynamic_obstacles=dyn_obs,
            )

    def get_observation(self, car_id: str | None = None) -> np.ndarray:
        """Expone directamente el vector de observación normalizado S."""
        return self.telemetry.get_observation(car_id)

    def shutdown(self) -> None:
        """Cierra limpiamente los recursos de infraestructura."""
        self.is_running = False
        if self.renderer is not None:
            self.renderer.close()
