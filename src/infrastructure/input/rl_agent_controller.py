"""Controlador de Red Neuronal (RLAgentController) que ejecuta el modelo entrenado en vivo."""

from __future__ import annotations

from pathlib import Path

import pygame

from src.domain.interfaces.input_controller import IInputController
from src.domain.interfaces.telemetry_provider import ITelemetryProvider
from src.domain.value_objects.car_action import CarAction
from src.infrastructure.ai.mlp_policy import MLPPolicy
from src.infrastructure.input.human_keyboard_controller import KEY_TO_MAP


class RLAgentController(IInputController):
    """Controlador autónomo que utiliza la red neuronal MLPPolicy entrenada por refuerzo."""

    def __init__(
        self,
        telemetry_provider: ITelemetryProvider,
        car_id: str,
        policy: MLPPolicy | None = None,
        model_path: str | Path | None = None,
        handle_window_events: bool = True,
    ) -> None:
        self.telemetry_provider = telemetry_provider
        self.car_id = car_id
        self.handle_window_events = handle_window_events

        if policy is not None:
            self.policy = policy
        elif model_path is not None and Path(model_path).exists():
            self.policy = MLPPolicy.load(model_path)
        else:
            self.policy = MLPPolicy.create_warm_start()

        self._quit_requested: bool = False
        self._debug_toggle_requested: bool = False
        self._map_selection_requested: str | None = None
        self._reset_car_requested: bool = False

        self._is_recovering_direction: bool = False
        self._wall_stuck_counter: int = 0
        self._reverse_escape_counter: int = 0

    def _process_events(self) -> None:
        """Permite cambiar de mapa (M/N, 1-0), alternar F3 o salir (ESC) mientras conduce la IA."""
        if not self.handle_window_events or not pygame.get_init():
            return
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._quit_requested = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self._quit_requested = True
                elif event.key == pygame.K_F3:
                    self._debug_toggle_requested = True
                elif event.key == pygame.K_m:
                    self._map_selection_requested = "next"
                elif event.key == pygame.K_n:
                    self._map_selection_requested = "prev"
                elif event.key in KEY_TO_MAP:
                    self._map_selection_requested = KEY_TO_MAP[event.key]
                elif event.key == pygame.K_r:
                    self._reset_car_requested = True

    def get_action(self) -> CarAction:
        """Consulta el vector de observación S y ejecuta la inferencia de la red neuronal."""
        self._process_events()
        obs = self.telemetry_provider.get_observation(self.car_id)
        if obs.size < 7:
            return CarAction.idle()

        norm_v = float(obs[5]) if obs.size > 5 else 0.0
        d_0 = float(obs[0]) if obs.size > 0 else 1.0

        # 1. PRIORIDAD 1: Desatasco en reversa recta si el morro queda trabado contra un obstáculo frontal
        # Si el coche está bloqueado contra un muro, jamás podrá rotar hacia él; debe retroceder primero.
        is_stuck_slow = abs(norm_v) < 0.05
        if is_stuck_slow and (d_0 < 0.22 or self._is_recovering_direction):
            self._wall_stuck_counter += 1
            if self._wall_stuck_counter > 12:
                self._reverse_escape_counter = 18
                self._wall_stuck_counter = 0
                self._is_recovering_direction = False
        else:
            self._wall_stuck_counter = 0

        if self._reverse_escape_counter > 0:
            self._reverse_escape_counter -= 1
            if d_0 > 0.32 and self._reverse_escape_counter < 10:
                self._reverse_escape_counter = 0
            else:
                return CarAction(
                    throttle=0.0,
                    brake=0.85,
                    steer=0.0,  # Reversa estrictamente recta para evitar inversiones de 180°
                    handbrake=False,
                )

        # 2. PRIORIDAD 2: Histéresis de recuperación de sentido correcto
        # Se activa si el coche queda encarado hacia atrás (ángulo > 96°, alignment < -0.10)
        # y se mantiene activo hasta realinear hacia adelante (alignment > 0.45, ángulo < 63°)
        alignment_fn = getattr(self.telemetry_provider, "get_heading_alignment", None)
        alignment = alignment_fn(self.car_id) if alignment_fn is not None else 1.0

        if not self._is_recovering_direction and alignment < -0.10:
            self._is_recovering_direction = True
        elif self._is_recovering_direction and alignment > 0.45:
            self._is_recovering_direction = False

        if self._is_recovering_direction:
            turn_dir_fn = getattr(self.telemetry_provider, "get_turn_to_track_dir", None)
            turn_dir = turn_dir_fn(self.car_id) if turn_dir_fn is not None else 1.0
            return CarAction(
                throttle=0.55,
                brake=0.0,
                steer=float(turn_dir),
                handbrake=False,
            )

        # 3. CONDUCCIÓN NORMAL POR RED NEURONAL
        return self.policy.predict_car_action(obs)

    def poll_debug_toggle(self) -> bool:
        self._process_events()
        toggled = self._debug_toggle_requested
        self._debug_toggle_requested = False
        return toggled

    def poll_map_selection(self) -> str | None:
        self._process_events()
        selected = self._map_selection_requested
        self._map_selection_requested = None
        return selected

    def poll_reset_car(self) -> bool:
        self._process_events()
        reset = self._reset_car_requested
        self._reset_car_requested = False
        return reset

    def poll_quit(self) -> bool:
        self._process_events()
        return self._quit_requested
