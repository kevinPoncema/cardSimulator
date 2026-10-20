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
