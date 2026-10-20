"""Controlador de teclado humano usando eventos de Pygame-CE."""

from __future__ import annotations

import pygame

from src.domain.interfaces.input_controller import IInputController
from src.domain.value_objects.car_action import CarAction


class HumanKeyboardController(IInputController):
    """Lee el estado del teclado (WASD/Flechas, Espacio, F3, 1-4/M para mapas, R para reiniciar)."""

    def __init__(self) -> None:
        self._quit_requested: bool = False
        self._debug_toggle_requested: bool = False
        self._map_selection_requested: str | None = None
        self._reset_car_requested: bool = False

    def _process_events(self) -> None:
        """Procesa la cola de eventos de Pygame para detectar cierre, F3, cambio de mapa y reset."""
        if not pygame.get_init():
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
                elif event.key == pygame.K_1:
                    self._map_selection_requested = "retro_grand_prix"
                elif event.key == pygame.K_2:
                    self._map_selection_requested = "neon_speedway"
                elif event.key == pygame.K_3:
                    self._map_selection_requested = "canyon_drift"
                elif event.key == pygame.K_4:
                    self._map_selection_requested = "starlight_loop"
                elif event.key == pygame.K_r:
                    self._reset_car_requested = True

    def get_action(self) -> CarAction:
        """Retorna el CarAction correspondiente a las teclas presionadas actualmente."""
        self._process_events()
        if not pygame.get_init():
            return CarAction.idle()

        keys = pygame.key.get_pressed()

        throttle = 1.0 if (keys[pygame.K_w] or keys[pygame.K_UP]) else 0.0
        brake = 1.0 if (keys[pygame.K_s] or keys[pygame.K_DOWN]) else 0.0

        steer = 0.0
        if keys[pygame.K_a] or keys[pygame.K_LEFT]:
            steer -= 1.0
        if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            steer += 1.0

        handbrake = bool(keys[pygame.K_SPACE])

        return CarAction(
            throttle=throttle,
            brake=brake,
            steer=steer,
            handbrake=handbrake,
        )

    def poll_debug_toggle(self) -> bool:
        """Consume y retorna el evento de alternancia de depuración (F3)."""
        self._process_events()
        toggled = self._debug_toggle_requested
        self._debug_toggle_requested = False
        return toggled

    def poll_map_selection(self) -> str | None:
        """Consume y retorna la solicitud de cambio de mapa (M o teclas 1-4)."""
        self._process_events()
        selected = self._map_selection_requested
        self._map_selection_requested = None
        return selected

    def poll_reset_car(self) -> bool:
        """Consume y retorna la solicitud de reinicio de posición del coche (tecla R)."""
        self._process_events()
        reset = self._reset_car_requested
        self._reset_car_requested = False
        return reset

    def poll_quit(self) -> bool:
        """Retorna True si el usuario cerró la ventana o presionó ESC."""
        self._process_events()
        return self._quit_requested
