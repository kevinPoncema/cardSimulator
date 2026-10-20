"""Controlador heurístico autónomo (Dummy AI) que consume el vector de observación LIDAR."""

from __future__ import annotations

from src.domain.interfaces.input_controller import IInputController
from src.domain.interfaces.telemetry_provider import ITelemetryProvider
from src.domain.value_objects.car_action import CarAction


class DummyAiController(IInputController):
    """Agente de prueba (LSP / OCP) que conduce guiándose por el vector LIDAR normalizado."""

    def __init__(
        self,
        telemetry_provider: ITelemetryProvider | None = None,
        car_id: str | None = None,
        target_throttle: float = 0.65,
    ) -> None:
        self.telemetry_provider = telemetry_provider
        self.car_id = car_id
        self.target_throttle = max(0.0, min(1.0, float(target_throttle)))

    def get_action(self) -> CarAction:
        """Calcula una acción reactiva sin efectos colaterales sobre el dominio."""
        if self.telemetry_provider is None:
            return CarAction(throttle=self.target_throttle, brake=0.0, steer=0.0, handbrake=False)

        obs = self.telemetry_provider.get_observation(self.car_id)
        if obs.shape[0] < 5:
            return CarAction(throttle=self.target_throttle, brake=0.0, steer=0.0, handbrake=False)

        d_0 = float(obs[0])
        d_45_left = float(obs[1])
        d_45_right = float(obs[2])
        d_90_left = float(obs[3])
        d_90_right = float(obs[4])

        # Diferencia entre espacio libre diagonal/lateral para orientar el volante
        steer_signal = (d_45_left - d_45_right) * 1.4 + (d_90_left - d_90_right) * 0.6
        steer = max(-1.0, min(1.0, steer_signal))

        throttle = self.target_throttle if d_0 > 0.35 else 0.2
        brake = 0.4 if d_0 < 0.22 else 0.0
        handbrake = bool(d_0 < 0.30 and abs(steer) > 0.75)

        return CarAction(
            throttle=throttle,
            brake=brake,
            steer=steer,
            handbrake=handbrake,
        )
