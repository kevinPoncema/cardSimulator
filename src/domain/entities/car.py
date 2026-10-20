"""Entidad de dominio Car con validación de vueltas completas, monedas, sentido y desatasco."""

from __future__ import annotations

from dataclasses import dataclass, field

from src.domain.interfaces.controllable import IControllable
from src.domain.value_objects.angle import Angle
from src.domain.value_objects.car_action import CarAction
from src.domain.value_objects.color import CAR_PALETTE, Color
from src.domain.value_objects.speed import Speed
from src.domain.value_objects.vector2d import Vector2D


@dataclass(frozen=True)
class CarConfig:
    """Parámetros dinámicos arcade configurables de un vehículo.

    Ajustados por defecto para ofrecer un manejo ágil, predecible y divertido
    para jugadores humanos y agentes RL.
    """

    drive_force: float = 1350.0
    brake_force: float = 1150.0
    turn_rate: float = 3.2
    longitudinal_traction: float = 1.0
    lateral_grip: float = 0.96
    handbrake_lateral_grip: float = 0.55
    longitudinal_drag: float = 1.8
    max_forward_speed: float = 480.0
    max_reverse_speed: float = 190.0
    max_angular_velocity: float = 4.5
    length: float = 46.0
    width: float = 26.0
    mass: float = 1.5


@dataclass
class Car(IControllable):
    """Entidad de dominio que representa un vehículo de carreras 2D y su estado de carrera."""

    car_id: str
    position: Vector2D = field(default_factory=lambda: Vector2D(0.0, 0.0))
    velocity: Vector2D = field(default_factory=lambda: Vector2D(0.0, 0.0))
    angle: Angle = field(default_factory=lambda: Angle(0.0))
    angular_velocity: float = 0.0
    color: Color = field(default_factory=lambda: CAR_PALETTE["crimson_red"])
    config: CarConfig = field(default_factory=CarConfig)
    current_action: CarAction = field(default_factory=CarAction.idle)
    current_surface_name: str = "asphalt"
    current_surface_friction: float = 1.0

    # Estado de puntuación, vueltas, monedas y colisiones
    score: int = 0
    laps_completed: int = 0
    coins_collected: int = 0
    next_checkpoint_id: int = 1
    checkpoints_passed_in_lap: int = 0
    is_wrong_way: bool = False
    track_heading_alignment: float = 1.0
    active_collisions: set[str] = field(default_factory=set)
    stuck_against_wall_steps: int = 0

    @property
    def current_lap(self) -> int:
        """Retorna el número de vuelta en la que va el coche actualmente (1-indexed)."""
        return self.laps_completed + 1

    def apply_action(self, action: CarAction) -> None:
        """Registra la intención de control del piloto/IA para el paso físico actual."""
        if not isinstance(action, CarAction):
            raise TypeError("Se esperaba una instancia de CarAction.")
        self.current_action = action

    def collect_coin(self, points: int = 1) -> None:
        """Registra la recolección de una moneda en la pista y suma sus puntos."""
        self.coins_collected += 1
        self.score += int(points)

    def reset_lap_progress(self, reset_score: bool = False) -> None:
        """Reinicia el progreso de sectores de la vuelta actual (y opcionalmente los puntos)."""
        self.next_checkpoint_id = 1
        self.checkpoints_passed_in_lap = 0
        self.is_wrong_way = False
        self.active_collisions.clear()
        self.stuck_against_wall_steps = 0
        if reset_score:
            self.score = 0
            self.laps_completed = 0
            self.coins_collected = 0

    def record_gate_crossing(
        self, gate_id: int, is_forward: bool, total_gates: int = 4
    ) -> bool:
        """Valida el cruce de un checkpoint o línea de meta y actualiza vueltas y puntos.

        Returns:
            True si se acaba de completar una vuelta válida y se sumó +1 punto; False en otro caso.
        """
        if total_gates < 2:
            raise ValueError("El circuito debe tener al menos 2 puertas de control.")

        if is_forward:
            if gate_id == self.next_checkpoint_id:
                if gate_id == 0 and self.checkpoints_passed_in_lap == total_gates - 1:
                    # ¡Vuelta completa válida en el sentido correcto! +1 punto y +1 vuelta
                    self.score += 1
                    self.laps_completed += 1
                    self.checkpoints_passed_in_lap = 0
                    self.next_checkpoint_id = 1
                    return True
                elif gate_id > 0:
                    self.checkpoints_passed_in_lap += 1
                    self.next_checkpoint_id = (gate_id + 1) % total_gates
        else:
            if gate_id == 0:
                # Cruce de meta en sentido contrario: penalización severa en el marcador
                self.score = max(0, self.score - 5)
                self.checkpoints_passed_in_lap = 0
                self.next_checkpoint_id = 1
            # Si regresa en sentido contrario por el último checkpoint validado, revierte el avance
            elif self.checkpoints_passed_in_lap > 0 and gate_id == self.checkpoints_passed_in_lap:
                self.checkpoints_passed_in_lap -= 1
                self.next_checkpoint_id = gate_id

        return False

    def update_kinematics(
        self,
        position: Vector2D,
        velocity: Vector2D,
        angle: Angle,
        angular_velocity: float,
    ) -> None:
        """Sincroniza el estado cinemático calculado por el motor de físicas."""
        self.position = position
        self.velocity = velocity
        self.angle = angle
        self.angular_velocity = float(angular_velocity)

    def forward_unit_vector(self) -> Vector2D:
        """Vector unitario longitudinal apuntando hacia el frente del chasis."""
        return self.angle.to_unit_vector()

    def right_unit_vector(self) -> Vector2D:
        """Vector unitario perpendicular lateral del chasis."""
        return self.angle.to_right_vector()

    def local_to_global_point(self, local_point: Vector2D) -> Vector2D:
        """Transforma un punto en coordenadas locales del chasis a coordenadas globales."""
        return self.position + local_point.rotate(self.angle.radians)

    def front_center_position(self) -> Vector2D:
        """Retorna la posición global del centro delantero del chasis (origen LIDAR)."""
        return self.local_to_global_point(Vector2D(self.config.length * 0.5, 0.0))

    def forward_speed(self) -> float:
        """Velocidad lineal actual proyectada en el eje longitudinal del vehículo (v)."""
        return self.velocity.dot(self.forward_unit_vector())

    def lateral_speed(self) -> float:
        """Velocidad lineal actual proyectada en el eje lateral del vehículo."""
        return self.velocity.dot(self.right_unit_vector())

    def speed_object(self) -> Speed:
        """Retorna el Value Object Speed correspondiente a la velocidad longitudinal."""
        return Speed(
            value=self.forward_speed(),
            max_forward=self.config.max_forward_speed,
            max_reverse=self.config.max_reverse_speed,
        )

    def normalized_forward_speed(self) -> float:
        """Retorna v / v_max acotado en [-1.0, 1.0]."""
        return self.speed_object().normalized()

    def normalized_angular_velocity(self) -> float:
        """Retorna omega / omega_max acotado en [-1.0, 1.0]."""
        if self.config.max_angular_velocity <= 0:
            return 0.0
        norm = self.angular_velocity / self.config.max_angular_velocity
        return max(-1.0, min(1.0, norm))

    def effective_lateral_grip(self) -> float:
        """Retorna el coeficiente de agarre lateral activo considerando freno de mano."""
        if self.current_action.handbrake:
            return self.config.handbrake_lateral_grip
        return self.config.lateral_grip
