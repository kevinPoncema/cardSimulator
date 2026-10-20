"""Cuerpo rígido vehicular en Pymunk con físicas arcade simplificadas y asistidas."""

from __future__ import annotations

import math

import pymunk

from src.domain.entities.car import Car
from src.domain.entities.track import SURFACE_CATALOG, SurfaceProperties
from src.domain.value_objects.angle import Angle
from src.domain.value_objects.vector2d import Vector2D

WALL_CATEGORY = 0b0001
CAR_CATEGORY = 0b0010


class PymunkCarBody:
    """Adaptador físico arcade que vincula una entidad Car con un Body/Poly de Pymunk."""

    def __init__(self, car: Car) -> None:
        self.car = car
        cfg = car.config
        moment = pymunk.moment_for_box(cfg.mass, (cfg.length, cfg.width))
        self.body = pymunk.Body(cfg.mass, moment)
        self.body.position = (car.position.x, car.position.y)
        self.body.angle = car.angle.radians

        self.shape = pymunk.Poly.create_box(self.body, (cfg.length, cfg.width), radius=3.0)
        # Baja elasticidad y baja fricción con muros para deslizar suavemente sin rebotes bruscos
        self.shape.elasticity = 0.15
        self.shape.friction = 0.12
        self.shape.filter = pymunk.ShapeFilter(categories=CAR_CATEGORY)

    def reset_pose(self, position: Vector2D, angle: Angle) -> None:
        """Reposiciona el cuerpo físico sin velocidad lineal ni angular."""
        self.body.position = (position.x, position.y)
        self.body.angle = angle.radians
        self.body.velocity = (0.0, 0.0)
        self.body.angular_velocity = 0.0
        self.sync_to_domain()

    def apply_controls_and_friction(
        self, dt: float, surface: SurfaceProperties | None = None
    ) -> None:
        """Aplica físicas arcade: respuesta inmediata, giro sin inercia residual y derrape asistido."""
        if dt <= 0:
            return

        surf = surface or SURFACE_CATALOG["asphalt"]
        self.car.current_surface_name = surf.name
        self.car.current_surface_friction = surf.drag_multiplier

        cfg = self.car.config
        action = self.car.current_action

        angle = self.body.angle
        fwd = pymunk.Vec2d(math.cos(angle), math.sin(angle))
        right = pymunk.Vec2d(-math.sin(angle), math.cos(angle))

        vel = self.body.velocity
        v_fwd = vel.dot(fwd)
        v_lat = vel.dot(right)

        # 1. Aceleración y frenado directo e intuitivo
        longitudinal_accel = 0.0
        if action.throttle > 0.0:
            longitudinal_accel += (
                action.throttle
                * cfg.drive_force
                * cfg.longitudinal_traction
                * surf.traction_multiplier
            )
        if action.brake > 0.0:
            if v_fwd > 10.0:
                # Frenado potente y predecible
                longitudinal_accel -= action.brake * cfg.brake_force * 1.4
            else:
                # Reversa inmediata y controlada
                longitudinal_accel -= (
                    action.brake * cfg.brake_force * 0.85 * surf.traction_multiplier
                )

        v_fwd_new = v_fwd + (longitudinal_accel / cfg.mass) * dt

        # 2. Freno motor automático al soltar el acelerador (facilita tomar curvas sin salirse)
        effective_drag = cfg.longitudinal_drag * surf.drag_multiplier
        if action.throttle == 0.0 and action.brake == 0.0:
            effective_drag *= 1.75
        if action.handbrake:
            effective_drag += 1.4

        drag_decay = max(0.0, 1.0 - effective_drag * dt)
        v_fwd_new *= drag_decay
        if abs(v_fwd_new) < 1.5 and action.throttle == 0.0 and action.brake == 0.0:
            v_fwd_new = 0.0

        # Tope de velocidad suave según superficie
        surface_speed_cap = cfg.max_forward_speed
        if surf.name == "kerb":
            surface_speed_cap = cfg.max_forward_speed * 0.92
        elif surf.name == "grass":
            surface_speed_cap = cfg.max_forward_speed * 0.65

        v_fwd_new = max(-cfg.max_reverse_speed, min(surface_speed_cap, v_fwd_new))

        # 3. Agarre lateral alto en modo normal y derrape divertido con Espacio (freno de mano)
        base_grip = self.car.effective_lateral_grip() * surf.lateral_grip_multiplier
        lateral_retention = max(0.0, min(1.0, 1.0 - base_grip))
        v_lat_new = v_lat * (lateral_retention ** (dt * 60.0))

        self.body.velocity = fwd * v_fwd_new + right * v_lat_new

        # 4. Giro directo y fluido (permite rotar para despegarse de muros incluso detenido si se presiona acelerador)
        speed_mag = abs(v_fwd_new)
        is_active_turn = abs(action.steer) > 1e-4 and (
            speed_mag > 0.8
            or action.throttle > 0.05
            or action.brake > 0.05
        )
        if is_active_turn:
            low_speed_ramp = max(0.70, min(1.0, speed_mag / 30.0))
            high_speed_stability = 1.0 - 0.20 * min(1.0, speed_mag / cfg.max_forward_speed)
            direction_sign = 1.0 if v_fwd_new >= -2.0 else -1.0
            drift_turn_boost = 1.18 if action.handbrake else 1.0

            target_omega = (
                action.steer
                * cfg.turn_rate
                * low_speed_ramp
                * high_speed_stability
                * direction_sign
                * drift_turn_boost
            )
            self.body.angular_velocity = max(
                -cfg.max_angular_velocity,
                min(cfg.max_angular_velocity, target_omega),
            )
        else:
            # Detener la rotación inmediatamente cuando el humano o IA suelta el volante
            self.body.angular_velocity = 0.0

    def sync_to_domain(self) -> None:
        """Copia posición, velocidad, ángulo y velocidad angular hacia la entidad Car."""
        pos = Vector2D(float(self.body.position.x), float(self.body.position.y))
        vel = Vector2D(float(self.body.velocity.x), float(self.body.velocity.y))
        ang = Angle(radians=float(self.body.angle)).normalized()
        omega = float(self.body.angular_velocity)
        self.car.update_kinematics(
            position=pos,
            velocity=vel,
            angle=ang,
            angular_velocity=omega,
        )
