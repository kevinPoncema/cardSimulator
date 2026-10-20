"""Motor de físicas 2D basado en Pymunk que implementa IPhysicsEngine y gestión de obstáculos."""

from __future__ import annotations

import math

import pymunk

from src.domain.entities.car import Car
from src.domain.entities.obstacle import Obstacle, ObstacleType
from src.domain.entities.track import Track
from src.domain.interfaces.physics_engine import IPhysicsEngine
from src.domain.value_objects.angle import Angle
from src.domain.value_objects.vector2d import Vector2D
from src.infrastructure.physics.pymunk_car_body import WALL_CATEGORY, PymunkCarBody

OBSTACLE_CATEGORY = 0b0100


class PymunkPhysicsEngine(IPhysicsEngine):
    """Implementación de IPhysicsEngine sobre pymunk.Space con soporte para obstáculos físicos."""

    def __init__(self) -> None:
        self.space = pymunk.Space()
        self.space.gravity = (0.0, 0.0)
        self.space.damping = 0.998
        self._car_bodies: dict[str, PymunkCarBody] = {}
        self._wall_shapes: list[pymunk.Segment] = []
        self._obstacle_items: list[tuple[pymunk.Body, pymunk.Shape, Obstacle]] = []
        self._track: Track | None = None

    def register_track(self, track: Track) -> None:
        """Registra los muros estáticos perimetrales y obstáculos físicos de la pista en Pymunk."""
        self._track = track

        # Limpiar muros previos
        for old_shape in self._wall_shapes:
            self.space.remove(old_shape)
        self._wall_shapes.clear()

        # Limpiar obstáculos físicos previos
        for body, shape, _ in self._obstacle_items:
            if body.body_type == pymunk.Body.DYNAMIC:
                self.space.remove(body, shape)
            else:
                self.space.remove(shape)
        self._obstacle_items.clear()

        # 1. Registrar muros perimetrales
        for seg in track.all_wall_segments():
            pymunk_seg = pymunk.Segment(
                self.space.static_body,
                seg.start.to_tuple(),
                seg.end.to_tuple(),
                3.0,
            )
            pymunk_seg.elasticity = seg.elasticity
            pymunk_seg.friction = seg.friction
            pymunk_seg.filter = pymunk.ShapeFilter(categories=WALL_CATEGORY)
            self.space.add(pymunk_seg)
            self._wall_shapes.append(pymunk_seg)

        # 2. Registrar obstáculos físicos (conos y barreras)
        for obs in track.obstacles:
            if obs.kind == ObstacleType.CONE:
                # Conos: Cuerpos dinámicos ligeros que se desplazan al ser golpeados
                mass = 0.08
                moment = pymunk.moment_for_circle(mass, 0, obs.radius)
                body = pymunk.Body(mass, moment)
                body.position = (obs.position.x, obs.position.y)
                shape = pymunk.Circle(body, obs.radius)
                shape.elasticity = 0.35
                shape.friction = 0.85
                shape.filter = pymunk.ShapeFilter(categories=OBSTACLE_CATEGORY)
                self.space.add(body, shape)
                self._obstacle_items.append((body, shape, obs))

            elif obs.kind == ObstacleType.BARRIER:
                # Barreras de hormigón: Cuerpos estáticos rígidos con ángulo
                body = pymunk.Body(body_type=pymunk.Body.STATIC)
                body.position = (obs.position.x, obs.position.y)
                body.angle = math.radians(obs.angle_deg)
                shape = pymunk.Poly.create_box(body, (obs.width, obs.height), radius=1.0)
                shape.elasticity = 0.20
                shape.friction = 0.40
                shape.filter = pymunk.ShapeFilter(categories=WALL_CATEGORY)
                self.space.add(body, shape)
                self._obstacle_items.append((body, shape, obs))

    def register_car(self, car: Car) -> None:
        """Crea y registra el PymunkCarBody asociado a la entidad Car."""
        if car.car_id in self._car_bodies:
            old_cb = self._car_bodies.pop(car.car_id)
            self.space.remove(old_cb.body, old_cb.shape)
        car_body = PymunkCarBody(car)
        self._car_bodies[car.car_id] = car_body
        self.space.add(car_body.body, car_body.shape)
        car_body.sync_to_domain()

    def reset_car_pose(self, car_id: str, position: Vector2D, angle: Angle) -> None:
        """Reposiciona un coche en la pista (ej. al cambiar de mapa o reiniciar)."""
        cb = self._car_bodies.get(car_id)
        if cb is not None:
            cb.reset_pose(position, angle)

    def reset_obstacles(self) -> None:
        """Restaura los conos y obstáculos físicos a sus posiciones y velocidades originales."""
        for body, _, obs in self._obstacle_items:
            if body.body_type == pymunk.Body.DYNAMIC:
                body.position = (obs.position.x, obs.position.y)
                body.velocity = (0.0, 0.0)
                body.angular_velocity = 0.0

    def get_physical_obstacles_state(self) -> list[tuple[Obstacle, Vector2D, float]]:
        """Retorna las posiciones y ángulos actuales de los obstáculos físicos para renderizado."""
        states: list[tuple[Obstacle, Vector2D, float]] = []
        for body, _, obs in self._obstacle_items:
            pos = Vector2D(float(body.position.x), float(body.position.y))
            ang_deg = math.degrees(float(body.angle))
            states.append((obs, pos, ang_deg))
        return states

    def get_car_body(self, car_id: str) -> PymunkCarBody | None:
        """Retorna el adaptador PymunkCarBody para consultas internas (ej. filtro LIDAR)."""
        return self._car_bodies.get(car_id)

    def step(self, dt: float) -> None:
        """Aplica dinámicas vehiculares, fricción en obstáculos, avanza pymunk.Space y sincroniza."""
        for car_body in self._car_bodies.values():
            surface = None
            if self._track is not None:
                surface = self._track.get_surface_at(car_body.car.position)
            car_body.apply_controls_and_friction(dt, surface)

        # Amortiguar movimiento de conos dinámicos tras ser golpeados
        for body, _, obs in self._obstacle_items:
            if body.body_type == pymunk.Body.DYNAMIC:
                body.velocity *= max(0.0, 1.0 - 2.8 * dt)
                body.angular_velocity *= max(0.0, 1.0 - 4.5 * dt)

        self.space.step(dt)
        self.sync_entities()

    def sync_entities(self) -> None:
        """Sincroniza el estado físico de todos los cuerpos con sus entidades de dominio."""
        for car_body in self._car_bodies.values():
            car_body.sync_to_domain()
