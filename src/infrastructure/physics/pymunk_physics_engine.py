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
HAZARD_SENSOR_CATEGORY = 0b1000


class PymunkPhysicsEngine(IPhysicsEngine):
    """Implementación de IPhysicsEngine sobre pymunk.Space con soporte para obstáculos y sensores RL."""

    def __init__(self) -> None:
        self.space = pymunk.Space()
        self.space.gravity = (0.0, 0.0)
        self.space.damping = 0.998
        self._car_bodies: dict[str, PymunkCarBody] = {}
        self._wall_shapes: list[pymunk.Segment] = []
        self._obstacle_items: list[tuple[pymunk.Body, pymunk.Shape, Obstacle]] = []
        self._hazard_sensor_shapes: list[pymunk.Shape] = []
        self._track: Track | None = None

    def register_track(self, track: Track) -> None:
        """Registra los muros estáticos perimetrales y obstáculos de la pista en Pymunk."""
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

        # Limpiar sensores de terreno previos (baches / manchas de aceite)
        for h_shape in self._hazard_sensor_shapes:
            self.space.remove(h_shape)
        self._hazard_sensor_shapes.clear()

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
            pymunk_seg.obstacle_tag = "wall"
            self.space.add(pymunk_seg)
            self._wall_shapes.append(pymunk_seg)

        # 2. Registrar obstáculos físicos (conos y barreras) y sensores de peligro (baches y aceite)
        for obs in track.obstacles:
            if obs.kind == ObstacleType.CONE:
                mass = 0.08
                moment = pymunk.moment_for_circle(mass, 0, obs.radius)
                body = pymunk.Body(mass, moment)
                body.position = (obs.position.x, obs.position.y)
                shape = pymunk.Circle(body, obs.radius)
                shape.elasticity = 0.35
                shape.friction = 0.85
                shape.filter = pymunk.ShapeFilter(categories=OBSTACLE_CATEGORY)
                shape.obstacle_tag = "cone"
                self.space.add(body, shape)
                self._obstacle_items.append((body, shape, obs))

            elif obs.kind == ObstacleType.BARRIER:
                body = pymunk.Body(body_type=pymunk.Body.STATIC)
                body.position = (obs.position.x, obs.position.y)
                body.angle = math.radians(obs.angle_deg)
                shape = pymunk.Poly.create_box(body, (obs.width, obs.height), radius=1.0)
                shape.elasticity = 0.20
                shape.friction = 0.40
                shape.filter = pymunk.ShapeFilter(categories=WALL_CATEGORY)
                shape.obstacle_tag = "barrier"
                self.space.add(body, shape)
                self._obstacle_items.append((body, shape, obs))

            elif obs.kind in (ObstacleType.POTHOLE, ObstacleType.OIL_SLICK):
                # Sensor circular sin rebote rígido pero detectable por segment_query (LIDAR)
                hazard_shape = pymunk.Circle(
                    self.space.static_body,
                    obs.radius,
                    obs.position.to_tuple(),
                )
                hazard_shape.sensor = True
                hazard_shape.filter = pymunk.ShapeFilter(categories=HAZARD_SENSOR_CATEGORY)
                hazard_shape.obstacle_tag = obs.kind.value
                self.space.add(hazard_shape)
                self._hazard_sensor_shapes.append(hazard_shape)

    def register_car(self, car: Car) -> None:
        """Crea y registra el PymunkCarBody asociado a la entidad Car."""
        if car.car_id in self._car_bodies:
            old_cb = self._car_bodies.pop(car.car_id)
            self.space.remove(old_cb.body, old_cb.shape)
        car_body = PymunkCarBody(car)
        car_body.shape.obstacle_tag = "car"
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

    def _update_car_collisions(self, car_body: PymunkCarBody) -> None:
        """Registra en car.active_collisions todas las etiquetas de objetos que tocan el vehículo."""
        tags: set[str] = set()

        def _collect_arbiter(arbiter: pymunk.Arbiter) -> None:
            for sh in arbiter.shapes:
                if sh is not car_body.shape:
                    tag = getattr(sh, "obstacle_tag", None)
                    if tag:
                        tags.add(tag)

        car_body.body.each_arbiter(_collect_arbiter)

        for query_info in self.space.shape_query(car_body.shape):
            if query_info.shape is not None and query_info.shape is not car_body.shape:
                tag = getattr(query_info.shape, "obstacle_tag", None)
                if tag:
                    tags.add(tag)

        car_body.car.active_collisions = tags

    def step(self, dt: float) -> None:
        """Aplica dinámicas vehiculares, fricción en obstáculos, avanza pymunk.Space y sincroniza."""
        for car_body in self._car_bodies.values():
            surface = None
            if self._track is not None:
                surface = self._track.get_surface_at(car_body.car.position)
            car_body.apply_controls_and_friction(dt, surface)

        for body, _, _ in self._obstacle_items:
            if body.body_type == pymunk.Body.DYNAMIC:
                body.velocity *= max(0.0, 1.0 - 2.8 * dt)
                body.angular_velocity *= max(0.0, 1.0 - 4.5 * dt)

        self.space.step(dt)
        self.sync_entities()

        for car_body in self._car_bodies.values():
            self._update_car_collisions(car_body)

    def sync_entities(self) -> None:
        """Sincroniza el estado físico de todos los cuerpos con sus entidades de dominio."""
        for car_body in self._car_bodies.values():
            car_body.sync_to_domain()
