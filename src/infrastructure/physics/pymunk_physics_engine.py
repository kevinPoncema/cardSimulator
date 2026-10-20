"""Motor de físicas 2D basado en Pymunk que implementa IPhysicsEngine."""

from __future__ import annotations

import pymunk

from src.domain.entities.car import Car
from src.domain.entities.track import Track
from src.domain.interfaces.physics_engine import IPhysicsEngine
from src.domain.value_objects.angle import Angle
from src.domain.value_objects.vector2d import Vector2D
from src.infrastructure.physics.pymunk_car_body import WALL_CATEGORY, PymunkCarBody


class PymunkPhysicsEngine(IPhysicsEngine):
    """Implementación de IPhysicsEngine sobre pymunk.Space (vista cenital sin gravedad)."""

    def __init__(self) -> None:
        self.space = pymunk.Space()
        self.space.gravity = (0.0, 0.0)
        self.space.damping = 0.998
        self._car_bodies: dict[str, PymunkCarBody] = {}
        self._wall_shapes: list[pymunk.Segment] = []
        self._track: Track | None = None

    def register_track(self, track: Track) -> None:
        """Registra los muros estáticos perimetrales e interiores de la pista en Pymunk."""
        self._track = track
        for old_shape in self._wall_shapes:
            self.space.remove(old_shape)
        self._wall_shapes.clear()

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

    def get_car_body(self, car_id: str) -> PymunkCarBody | None:
        """Retorna el adaptador PymunkCarBody para consultas internas (ej. filtro LIDAR)."""
        return self._car_bodies.get(car_id)

    def step(self, dt: float) -> None:
        """Aplica dinámicas vehiculares, avanza pymunk.Space y sincroniza las entidades."""
        for car_body in self._car_bodies.values():
            surface = None
            if self._track is not None:
                surface = self._track.get_surface_at(car_body.car.position)
            car_body.apply_controls_and_friction(dt, surface)

        self.space.step(dt)
        self.sync_entities()

    def sync_entities(self) -> None:
        """Sincroniza el estado físico de todos los cuerpos con sus entidades de dominio."""
        for car_body in self._car_bodies.values():
            car_body.sync_to_domain()
