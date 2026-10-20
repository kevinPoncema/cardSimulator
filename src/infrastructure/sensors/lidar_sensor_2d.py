"""Implementación del sensor LIDAR 2D utilizando consultas espaciales segment_query de Pymunk."""

from __future__ import annotations

import pymunk

from src.domain.entities.car import Car
from src.domain.entities.sensor_reading import (
    DEFAULT_LIDAR_MAX_RANGE,
    LIDAR_RAY_SPECS,
    SensorRay,
    SensorReading,
    local_ray_to_global_segment,
)
from src.domain.interfaces.sensor import ISensor
from src.domain.value_objects.vector2d import Vector2D
from src.infrastructure.physics.pymunk_physics_engine import PymunkPhysicsEngine


class LidarSensor2D(ISensor):
    """Sensor LIDAR 2D que emite 5 rayos desde el centro delantero del chasis."""

    def __init__(
        self,
        physics_engine: PymunkPhysicsEngine,
        max_range: float = DEFAULT_LIDAR_MAX_RANGE,
        ray_specs: tuple[tuple[str, float], ...] = LIDAR_RAY_SPECS,
    ) -> None:
        if max_range <= 0:
            raise ValueError("max_range debe ser positivo.")
        self.physics_engine = physics_engine
        self.max_range = float(max_range)
        self.ray_specs = ray_specs

    def scan(self, car: Car) -> SensorReading:
        """Lanza los rayos de sondeo mediante segment_query y calcula distancias de impacto."""
        origin = car.front_center_position()
        own_car_body = self.physics_engine.get_car_body(car.car_id)
        own_pymunk_body = own_car_body.body if own_car_body is not None else None

        rays_result: list[SensorRay] = []
        shape_filter = pymunk.ShapeFilter()

        for ray_id, rel_angle_deg in self.ray_specs:
            start_pt, max_end_pt, _ = local_ray_to_global_segment(
                origin=origin,
                car_angle=car.angle,
                relative_angle_deg=rel_angle_deg,
                max_range=self.max_range,
            )

            query_hits = self.physics_engine.space.segment_query(
                start_pt.to_tuple(),
                max_end_pt.to_tuple(),
                0.0,
                shape_filter,
            )

            # Filtrar el propio cuerpo del coche y seleccionar el impacto más cercano (menor alpha)
            closest_hit: pymunk.SegmentQueryInfo | None = None
            for info in query_hits:
                if info.shape is not None and info.shape.body is own_pymunk_body:
                    continue
                if closest_hit is None or info.alpha < closest_hit.alpha:
                    closest_hit = info

            if closest_hit is not None:
                hit_point = Vector2D(float(closest_hit.point.x), float(closest_hit.point.y))
                distance = start_pt.distance_to(hit_point)
                distance = max(0.0, min(self.max_range, distance))
                rays_result.append(
                    SensorRay(
                        ray_id=ray_id,
                        relative_angle_deg=rel_angle_deg,
                        origin=start_pt,
                        end_point=hit_point,
                        distance=distance,
                        max_range=self.max_range,
                        hit=True,
                    )
                )
            else:
                rays_result.append(
                    SensorRay(
                        ray_id=ray_id,
                        relative_angle_deg=rel_angle_deg,
                        origin=start_pt,
                        end_point=max_end_pt,
                        distance=self.max_range,
                        max_range=self.max_range,
                        hit=False,
                    )
                )

        return SensorReading(
            car_id=car.car_id,
            rays=tuple(rays_result),
            max_range=self.max_range,
        )
