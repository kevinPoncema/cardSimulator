"""Entidad de dominio Track que modela circuitos 2D de gran escala, obstáculos y superficies."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from src.domain.entities.obstacle import Obstacle, ObstacleType
from src.domain.value_objects.angle import Angle
from src.domain.value_objects.vector2d import Vector2D


@dataclass(frozen=True)
class TrackSegment:
    """Segmento geométrico estático que conforma un muro o límite físico de la pista."""

    start: Vector2D
    end: Vector2D
    elasticity: float = 0.15
    friction: float = 0.15


@dataclass(frozen=True)
class TrackGate:
    """Puerta de control (checkpoint o línea de meta) transversal a la pista."""

    gate_id: int
    outer_point: Vector2D
    inner_point: Vector2D
    center: Vector2D
    forward_normal: Vector2D


@dataclass(frozen=True)
class SurfaceProperties:
    """Propiedades físicas arcade de una zona de superficie del circuito."""

    name: str
    traction_multiplier: float
    drag_multiplier: float
    lateral_grip_multiplier: float


SURFACE_CATALOG: dict[str, SurfaceProperties] = {
    "asphalt": SurfaceProperties(
        name="asphalt",
        traction_multiplier=1.0,
        drag_multiplier=1.0,
        lateral_grip_multiplier=1.0,
    ),
    "kerb": SurfaceProperties(
        name="kerb",
        traction_multiplier=0.92,
        drag_multiplier=1.35,
        lateral_grip_multiplier=0.95,
    ),
    "grass": SurfaceProperties(
        name="grass",
        traction_multiplier=0.65,
        drag_multiplier=2.8,
        lateral_grip_multiplier=0.80,
    ),
    "pothole": SurfaceProperties(
        name="pothole",
        traction_multiplier=0.35,
        drag_multiplier=5.2,
        lateral_grip_multiplier=0.70,
    ),
    "oil": SurfaceProperties(
        name="oil",
        traction_multiplier=0.50,
        drag_multiplier=0.85,
        lateral_grip_multiplier=0.12,
    ),
}


def _point_to_segment_distance(p: Vector2D, a: Vector2D, b: Vector2D) -> float:
    """Calcula la distancia euclidiana mínima desde el punto p al segmento ab."""
    ab = b - a
    len_sq = ab.x * ab.x + ab.y * ab.y
    if len_sq < 1e-12:
        return p.distance_to(a)
    t = ((p.x - a.x) * ab.x + (p.y - a.y) * ab.y) / len_sq
    t = max(0.0, min(1.0, t))
    projection = Vector2D(a.x + t * ab.x, a.y + t * ab.y)
    return p.distance_to(projection)


def segments_intersect(
    p1: Vector2D, p2: Vector2D, q1: Vector2D, q2: Vector2D
) -> bool:
    """Determina si el segmento de movimiento (p1, p2) intersecta la puerta (q1, q2)."""
    def ccw(a: Vector2D, b: Vector2D, c: Vector2D) -> float:
        return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)

    d1 = ccw(p1, p2, q1)
    d2 = ccw(p1, p2, q2)
    d3 = ccw(q1, q2, p1)
    d4 = ccw(q1, q2, p2)

    return ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and (
        (d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)
    )


def _catmull_rom_closed_spline(
    control_points: list[Vector2D], subdivisions_per_segment: int = 12
) -> list[Vector2D]:
    """Interpola una curva cerrada suave Catmull-Rom a partir de puntos de control."""
    n = len(control_points)
    if n < 3:
        raise ValueError("Se requieren al menos 3 puntos de control para un circuito cerrado.")
    smooth_points: list[Vector2D] = []
    for i in range(n):
        p0 = control_points[(i - 1) % n]
        p1 = control_points[i]
        p2 = control_points[(i + 1) % n]
        p3 = control_points[(i + 2) % n]
        for step in range(subdivisions_per_segment):
            t = step / float(subdivisions_per_segment)
            t2 = t * t
            t3 = t2 * t
            x = 0.5 * (
                (2.0 * p1.x)
                + (-p0.x + p2.x) * t
                + (2.0 * p0.x - 5.0 * p1.x + 4.0 * p2.x - p3.x) * t2
                + (-p0.x + 3.0 * p1.x - 3.0 * p2.x + p3.x) * t3
            )
            y = 0.5 * (
                (2.0 * p1.y)
                + (-p0.y + p2.y) * t
                + (2.0 * p0.y - 5.0 * p1.y + 4.0 * p2.y - p3.y) * t2
                + (-p0.y + 3.0 * p1.y - 3.0 * p2.y + p3.y) * t3
            )
            smooth_points.append(Vector2D(x, y))
    return smooth_points


def _compute_offset_rings(
    centerline: list[Vector2D], offset: float
) -> tuple[list[Vector2D], list[Vector2D]]:
    """Genera los anillos exterior (+offset) e interior (-offset) respecto a la línea central."""
    n = len(centerline)
    outer_ring: list[Vector2D] = []
    inner_ring: list[Vector2D] = []
    for i in range(n):
        prev_pt = centerline[(i - 1) % n]
        next_pt = centerline[(i + 1) % n]
        tangent = (next_pt - prev_pt).normalized()
        normal = Vector2D(-tangent.y, tangent.x)
        outer_ring.append(centerline[i] + normal * offset)
        inner_ring.append(centerline[i] - normal * offset)
    return outer_ring, inner_ring


def _ring_to_segments(
    ring: list[Vector2D], elasticity: float = 0.15, friction: float = 0.15
) -> list[TrackSegment]:
    """Convierte un anillo cerrado de vértices en una lista de segmentos de colisión."""
    n = len(ring)
    return [
        TrackSegment(
            start=ring[i],
            end=ring[(i + 1) % n],
            elasticity=elasticity,
            friction=friction,
        )
        for i in range(n)
    ]


# Catálogo completo de 12 Circuitos (3 Pequeños, 6 Medianos, 3 Grandes)
PRESET_MAPS: dict[str, dict] = {
    # -------------------------------------------------------------
    # 3 CIRCUITOS PEQUEÑOS (1600x1100 a 1800x1300 px)
    # -------------------------------------------------------------
    "kart_classic": {
        "name": "1. Mini Karting Classic (Pequeño)",
        "description": "Circuito ratonero y compacto con curvas cerradas, baches y conos.",
        "category": "small",
        "width": 1600.0,
        "height": 1200.0,
        "asphalt_half_width": 90.0,
        "kerb_width": 15.0,
        "wall_half_width": 135.0,
        "subdivisions_per_segment": 10,
        "start_index": 8,
        "control_points": [
            [500.0, 950.0],
            [900.0, 950.0],
            [1300.0, 850.0],
            [1320.0, 450.0],
            [950.0, 300.0],
            [680.0, 520.0],
            [450.0, 320.0],
            [280.0, 600.0],
        ],
        "obstacles": [
            {"id": "k_cone_1", "kind": "cone", "x": 680.0, "y": 480.0, "radius": 12.0},
            {"id": "k_cone_2", "kind": "cone", "x": 700.0, "y": 550.0, "radius": 12.0},
            {"id": "k_pothole_1", "kind": "pothole", "x": 1300.0, "y": 650.0, "radius": 36.0},
            {"id": "k_oil_1", "kind": "oil_slick", "x": 950.0, "y": 300.0, "radius": 32.0},
        ],
    },
    "oval_arena": {
        "name": "2. Thunder Oval Arena (Pequeño)",
        "description": "Óvalo de velocidad pura y compacta con baches traicioneros en el interior.",
        "category": "small",
        "width": 1600.0,
        "height": 1100.0,
        "asphalt_half_width": 100.0,
        "kerb_width": 16.0,
        "wall_half_width": 145.0,
        "subdivisions_per_segment": 10,
        "start_index": 8,
        "control_points": [
            [550.0, 880.0],
            [1050.0, 880.0],
            [1380.0, 720.0],
            [1380.0, 380.0],
            [1050.0, 220.0],
            [550.0, 220.0],
            [220.0, 380.0],
            [220.0, 720.0],
        ],
        "obstacles": [
            {"id": "ov_pothole_1", "kind": "pothole", "x": 800.0, "y": 860.0, "radius": 40.0},
            {"id": "ov_oil_1", "kind": "oil_slick", "x": 1350.0, "y": 550.0, "radius": 35.0},
            {"id": "ov_cone_1", "kind": "cone", "x": 1050.0, "y": 240.0, "radius": 12.0},
            {"id": "ov_cone_2", "kind": "cone", "x": 1070.0, "y": 200.0, "radius": 12.0},
        ],
    },
    "pocket_chicane": {
        "name": "3. Pocket Chicane Circuit (Pequeño)",
        "description": "Trazado técnico corto en 'S' con conos marcando el vértice de la chicane.",
        "category": "small",
        "width": 1800.0,
        "height": 1300.0,
        "asphalt_half_width": 95.0,
        "kerb_width": 15.0,
        "wall_half_width": 140.0,
        "subdivisions_per_segment": 10,
        "start_index": 8,
        "control_points": [
            [550.0, 1050.0],
            [1100.0, 1050.0],
            [1520.0, 880.0],
            [1350.0, 520.0],
            [900.0, 750.0],
            [680.0, 420.0],
            [1050.0, 250.0],
            [550.0, 250.0],
            [280.0, 650.0],
        ],
        "obstacles": [
            {"id": "pk_cone_1", "kind": "cone", "x": 900.0, "y": 720.0, "radius": 12.0},
            {"id": "pk_cone_2", "kind": "cone", "x": 900.0, "y": 780.0, "radius": 12.0},
            {"id": "pk_pothole_1", "kind": "pothole", "x": 1350.0, "y": 520.0, "radius": 38.0},
            {"id": "pk_oil_1", "kind": "oil_slick", "x": 680.0, "y": 420.0, "radius": 34.0},
        ],
    },

    # -------------------------------------------------------------
    # 6 CIRCUITOS MEDIANOS (2600x1800 a 2800x2000 px)
    # -------------------------------------------------------------
    "retro_grand_prix": {
        "name": "4. Retro Grand Prix Classic (Mediano)",
        "description": "El trazado equilibrado con curvas medias y baches en la frenada.",
        "category": "medium",
        "width": 2600.0,
        "height": 1800.0,
        "asphalt_half_width": 110.0,
        "kerb_width": 18.0,
        "wall_half_width": 160.0,
        "subdivisions_per_segment": 12,
        "start_index": 12,
        "control_points": [
            [800.0, 1500.0],
            [1400.0, 1500.0],
            [2050.0, 1480.0],
            [2350.0, 1200.0],
            [2300.0, 680.0],
            [1950.0, 350.0],
            [1500.0, 420.0],
            [1220.0, 680.0],
            [850.0, 360.0],
            [400.0, 480.0],
            [350.0, 1000.0],
            [480.0, 1400.0],
        ],
        "obstacles": [
            {"id": "rgp_pothole_1", "kind": "pothole", "x": 2350.0, "y": 1200.0, "radius": 42.0},
            {"id": "rgp_oil_1", "kind": "oil_slick", "x": 1220.0, "y": 680.0, "radius": 38.0},
            {"id": "rgp_cone_1", "kind": "cone", "x": 1500.0, "y": 390.0, "radius": 12.0},
            {"id": "rgp_cone_2", "kind": "cone", "x": 1500.0, "y": 450.0, "radius": 12.0},
            {"id": "rgp_barrier_1", "kind": "barrier", "x": 400.0, "y": 480.0, "width": 42.0, "height": 16.0, "angle_deg": 30.0},
        ],
    },
    "neon_speedway": {
        "name": "5. Neon Speedway (Mediano)",
        "description": "Anillo de alta velocidad con chicane central y conos delimitadores.",
        "category": "medium",
        "width": 2800.0,
        "height": 1800.0,
        "asphalt_half_width": 120.0,
        "kerb_width": 18.0,
        "wall_half_width": 170.0,
        "subdivisions_per_segment": 12,
        "start_index": 12,
        "control_points": [
            [750.0, 1480.0],
            [1400.0, 1500.0],
            [2150.0, 1460.0],
            [2500.0, 1150.0],
            [2500.0, 650.0],
            [2100.0, 350.0],
            [1550.0, 350.0],
            [1350.0, 520.0],
            [1100.0, 350.0],
            [600.0, 380.0],
            [320.0, 720.0],
            [350.0, 1200.0],
        ],
        "obstacles": [
            {"id": "ns_cone_1", "kind": "cone", "x": 1350.0, "y": 490.0, "radius": 12.0},
            {"id": "ns_cone_2", "kind": "cone", "x": 1350.0, "y": 550.0, "radius": 12.0},
            {"id": "ns_pothole_1", "kind": "pothole", "x": 2150.0, "y": 1460.0, "radius": 44.0},
            {"id": "ns_oil_1", "kind": "oil_slick", "x": 320.0, "y": 720.0, "radius": 40.0},
        ],
    },
    "canyon_drift": {
        "name": "6. Canyon Drift Circuit (Mediano)",
        "description": "Curvas en S para derrapar sobre asfalto, aceite y baches.",
        "category": "medium",
        "width": 2800.0,
        "height": 2000.0,
        "asphalt_half_width": 110.0,
        "kerb_width": 18.0,
        "wall_half_width": 160.0,
        "subdivisions_per_segment": 12,
        "start_index": 12,
        "control_points": [
            [700.0, 1650.0],
            [1350.0, 1680.0],
            [2100.0, 1620.0],
            [2500.0, 1280.0],
            [2180.0, 920.0],
            [2480.0, 520.0],
            [2000.0, 320.0],
            [1450.0, 600.0],
            [1050.0, 320.0],
            [420.0, 450.0],
            [380.0, 950.0],
            [650.0, 1250.0],
            [380.0, 1550.0],
        ],
        "obstacles": [
            {"id": "cd_oil_1", "kind": "oil_slick", "x": 2180.0, "y": 920.0, "radius": 42.0},
            {"id": "cd_oil_2", "kind": "oil_slick", "x": 1450.0, "y": 600.0, "radius": 38.0},
            {"id": "cd_pothole_1", "kind": "pothole", "x": 650.0, "y": 1250.0, "radius": 45.0},
            {"id": "cd_cone_1", "kind": "cone", "x": 2480.0, "y": 520.0, "radius": 12.0},
            {"id": "cd_barrier_1", "kind": "barrier", "x": 1050.0, "y": 320.0, "width": 45.0, "height": 18.0, "angle_deg": 15.0},
        ],
    },
    "starlight_loop": {
        "name": "7. Starlight Grand Loop (Mediano)",
        "description": "Trébol panorámico de curvas abiertas con zonas de asfalto roto.",
        "category": "medium",
        "width": 2600.0,
        "height": 2000.0,
        "asphalt_half_width": 115.0,
        "kerb_width": 18.0,
        "wall_half_width": 165.0,
        "subdivisions_per_segment": 12,
        "start_index": 12,
        "control_points": [
            [800.0, 1650.0],
            [1350.0, 1650.0],
            [2000.0, 1650.0],
            [2350.0, 1350.0],
            [2050.0, 980.0],
            [2350.0, 600.0],
            [1950.0, 320.0],
            [1300.0, 520.0],
            [700.0, 320.0],
            [350.0, 650.0],
            [600.0, 1050.0],
            [350.0, 1420.0],
        ],
        "obstacles": [
            {"id": "sl_pothole_1", "kind": "pothole", "x": 2050.0, "y": 980.0, "radius": 42.0},
            {"id": "sl_oil_1", "kind": "oil_slick", "x": 1300.0, "y": 520.0, "radius": 36.0},
            {"id": "sl_cone_1", "kind": "cone", "x": 600.0, "y": 1050.0, "radius": 12.0},
            {"id": "sl_barrier_1", "kind": "barrier", "x": 2350.0, "y": 600.0, "width": 40.0, "height": 16.0, "angle_deg": -25.0},
        ],
    },
    "desert_dunes": {
        "name": "8. Desert Dunes Ring (Mediano)",
        "description": "Trazado árido con curvas prolongadas y baches gigantes en la zona sur.",
        "category": "medium",
        "width": 2800.0,
        "height": 1900.0,
        "asphalt_half_width": 115.0,
        "kerb_width": 18.0,
        "wall_half_width": 165.0,
        "subdivisions_per_segment": 12,
        "start_index": 12,
        "control_points": [
            [750.0, 1550.0],
            [1400.0, 1580.0],
            [2100.0, 1520.0],
            [2480.0, 1200.0],
            [2450.0, 650.0],
            [2050.0, 350.0],
            [1550.0, 550.0],
            [1150.0, 350.0],
            [650.0, 400.0],
            [350.0, 780.0],
            [350.0, 1250.0],
        ],
        "obstacles": [
            {"id": "dd_pothole_1", "kind": "pothole", "x": 1400.0, "y": 1580.0, "radius": 46.0},
            {"id": "dd_pothole_2", "kind": "pothole", "x": 2480.0, "y": 1200.0, "radius": 44.0},
            {"id": "dd_cone_1", "kind": "cone", "x": 1550.0, "y": 550.0, "radius": 12.0},
            {"id": "dd_oil_1", "kind": "oil_slick", "x": 650.0, "y": 400.0, "radius": 36.0},
        ],
    },
    "dockyard_sprint": {
        "name": "9. Dockyard Industrial Sprint (Mediano)",
        "description": "Zonas industriales de 90° entre barreras de hormigón y manchas de aceite.",
        "category": "medium",
        "width": 2700.0,
        "height": 1900.0,
        "asphalt_half_width": 105.0,
        "kerb_width": 16.0,
        "wall_half_width": 150.0,
        "subdivisions_per_segment": 12,
        "start_index": 12,
        "control_points": [
            [700.0, 1550.0],
            [1350.0, 1550.0],
            [2150.0, 1550.0],
            [2380.0, 1150.0],
            [2050.0, 850.0],
            [2380.0, 450.0],
            [1650.0, 320.0],
            [1300.0, 700.0],
            [750.0, 350.0],
            [350.0, 680.0],
            [400.0, 1250.0],
        ],
        "obstacles": [
            {"id": "dy_barrier_1", "kind": "barrier", "x": 2050.0, "y": 850.0, "width": 45.0, "height": 18.0, "angle_deg": 45.0},
            {"id": "dy_oil_1", "kind": "oil_slick", "x": 1300.0, "y": 700.0, "radius": 42.0},
            {"id": "dy_pothole_1", "kind": "pothole", "x": 750.0, "y": 350.0, "radius": 40.0},
            {"id": "dy_cone_1", "kind": "cone", "x": 1350.0, "y": 1530.0, "radius": 12.0},
            {"id": "dy_cone_2", "kind": "cone", "x": 1350.0, "y": 1570.0, "radius": 12.0},
        ],
    },

    # -------------------------------------------------------------
    # 3 CIRCUITOS GRANDES (3500x2500 a 3800x2600 px)
    # -------------------------------------------------------------
    "megaring_international": {
        "name": "10. MegaRing International GP (Grande)",
        "description": "Circuito masivo estilo Grand Prix con múltiples sectores y chicanes técnicas.",
        "category": "large",
        "width": 3600.0,
        "height": 2500.0,
        "asphalt_half_width": 120.0,
        "kerb_width": 18.0,
        "wall_half_width": 170.0,
        "subdivisions_per_segment": 12,
        "start_index": 12,
        "control_points": [
            [1100.0, 2050.0],
            [1850.0, 2050.0],
            [2750.0, 2020.0],
            [3250.0, 1680.0],
            [3200.0, 950.0],
            [2750.0, 480.0],
            [2050.0, 520.0],
            [1700.0, 820.0],
            [1200.0, 480.0],
            [550.0, 620.0],
            [480.0, 1350.0],
            [680.0, 1900.0],
        ],
        "obstacles": [
            {"id": "mr_pothole_1", "kind": "pothole", "x": 3250.0, "y": 1680.0, "radius": 48.0},
            {"id": "mr_oil_1", "kind": "oil_slick", "x": 1700.0, "y": 820.0, "radius": 44.0},
            {"id": "mr_cone_1", "kind": "cone", "x": 2050.0, "y": 490.0, "radius": 12.0},
            {"id": "mr_cone_2", "kind": "cone", "x": 2050.0, "y": 550.0, "radius": 12.0},
            {"id": "mr_barrier_1", "kind": "barrier", "x": 550.0, "y": 620.0, "width": 50.0, "height": 20.0, "angle_deg": 40.0},
        ],
    },
    "volcano_ridge": {
        "name": "11. Volcano Ridge Expressway (Grande)",
        "description": "Circuito colosal de velocidad punta extrema con zonas de desprendimiento.",
        "category": "large",
        "width": 3800.0,
        "height": 2600.0,
        "asphalt_half_width": 125.0,
        "kerb_width": 20.0,
        "wall_half_width": 180.0,
        "subdivisions_per_segment": 12,
        "start_index": 12,
        "control_points": [
            [1050.0, 2150.0],
            [1950.0, 2180.0],
            [2950.0, 2120.0],
            [3450.0, 1680.0],
            [3420.0, 920.0],
            [2900.0, 480.0],
            [2150.0, 450.0],
            [1850.0, 720.0],
            [1500.0, 450.0],
            [850.0, 500.0],
            [480.0, 980.0],
            [500.0, 1680.0],
        ],
        "obstacles": [
            {"id": "vr_pothole_1", "kind": "pothole", "x": 1950.0, "y": 2180.0, "radius": 50.0},
            {"id": "vr_pothole_2", "kind": "pothole", "x": 3450.0, "y": 1680.0, "radius": 48.0},
            {"id": "vr_oil_1", "kind": "oil_slick", "x": 1850.0, "y": 720.0, "radius": 46.0},
            {"id": "vr_cone_1", "kind": "cone", "x": 2900.0, "y": 480.0, "radius": 12.0},
            {"id": "vr_cone_2", "kind": "cone", "x": 2920.0, "y": 520.0, "radius": 12.0},
            {"id": "vr_barrier_1", "kind": "barrier", "x": 850.0, "y": 500.0, "width": 52.0, "height": 20.0, "angle_deg": 10.0},
        ],
    },
    "monaco_night": {
        "name": "12. Monaco Night Grand Prix (Grande)",
        "description": "Trazado nocturno técnico y sinuoso con chicane estrecha, pianos dobles y aceite.",
        "category": "large",
        "width": 3500.0,
        "height": 2500.0,
        "asphalt_half_width": 115.0,
        "kerb_width": 18.0,
        "wall_half_width": 165.0,
        "subdivisions_per_segment": 12,
        "start_index": 12,
        "control_points": [
            [950.0, 2050.0],
            [1800.0, 2080.0],
            [2750.0, 2000.0],
            [3200.0, 1620.0],
            [2800.0, 1150.0],
            [3200.0, 680.0],
            [2600.0, 420.0],
            [1900.0, 750.0],
            [1350.0, 420.0],
            [580.0, 580.0],
            [500.0, 1200.0],
            [850.0, 1600.0],
            [500.0, 1950.0],
        ],
        "obstacles": [
            {"id": "mn_oil_1", "kind": "oil_slick", "x": 2800.0, "y": 1150.0, "radius": 45.0},
            {"id": "mn_oil_2", "kind": "oil_slick", "x": 1900.0, "y": 750.0, "radius": 42.0},
            {"id": "mn_pothole_1", "kind": "pothole", "x": 850.0, "y": 1600.0, "radius": 46.0},
            {"id": "mn_cone_1", "kind": "cone", "x": 2600.0, "y": 420.0, "radius": 12.0},
            {"id": "mn_cone_2", "kind": "cone", "x": 2620.0, "y": 460.0, "radius": 12.0},
            {"id": "mn_barrier_1", "kind": "barrier", "x": 1350.0, "y": 420.0, "width": 48.0, "height": 18.0, "angle_deg": -30.0},
        ],
    },
}


@dataclass
class Track:
    """Entidad de dominio que representa la pista 2D cerrada, sus zonas, checkpoints y obstáculos."""

    name: str
    width: float
    height: float
    centerline: list[Vector2D]
    asphalt_half_width: float = 115.0
    kerb_width: float = 18.0
    wall_half_width: float = 165.0
    start_position: Vector2D = field(default_factory=lambda: Vector2D(1600.0, 1800.0))
    start_angle: Angle = field(default_factory=lambda: Angle(0.0))
    start_index: int = 12
    description: str = ""
    category: str = "medium"
    obstacles: list[Obstacle] = field(default_factory=list)
    outer_walls: list[TrackSegment] = field(default_factory=list)
    inner_walls: list[TrackSegment] = field(default_factory=list)
    outer_asphalt_polygon: list[Vector2D] = field(default_factory=list)
    inner_asphalt_polygon: list[Vector2D] = field(default_factory=list)
    outer_kerb_polygon: list[Vector2D] = field(default_factory=list)
    inner_kerb_polygon: list[Vector2D] = field(default_factory=list)
    outer_wall_polygon: list[Vector2D] = field(default_factory=list)
    inner_wall_polygon: list[Vector2D] = field(default_factory=list)
    gates: list[TrackGate] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.outer_walls or not self.inner_walls:
            self._build_geometry()
        if not self.gates and len(self.outer_wall_polygon) >= 3:
            self._build_gates()

    def _build_geometry(self) -> None:
        """Construye los polígonos de asfalto, pianos y muros perimetrales a partir del centerline."""
        self.outer_asphalt_polygon, self.inner_asphalt_polygon = _compute_offset_rings(
            self.centerline, self.asphalt_half_width
        )
        self.outer_kerb_polygon, self.inner_kerb_polygon = _compute_offset_rings(
            self.centerline, self.asphalt_half_width + self.kerb_width
        )
        self.outer_wall_polygon, self.inner_wall_polygon = _compute_offset_rings(
            self.centerline, self.wall_half_width
        )
        self.outer_walls = _ring_to_segments(self.outer_wall_polygon)
        self.inner_walls = _ring_to_segments(self.inner_wall_polygon)

    def _build_gates(self, num_gates: int = 4) -> None:
        """Genera 4 puertas transversales equidistantes (0 = Meta, 1..3 = Checkpoints de vuelta)."""
        n = len(self.centerline)
        if n < num_gates:
            return
        self.gates = []
        for g in range(num_gates):
            idx = (self.start_index + (g * n) // num_gates) % n
            prev_pt = self.centerline[(idx - 1) % n]
            next_pt = self.centerline[(idx + 1) % n]
            forward_normal = (next_pt - prev_pt).normalized()
            self.gates.append(
                TrackGate(
                    gate_id=g,
                    outer_point=self.outer_wall_polygon[idx],
                    inner_point=self.inner_wall_polygon[idx],
                    center=self.centerline[idx],
                    forward_normal=forward_normal,
                )
            )

    def check_gate_crossings(
        self, prev_pos: Vector2D, curr_pos: Vector2D
    ) -> list[tuple[int, bool]]:
        """Detecta qué puertas fueron cruzadas entre prev_pos y curr_pos y en qué sentido."""
        movement = curr_pos - prev_pos
        if movement.magnitude() < 1e-6:
            return []

        crossings: list[tuple[int, bool]] = []
        for gate in self.gates:
            if segments_intersect(prev_pos, curr_pos, gate.outer_point, gate.inner_point):
                is_forward = movement.dot(gate.forward_normal) > 0.0
                crossings.append((gate.gate_id, is_forward))
        return crossings

    def closest_centerline_tangent(self, point: Vector2D) -> Vector2D:
        """Retorna el vector unitario tangente del circuito en el punto más cercano al coche."""
        n = len(self.centerline)
        min_dist = float("inf")
        best_idx = 0
        for i in range(n):
            d = _point_to_segment_distance(point, self.centerline[i], self.centerline[(i + 1) % n])
            if d < min_dist:
                min_dist = d
                best_idx = i
        prev_pt = self.centerline[(best_idx - 1) % n]
        next_pt = self.centerline[(best_idx + 1) % n]
        return (next_pt - prev_pt).normalized()

    def all_wall_segments(self) -> list[TrackSegment]:
        """Retorna todos los segmentos físicos de barrera (exteriores e interiores)."""
        return self.outer_walls + self.inner_walls

    def distance_to_centerline(self, point: Vector2D) -> float:
        """Calcula la distancia mínima desde un punto a la polilínea central de la pista."""
        n = len(self.centerline)
        min_dist = float("inf")
        for i in range(n):
            d = _point_to_segment_distance(point, self.centerline[i], self.centerline[(i + 1) % n])
            if d < min_dist:
                min_dist = d
        return min_dist

    def get_surface_at(self, point: Vector2D) -> SurfaceProperties:
        """Determina la superficie activa bajo el coche considerando baches y manchas de aceite."""
        # 1. Comprobar si está sobre un obstáculo de terreno (bache o aceite)
        for obs in self.obstacles:
            if obs.is_hazard_surface:
                if point.distance_to(obs.position) <= obs.radius:
                    if obs.kind == ObstacleType.POTHOLE:
                        return SURFACE_CATALOG["pothole"]
                    elif obs.kind == ObstacleType.OIL_SLICK:
                        return SURFACE_CATALOG["oil"]

        # 2. Comprobar superficie estándar de la pista (asfalto, kerb, grass)
        dist = self.distance_to_centerline(point)
        if dist <= self.asphalt_half_width:
            return SURFACE_CATALOG["asphalt"]
        if dist <= self.asphalt_half_width + self.kerb_width:
            return SURFACE_CATALOG["kerb"]
        return SURFACE_CATALOG["grass"]

    @classmethod
    def from_dict(cls, data: dict) -> Track:
        """Construye una instancia Track desde un diccionario de configuración."""
        control_points = [Vector2D(pt[0], pt[1]) for pt in data["control_points"]]
        subdivisions = int(data.get("subdivisions_per_segment", 12))
        smooth_centerline = _catmull_rom_closed_spline(
            control_points, subdivisions_per_segment=subdivisions
        )
        start_idx = int(data.get("start_index", 12)) % len(smooth_centerline)
        spawn_idx = (start_idx + 2) % len(smooth_centerline)
        start_pos = smooth_centerline[spawn_idx]
        next_pos = smooth_centerline[(spawn_idx + 1) % len(smooth_centerline)]
        start_dir = next_pos - start_pos
        start_angle = Angle(radians=math.atan2(start_dir.y, start_dir.x))

        obstacles: list[Obstacle] = []
        for raw_obs in data.get("obstacles", []):
            obstacles.append(
                Obstacle(
                    obstacle_id=raw_obs["id"],
                    kind=ObstacleType(raw_obs["kind"]),
                    position=Vector2D(float(raw_obs["x"]), float(raw_obs["y"])),
                    radius=float(raw_obs.get("radius", 14.0)),
                    width=float(raw_obs.get("width", 36.0)),
                    height=float(raw_obs.get("height", 14.0)),
                    angle_deg=float(raw_obs.get("angle_deg", 0.0)),
                )
            )

        return cls(
            name=str(data.get("name", "Custom Circuit")),
            description=str(data.get("description", "")),
            category=str(data.get("category", "medium")),
            width=float(data.get("width", 3200.0)),
            height=float(data.get("height", 2200.0)),
            centerline=smooth_centerline,
            asphalt_half_width=float(data.get("asphalt_half_width", 115.0)),
            kerb_width=float(data.get("kerb_width", 18.0)),
            wall_half_width=float(data.get("wall_half_width", 165.0)),
            start_position=start_pos,
            start_angle=start_angle,
            start_index=start_idx,
            obstacles=obstacles,
        )

    @classmethod
    def create_default_circuit(cls, preset_key: str = "kart_classic") -> Track:
        """Genera uno de los circuitos del catálogo predefinido."""
        key = preset_key if preset_key in PRESET_MAPS else "kart_classic"
        return cls.from_dict(PRESET_MAPS[key])

    @classmethod
    def from_json_file(cls, map_path: str | Path) -> Track:
        """Carga la definición de un circuito desde un archivo JSON en assets/maps/."""
        path = Path(map_path)
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_dict(data)

    @classmethod
    def available_map_keys(cls) -> list[str]:
        """Retorna las claves ordenadas de los 12 mapas disponibles."""
        return list(PRESET_MAPS.keys())
