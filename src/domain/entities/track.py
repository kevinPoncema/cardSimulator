"""Entidad de dominio Track que modela circuitos 2D de gran escala, checkpoints y barreras."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

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
    """Puerta de control (checkpoint o línea de meta) transversal a la pista.

    Atributos:
        gate_id: Índice secuencial (0 = Línea de Meta, 1..3 = Checkpoints intermedios).
        outer_point: Extremo exterior de la puerta (hasta el muro exterior).
        inner_point: Extremo interior de la puerta (hasta el muro interior).
        center: Punto central sobre el centerline.
        forward_normal: Vector unitario que apunta en el sentido válido de carrera.
    """

    gate_id: int
    outer_point: Vector2D
    inner_point: Vector2D
    center: Vector2D
    forward_normal: Vector2D


@dataclass(frozen=True)
class SurfaceProperties:
    """Propiedades físicas arcade de una zona de superficie del circuito.

    Atributos:
        name: Nombre identificador ('asphalt', 'kerb', 'grass').
        traction_multiplier: Multiplicador de fuerza motriz [0.0, 1.0].
        drag_multiplier: Multiplicador de resistencia longitudinal (>= 1.0).
        lateral_grip_multiplier: Multiplicador de agarre lateral.
    """

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

    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and (
        (d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)
    ):
        return True
    return False


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


# Definiciones preconfiguradas de mapas de gran tamaño
PRESET_MAPS: dict[str, dict] = {
    "retro_grand_prix": {
        "name": "1. Retro Grand Prix (Clásico Amplio)",
        "description": "Circuito equilibrado de gran escala con rectas largas y curvas suaves.",
        "width": 3200.0,
        "height": 2200.0,
        "asphalt_half_width": 115.0,
        "kerb_width": 18.0,
        "wall_half_width": 165.0,
        "subdivisions_per_segment": 12,
        "start_index": 12,
        "control_points": [
            [950.0, 1800.0],
            [1600.0, 1800.0],
            [2350.0, 1780.0],
            [2800.0, 1480.0],
            [2780.0, 820.0],
            [2400.0, 420.0],
            [1820.0, 460.0],
            [1480.0, 720.0],
            [1050.0, 430.0],
            [480.0, 560.0],
            [420.0, 1200.0],
            [580.0, 1680.0],
        ],
    },
    "neon_speedway": {
        "name": "2. Super Oval & Chicane Speedway (Alta Velocidad)",
        "description": "Anillo de máxima velocidad con grandes curvones para ir a fondo y derrapar.",
        "width": 3400.0,
        "height": 2200.0,
        "asphalt_half_width": 130.0,
        "kerb_width": 20.0,
        "wall_half_width": 185.0,
        "subdivisions_per_segment": 12,
        "start_index": 12,
        "control_points": [
            [900.0, 1780.0],
            [1700.0, 1800.0],
            [2550.0, 1760.0],
            [3000.0, 1400.0],
            [3000.0, 800.0],
            [2550.0, 420.0],
            [1900.0, 400.0],
            [1650.0, 580.0],
            [1350.0, 400.0],
            [750.0, 440.0],
            [400.0, 850.0],
            [420.0, 1450.0],
        ],
    },
    "canyon_drift": {
        "name": "3. Canyon Drift Circuit (Curvas Enlazadas)",
        "description": "Circuito técnico con curvas en S amplias ideal para usar el freno de mano (Espacio).",
        "width": 3400.0,
        "height": 2400.0,
        "asphalt_half_width": 115.0,
        "kerb_width": 18.0,
        "wall_half_width": 168.0,
        "subdivisions_per_segment": 12,
        "start_index": 12,
        "control_points": [
            [850.0, 1960.0],
            [1650.0, 1980.0],
            [2550.0, 1920.0],
            [2980.0, 1520.0],
            [2600.0, 1080.0],
            [2950.0, 620.0],
            [2400.0, 380.0],
            [1750.0, 680.0],
            [1250.0, 380.0],
            [520.0, 520.0],
            [440.0, 1120.0],
            [780.0, 1480.0],
            [450.0, 1820.0],
        ],
    },
    "starlight_loop": {
        "name": "4. Starlight Grand Loop (Trébol Panorámico)",
        "description": "Circuito panorámico con horquillas abiertas y rectas diagonales rápidas.",
        "width": 3200.0,
        "height": 2400.0,
        "asphalt_half_width": 120.0,
        "kerb_width": 18.0,
        "wall_half_width": 172.0,
        "subdivisions_per_segment": 12,
        "start_index": 12,
        "control_points": [
            [1000.0, 1950.0],
            [1650.0, 1950.0],
            [2400.0, 1950.0],
            [2820.0, 1600.0],
            [2500.0, 1150.0],
            [2820.0, 700.0],
            [2350.0, 400.0],
            [1600.0, 620.0],
            [850.0, 400.0],
            [420.0, 750.0],
            [720.0, 1220.0],
            [420.0, 1680.0],
        ],
    },
}


@dataclass
class Track:
    """Entidad de dominio que representa la pista 2D cerrada, sus zonas, checkpoints y muros."""

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
        """Detecta qué puertas fueron cruzadas entre prev_pos y curr_pos y en qué sentido.

        Returns:
            Lista de tuplas (gate_id, is_forward_direction).
        """
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
        """Determina la superficie ('asphalt', 'kerb' o 'grass') bajo el punto consultado."""
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
        # Colocar el coche ligeramente detrás de la línea de meta (start_idx - 2)
        # para que empiece justo antes de la línea o justo después sin cruzarla en falso.
        spawn_idx = (start_idx + 2) % len(smooth_centerline)
        start_pos = smooth_centerline[spawn_idx]
        next_pos = smooth_centerline[(spawn_idx + 1) % len(smooth_centerline)]
        start_dir = next_pos - start_pos
        start_angle = Angle(radians=math.atan2(start_dir.y, start_dir.x))
        return cls(
            name=str(data.get("name", "Custom Circuit")),
            description=str(data.get("description", "")),
            width=float(data.get("width", 3200.0)),
            height=float(data.get("height", 2200.0)),
            centerline=smooth_centerline,
            asphalt_half_width=float(data.get("asphalt_half_width", 115.0)),
            kerb_width=float(data.get("kerb_width", 18.0)),
            wall_half_width=float(data.get("wall_half_width", 165.0)),
            start_position=start_pos,
            start_angle=start_angle,
            start_index=start_idx,
        )

    @classmethod
    def create_default_circuit(cls, preset_key: str = "retro_grand_prix") -> Track:
        """Genera uno de los circuitos del catálogo predefinido."""
        key = preset_key if preset_key in PRESET_MAPS else "retro_grand_prix"
        return cls.from_dict(PRESET_MAPS[key])

    @classmethod
    def from_json_file(cls, map_path: str | Path) -> Track:
        """Carga la definición de un circuito desde un archivo JSON en assets/maps/."""
        path = Path(map_path)
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_dict(data)

    @classmethod
    def available_map_keys(cls) -> list[str]:
        """Retorna las claves ordenadas de los mapas disponibles."""
        return list(PRESET_MAPS.keys())
