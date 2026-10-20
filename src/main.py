"""Punto de entrada y Raíz de Composición (Composition Root) del Simulador de Carreras 2D."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Garantizar que el directorio raíz del proyecto esté en sys.path al ejecutar `python src/main.py`
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pygame

from src.application.race_orchestrator import RaceOrchestrator
from src.application.telemetry_collector import TelemetryCollector
from src.domain.entities.car import Car, CarConfig
from src.domain.entities.sensor_reading import DEFAULT_LIDAR_MAX_RANGE
from src.domain.entities.track import PRESET_MAPS, Track
from src.domain.interfaces.input_controller import IInputController
from src.domain.value_objects.color import CAR_PALETTE, Color
from src.infrastructure.input.dummy_ai_controller import DummyAiController
from src.infrastructure.input.human_keyboard_controller import HumanKeyboardController
from src.infrastructure.input.rl_agent_controller import RLAgentController
from src.infrastructure.physics.pymunk_physics_engine import PymunkPhysicsEngine
from src.infrastructure.rendering.pygame_renderer import PygameRenderer
from src.infrastructure.sensors.lidar_sensor_2d import LidarSensor2D

TARGET_FPS: int = 60
FIXED_DT: float = 1.0 / float(TARGET_FPS)
VIEWPORT_WIDTH: int = 1280
VIEWPORT_HEIGHT: int = 800


def load_all_tracks() -> dict[str, Track]:
    """Carga todos los circuitos disponibles en assets/maps/ o desde el catálogo predefinido."""
    maps_dir = PROJECT_ROOT / "assets" / "maps"
    tracks: dict[str, Track] = {}
    for map_key in PRESET_MAPS:
        json_file = maps_dir / f"{map_key}.json"
        if json_file.exists():
            tracks[map_key] = Track.from_json_file(json_file)
        else:
            tracks[map_key] = Track.create_default_circuit(map_key)
    return tracks


def build_orchestrator(
    map_key: str = "kart_classic",
    car_color: Color | None = None,
    controller_type: str = "human",
    model_path: str | Path | None = None,
    vs_ai: bool = False,
    num_cars: int = 1,
    lidar_max_range: float = DEFAULT_LIDAR_MAX_RANGE,
    headless: bool = False,
    debug_mode: bool = True,
) -> RaceOrchestrator:
    """Ensambla las dependencias de dominio, aplicación e infraestructura (DIP)."""
    available_tracks = load_all_tracks()
    chosen_map_key = map_key if map_key in available_tracks else "kart_classic"
    track = available_tracks[chosen_map_key]

    physics_engine = PymunkPhysicsEngine()
    lidar_sensor = LidarSensor2D(physics_engine=physics_engine, max_range=lidar_max_range)
    telemetry_collector = TelemetryCollector()
    renderer = PygameRenderer(
        width=VIEWPORT_WIDTH,
        height=VIEWPORT_HEIGHT,
        headless=headless,
    )

    # Exportar sprites de la paleta en assets/sprites si el directorio está vacío
    sprites_dir = PROJECT_ROOT / "assets" / "sprites"
    if not sprites_dir.exists() or not any(sprites_dir.glob("*.png")):
        renderer.sprite_manager.export_default_palette_sprites(sprites_dir)

    orchestrator = RaceOrchestrator(
        track=track,
        physics_engine=physics_engine,
        sensor=lidar_sensor,
        renderer=renderer,
        telemetry_collector=telemetry_collector,
        debug_mode=debug_mode,
        available_tracks=available_tracks,
        current_map_key=chosen_map_key,
    )

    default_model_file = (
        Path(model_path)
        if model_path is not None
        else (PROJECT_ROOT / "models" / "rl_agent_v1.npz")
    )

    total_cars = max(1, int(num_cars))
    if vs_ai and total_cars < 2:
        total_cars = 2

    palette_colors = list(CAR_PALETTE.values())
    chosen_primary_color = car_color or CAR_PALETTE["crimson_red"]

    for idx in range(total_cars):
        spawn_pos = orchestrator.compute_grid_spawn_position(idx)
        if idx == 0:
            c_color = chosen_primary_color
            c_id = "car_player_1"
        else:
            c_color = palette_colors[idx % len(palette_colors)]
            c_id = f"car_rival_{idx + 1}"

        car_entity = Car(
            car_id=c_id,
            position=spawn_pos,
            angle=track.start_angle,
            color=c_color,
            config=CarConfig(),
        )

        controller: IInputController
        if idx == 0:
            if controller_type == "rl_ai":
                controller = RLAgentController(
                    telemetry_provider=telemetry_collector,
                    car_id=car_entity.car_id,
                    model_path=default_model_file,
                    handle_window_events=True,
                )
            elif controller_type == "dummy_ai":
                controller = DummyAiController(
                    telemetry_provider=telemetry_collector,
                    car_id=car_entity.car_id,
                )
            else:
                controller = HumanKeyboardController()
        else:
            if controller_type == "dummy_ai":
                controller = DummyAiController(
                    telemetry_provider=telemetry_collector,
                    car_id=car_entity.car_id,
                )
            else:
                controller = RLAgentController(
                    telemetry_provider=telemetry_collector,
                    car_id=car_entity.car_id,
                    model_path=default_model_file,
                    handle_window_events=False,
                )

        orchestrator.add_participant(car_entity, controller)

    return orchestrator


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Analiza argumentos de línea de comandos para configurar la ejecución."""
    parser = argparse.ArgumentParser(
        description="Simulador de Carreras 2D con Telemetría LIDAR y Agente RL (Pymunk + Pygame-CE)"
    )
    parser.add_argument(
        "--map",
        choices=list(PRESET_MAPS.keys()),
        default="kart_classic",
        help="Circuito inicial (12 disponibles: kart_classic, oval_arena, megaring_international, etc.).",
    )
    parser.add_argument(
        "--color",
        type=str,
        default="crimson_red",
        help="Nombre de paleta (crimson_red, electric_blue, emerald_green, amber_gold, royal_purple) o código hex (#RRGGBB).",
    )
    parser.add_argument(
        "--controller",
        choices=["human", "rl_ai", "dummy_ai"],
        default="human",
        help="Controlador del vehículo principal: human, rl_ai (Red Neuronal entrenada) o dummy_ai.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=str(PROJECT_ROOT / "models" / "rl_agent_v1.npz"),
        help="Ruta al archivo .npz de la red neuronal entrenada.",
    )
    parser.add_argument(
        "--vs-ai",
        action="store_true",
        help="Añade un segundo coche rival controlado por la Red Neuronal RL entrenada.",
    )
    parser.add_argument(
        "--num-cars",
        type=int,
        default=1,
        help="Número total de coches simultáneos en la simulación (por defecto: 1).",
    )
    parser.add_argument(
        "--lidar-range",
        type=float,
        default=DEFAULT_LIDAR_MAX_RANGE,
        help="Alcance máximo de los rayos LIDAR en píxeles (por defecto: 300.0).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Número máximo de fotogramas a ejecutar (0 = ilimitado hasta cerrar).",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Ejecuta con ventana oculta (útil para verificación automatizada o entrenamiento IA).",
    )
    return parser.parse_args(argv)


def resolve_color(color_arg: str) -> Color:
    """Resuelve un argumento de color por nombre de paleta o string hexadecimal."""
    if color_arg in CAR_PALETTE:
        return CAR_PALETTE[color_arg]
    return Color.from_hex(color_arg)


def main(argv: list[str] | None = None) -> int:
    """Inicia el bucle principal de simulación a 60 FPS constantes."""
    args = parse_args(argv)
    car_color = resolve_color(args.color)

    orchestrator = build_orchestrator(
        map_key=args.map,
        car_color=car_color,
        controller_type=args.controller,
        model_path=args.model_path,
        vs_ai=args.vs_ai,
        num_cars=args.num_cars,
        lidar_max_range=args.lidar_range,
        headless=args.headless,
        debug_mode=True,
    )

    clock = pygame.time.Clock()
    orchestrator.is_running = True
    frame_count = 0

    try:
        while orchestrator.is_running:
            if not orchestrator.step_simulation(FIXED_DT):
                break

            current_fps = clock.get_fps() or float(TARGET_FPS)
            orchestrator.render_frame(fps=current_fps)
            clock.tick(TARGET_FPS)

            frame_count += 1
            if args.max_frames > 0 and frame_count >= args.max_frames:
                break
    finally:
        orchestrator.shutdown()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
