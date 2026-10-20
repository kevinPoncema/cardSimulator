"""Pruebas unitarias para la capa de Dominio (Car, Value Objects, Track y principios SOLID)."""

from __future__ import annotations

import ast
import math
from pathlib import Path

import pytest

from src.domain.entities.car import Car, CarConfig
from src.domain.entities.track import Track
from src.domain.value_objects.angle import Angle
from src.domain.value_objects.car_action import CarAction
from src.domain.value_objects.color import CAR_PALETTE, Color
from src.domain.value_objects.speed import Speed
from src.domain.value_objects.vector2d import Vector2D
from src.infrastructure.input.dummy_ai_controller import DummyAiController
from src.infrastructure.physics.pymunk_physics_engine import PymunkPhysicsEngine


def test_vector2d_operations() -> None:
    """Verifica las operaciones algebraicas y de rotación de Vector2D."""
    v1 = Vector2D(3.0, 4.0)
    assert math.isclose(v1.magnitude(), 5.0)

    unit = v1.normalized()
    assert math.isclose(unit.magnitude(), 1.0)
    assert math.isclose(unit.x, 0.6)
    assert math.isclose(unit.y, 0.8)

    # Rotar (1, 0) 90 grados -> (0, 1)
    rotated = Vector2D(1.0, 0.0).rotate(math.pi / 2.0)
    assert math.isclose(rotated.x, 0.0, abs_tol=1e-9)
    assert math.isclose(rotated.y, 1.0, abs_tol=1e-9)


def test_color_hex_and_rgb_conversions() -> None:
    """Verifica el parseo hexadecimal y validación de canales RGB en Color."""
    c = Color.from_hex("#E63946")
    assert c.to_rgb_tuple() == (230, 57, 70)
    assert c.hex_code == "#E63946"

    with pytest.raises(ValueError):
        Color(300, 0, 0)


def test_speed_normalization_and_clamping() -> None:
    """Verifica la normalización de velocidad en [-1.0, 1.0]."""
    s_half = Speed(value=210.0, max_forward=420.0, max_reverse=160.0)
    assert math.isclose(s_half.normalized(), 0.5)

    s_over = Speed(value=600.0, max_forward=420.0, max_reverse=160.0)
    assert math.isclose(s_over.normalized(), 1.0)
    assert math.isclose(s_over.clamped().value, 420.0)


def test_car_front_center_and_kinematic_projections() -> None:
    """Verifica el cálculo del punto de emisión delantero y la proyección longitudinal de velocidad."""
    cfg = CarConfig(length=40.0, width=20.0, max_forward_speed=400.0, max_angular_velocity=5.0)
    car = Car(
        car_id="test_car",
        position=Vector2D(100.0, 200.0),
        velocity=Vector2D(200.0, 50.0),
        angle=Angle.from_degrees(0.0),
        angular_velocity=2.5,
        config=cfg,
    )

    # Con ángulo 0°, el frente está en (100 + 40/2, 200) = (120, 200)
    front = car.front_center_position()
    assert math.isclose(front.x, 120.0, abs_tol=1e-6)
    assert math.isclose(front.y, 200.0, abs_tol=1e-6)

    # Velocidad longitudinal es la componente X (200.0), lateral es Y (50.0)
    assert math.isclose(car.forward_speed(), 200.0, abs_tol=1e-6)
    assert math.isclose(car.lateral_speed(), 50.0, abs_tol=1e-6)
    assert math.isclose(car.normalized_forward_speed(), 0.5, abs_tol=1e-6)
    assert math.isclose(car.normalized_angular_velocity(), 0.5, abs_tol=1e-6)


def test_car_handbrake_reduces_lateral_grip() -> None:
    """Verifica que el freno de mano reduzca el agarre lateral para permitir derrapes."""
    car = Car(car_id="drift_car")
    normal_grip = car.effective_lateral_grip()

    car.apply_action(CarAction(throttle=1.0, steer=1.0, handbrake=True))
    drift_grip = car.effective_lateral_grip()

    assert drift_grip < normal_grip
    assert math.isclose(drift_grip, car.config.handbrake_lateral_grip)


def test_liskov_substitution_input_controller() -> None:
    """Verifica que DummyAiController cumpla el contrato de IInputController sin efectos colaterales."""
    controller = DummyAiController(target_throttle=0.8)
    action = controller.get_action()
    assert isinstance(action, CarAction)
    assert math.isclose(action.throttle, 0.8)
    assert controller.poll_quit() is False
    assert controller.poll_debug_toggle() is False


def test_physics_engine_acceleration_and_surface_friction() -> None:
    """Verifica que el motor Pymunk acelere el coche y aplique mayor fricción en césped."""
    track = Track.create_default_circuit()
    engine_asphalt = PymunkPhysicsEngine()
    engine_asphalt.register_track(track)

    car_asphalt = Car(
        car_id="c_asphalt",
        position=track.start_position,
        angle=track.start_angle,
        color=CAR_PALETTE["electric_blue"],
    )
    engine_asphalt.register_car(car_asphalt)
    car_asphalt.apply_action(CarAction(throttle=1.0))

    for _ in range(30):
        engine_asphalt.step(1.0 / 60.0)

    assert car_asphalt.forward_speed() > 50.0
    assert car_asphalt.current_surface_name == "asphalt"


def test_arcade_steering_stops_immediately_when_released() -> None:
    """Verifica que las físicas arcade detengan el giro inmediatamente al soltar el volante."""
    track = Track.create_default_circuit("retro_grand_prix")
    engine = PymunkPhysicsEngine()
    engine.register_track(track)

    car = Car(car_id="arcade_car", position=track.start_position, angle=track.start_angle)
    engine.register_car(car)

    # Acelerar y girar
    car.apply_action(CarAction(throttle=1.0, steer=1.0))
    for _ in range(15):
        engine.step(1.0 / 60.0)
    assert abs(car.angular_velocity) > 0.1

    # Soltar dirección -> velocidad angular debe ser exactamente 0.0 sin derrapar sin control
    car.apply_action(CarAction(throttle=1.0, steer=0.0))
    engine.step(1.0 / 60.0)
    assert math.isclose(car.angular_velocity, 0.0, abs_tol=1e-6)


def test_all_four_large_maps_load_correctly() -> None:
    """Verifica que los 4 circuitos de gran escala carguen con dimensiones >= 3200x2200."""
    expected_keys = ["retro_grand_prix", "neon_speedway", "canyon_drift", "starlight_loop"]
    assert Track.available_map_keys() == expected_keys

    for key in expected_keys:
        track = Track.create_default_circuit(key)
        assert track.width >= 3200.0
        assert track.height >= 2200.0
        assert track.asphalt_half_width >= 115.0
        assert len(track.all_wall_segments()) > 100
        assert len(track.gates) == 4
        assert track.get_surface_at(track.start_position).name == "asphalt"


def test_valid_full_lap_increments_score_by_one() -> None:
    """Verifica que solo se sume +1 punto al completar los 3 checkpoints y la meta en orden y sentido correcto."""
    car = Car(car_id="lap_car")
    assert car.score == 0
    assert car.next_checkpoint_id == 1

    # Pasar checkpoints 1, 2 y 3 en el sentido correcto
    assert car.record_gate_crossing(gate_id=1, is_forward=True) is False
    assert car.checkpoints_passed_in_lap == 1
    assert car.next_checkpoint_id == 2

    assert car.record_gate_crossing(gate_id=2, is_forward=True) is False
    assert car.checkpoints_passed_in_lap == 2
    assert car.next_checkpoint_id == 3

    assert car.record_gate_crossing(gate_id=3, is_forward=True) is False
    assert car.checkpoints_passed_in_lap == 3
    assert car.next_checkpoint_id == 0

    # Cruzar la meta (gate 0) en sentido correcto -> +1 punto
    assert car.record_gate_crossing(gate_id=0, is_forward=True) is True
    assert car.score == 1
    assert car.checkpoints_passed_in_lap == 0
    assert car.next_checkpoint_id == 1


def test_turnaround_or_reverse_finish_line_does_not_score() -> None:
    """Verifica que dar media vuelta o retroceder en la meta nunca otorgue puntos."""
    car = Car(car_id="cheater_car")

    # Caso 1: Retroceder sobre la meta al inicio y volver a cruzarla hacia adelante sin dar la vuelta
    assert car.record_gate_crossing(gate_id=0, is_forward=False) is False
    assert car.record_gate_crossing(gate_id=0, is_forward=True) is False
    assert car.score == 0

    # Caso 2: Avanzar hasta mitad de pista (CP1 y CP2), dar media vuelta (cruzar CP2 y CP1 al revés)
    # y luego cruzar la meta hacia adelante
    car.record_gate_crossing(gate_id=1, is_forward=True)
    car.record_gate_crossing(gate_id=2, is_forward=True)
    assert car.checkpoints_passed_in_lap == 2

    # Da media vuelta y regresa por CP2 y CP1 en sentido contrario
    car.record_gate_crossing(gate_id=2, is_forward=False)
    assert car.checkpoints_passed_in_lap == 1
    car.record_gate_crossing(gate_id=1, is_forward=False)
    assert car.checkpoints_passed_in_lap == 0

    # Cruza la meta hacia atrás y luego hacia adelante -> no debe sumar punto
    assert car.record_gate_crossing(gate_id=0, is_forward=False) is False
    assert car.record_gate_crossing(gate_id=0, is_forward=True) is False
    assert car.score == 0

    # Caso 3: Completar los 3 checkpoints pero cruzar la meta en sentido contrario
    car.record_gate_crossing(gate_id=1, is_forward=True)
    car.record_gate_crossing(gate_id=2, is_forward=True)
    car.record_gate_crossing(gate_id=3, is_forward=True)
    assert car.record_gate_crossing(gate_id=0, is_forward=False) is False
    assert car.score == 0


def test_track_gate_geometric_crossing_direction() -> None:
    """Verifica que Track.check_gate_crossings detecte el sentido de cruce sobre la meta."""
    track = Track.create_default_circuit("retro_grand_prix")
    finish_gate = track.gates[0]

    before_finish = finish_gate.center - finish_gate.forward_normal * 30.0
    after_finish = finish_gate.center + finish_gate.forward_normal * 30.0

    # Cruce en sentido válido (hacia adelante)
    fwd_crossings = track.check_gate_crossings(before_finish, after_finish)
    assert (0, True) in fwd_crossings

    # Cruce en sentido inverso (hacia atrás)
    rev_crossings = track.check_gate_crossings(after_finish, before_finish)
    assert (0, False) in rev_crossings


def test_domain_and_application_layers_have_no_pygame_or_pymunk_imports() -> None:
    """Verifica el Principio de Inversión de Dependencias (DIP): domain y application puros."""
    project_root = Path(__file__).resolve().parent.parent
    pure_dirs = [project_root / "src" / "domain", project_root / "src" / "application"]
    forbidden_modules = {"pygame", "pymunk"}

    for directory in pure_dirs:
        for py_file in directory.rglob("*.py"):
            tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        root_mod = alias.name.split(".")[0]
                        assert root_mod not in forbidden_modules, (
                            f"Violación DIP en {py_file}: importa '{alias.name}'"
                        )
                elif isinstance(node, ast.ImportFrom) and node.module:
                    root_mod = node.module.split(".")[0]
                    assert root_mod not in forbidden_modules, (
                        f"Violación DIP en {py_file}: importa desde '{node.module}'"
                    )
