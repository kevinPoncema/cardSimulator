"""Configuración parametrizable y calculadora de recompensas para Aprendizaje por Refuerzo."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from src.domain.entities.car import Car
from src.domain.entities.sensor_reading import SensorReading
from src.domain.entities.track import Track


@dataclass
class RewardConfig:
    """Parámetros de recompensa y penalización fácilmente ajustables (vía código o JSON).

    Incluye recompensas por avance, curvas, monedas, checkpoints, vueltas completas
    y maniobras de desatasco en reversa, así como fuertes penalizaciones por quedarse
    estancado contra un muro, chocar con obstáculos o ir en sentido contrario.
    """

    # Recompensas continuas por conducción, curvas y esquiva
    forward_progress_weight: float = 1.6
    curve_alignment_weight: float = 0.85
    asphalt_center_bonus: float = 0.35
    obstacle_clearance_bonus: float = 0.50

    # Grandes premios por monedas, checkpoints y vueltas completas
    coin_collected_reward: float = 180.0
    coin_proximity_weight: float = 2.5
    checkpoint_reward: float = 65.0
    lap_completion_reward: float = 250.0

    # Castigos por colisión contra obstáculos físicos y muros
    cone_collision_penalty: float = -40.0
    barrier_collision_penalty: float = -75.0
    wall_collision_penalty: float = -55.0

    # Castigo severo por quedarse estancado contra la pared y premio por retroceder y corregir
    wall_stuck_step_penalty: float = -14.0
    wall_stuck_escalation_factor: float = 2.2
    max_wall_stuck_steps: int = 45
    wall_stuck_timeout_penalty: float = -160.0
    reverse_escape_reward: float = 4.5

    # Castigos por pisar obstáculos de terreno y salirse de la calzada
    pothole_step_penalty: float = -4.0
    oil_slick_step_penalty: float = -4.5
    grass_step_penalty: float = -1.5
    kerb_step_penalty: float = -0.2

    # Castigos severos por sentido contrario y cruce inverso de meta/checkpoints
    wrong_way_step_penalty: float = -25.0
    wrong_way_escalation_factor: float = 1.5
    max_wrong_way_steps: int = 90
    wrong_way_timeout_penalty: float = -1000.0
    reverse_finish_line_penalty: float = -1500.0
    reverse_checkpoint_penalty: float = -400.0

    # Castigos por quedarse detenido / atascado
    idle_speed_threshold: float = 20.0
    idle_step_penalty: float = -1.2
    max_idle_steps: int = 90
    idle_timeout_penalty: float = -60.0

    # Reglas de terminación anticipada del episodio
    terminate_on_wall_crash: bool = False
    terminate_on_barrier_crash: bool = True
    terminate_on_reverse_finish_line: bool = True

    @classmethod
    def from_json_file(cls, path: str | Path) -> RewardConfig:
        """Carga la configuración de recompensas desde un archivo JSON."""
        file_path = Path(path)
        data = json.loads(file_path.read_text(encoding="utf-8"))
        return cls(**data)

    def to_json_file(self, path: str | Path) -> None:
        """Guarda la configuración actual en un archivo JSON."""
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")


class RewardCalculator:
    """Evalúa cada paso de simulación y calcula la recompensa escalar y su desglose."""

    def __init__(self, config: RewardConfig | None = None) -> None:
        self.config = config or RewardConfig()
        self.wrong_way_Dynamic_counter: int = 0
        self.idle_steps_counter: int = 0
        self.wall_stuck_counter: int = 0
        self._prev_d0: float = 1.0
        self._prev_cone_contact: bool = False
        self._prev_barrier_contact: bool = False
        self._prev_wall_contact: bool = False

    def reset(self) -> None:
        """Reinicia los contadores acumulativos al comenzar un nuevo episodio."""
        self.wrong_way_Dynamic_counter = 0
        self.idle_steps_counter = 0
        self.wall_stuck_counter = 0
        self._prev_d0 = 1.0
        self._prev_cone_contact = False
        self._prev_barrier_contact = False
        self._prev_wall_contact = False

    def compute_step_reward(
        self,
        car: Car,
        track: Track,
        reading: SensorReading,
        gate_crossings: list[tuple[int, bool]],
        lap_completed: bool,
        checkpoint_advanced: bool,
        coins_collected_count: int = 0,
    ) -> tuple[float, bool, dict[str, float | bool | str]]:
        """Calcula (reward, terminated, breakdown_info) para el paso actual."""
        cfg = self.config
        total_reward = 0.0
        terminated = False
        termination_reason = ""
        breakdown: dict[str, float | bool | str] = {}

        norm_dists = reading.get_normalized_distances()
        d_0 = norm_dists[0]
        min_front_clearance = min(norm_dists[0], norm_dists[1], norm_dists[2])

        # 1. Velocidad proyectada sobre el sentido válido de la curva (tangente del circuito)
        tangent = track.closest_centerline_tangent(car.position)
        track_progress_speed = car.velocity.dot(tangent)
        norm_progress = track_progress_speed / max(1.0, car.config.max_forward_speed)
        heading_alignment = car.forward_unit_vector().dot(tangent)
        v_fwd = car.forward_speed()

        if not car.is_wrong_way and norm_progress > 0.02 and d_0 > 0.14:
            r_progress = norm_progress * cfg.forward_progress_weight
            r_align = max(0.0, heading_alignment) * norm_progress * cfg.curve_alignment_weight
            total_reward += r_progress + r_align
            breakdown["forward_progress"] = round(r_progress, 4)
            breakdown["curve_alignment"] = round(r_align, 4)

        # 2. Bonificación por mantenerse en el asfalto y mantener despejado el LIDAR
        if car.current_surface_name == "asphalt" and norm_progress > 0.1 and d_0 > 0.18:
            r_asphalt = cfg.asphalt_center_bonus
            r_clearance = min_front_clearance * cfg.obstacle_clearance_bonus
            total_reward += r_asphalt + r_clearance
            breakdown["asphalt_and_clearance"] = round(r_asphalt + r_clearance, 4)

        # 3. Premio masivo por recolectar Monedas en la pista y bonificación por aproximación
        if coins_collected_count > 0:
            r_coins = float(coins_collected_count) * cfg.coin_collected_reward
            total_reward += r_coins
            breakdown["coin_collected"] = r_coins

        # Guía continua hacia monedas alcanzables activas por delante del vehículo
        if track.coins and norm_progress > 0.05 and not car.is_wrong_way:
            active_coins = [c for c in track.coins if c.active]
            if active_coins:
                fwd_vec = car.forward_unit_vector()
                # Monedas dentro de un radio de 240 px y con proyección hacia adelante
                close_coins = [
                    c for c in active_coins
                    if car.position.distance_to(c.position) < 240.0
                    and (c.position - car.position).dot(fwd_vec) > 10.0
                ]
                if close_coins:
                    nearest = min(close_coins, key=lambda c: car.position.distance_to(c.position))
                    dist = car.position.distance_to(nearest.position)
                    r_prox = max(0.0, (240.0 - dist) / 240.0) * cfg.coin_proximity_weight
                    total_reward += r_prox
                    breakdown["coin_proximity"] = round(r_prox, 4)

        # 4. Premios y castigos por cruzar Checkpoints o Línea de Meta
        if lap_completed:
            total_reward += cfg.lap_completion_reward
            breakdown["lap_completion"] = cfg.lap_completion_reward
        elif checkpoint_advanced:
            total_reward += cfg.checkpoint_reward
            breakdown["checkpoint"] = cfg.checkpoint_reward

        for gate_id, is_forward in gate_crossings:
            if not is_forward:
                if gate_id == 0:
                    total_reward += cfg.reverse_finish_line_penalty
                    breakdown["reverse_finish_line"] = cfg.reverse_finish_line_penalty
                    if cfg.terminate_on_reverse_finish_line:
                        terminated = True
                        termination_reason = "reverse_finish_line"
                else:
                    total_reward += cfg.reverse_checkpoint_penalty
                    breakdown["reverse_checkpoint"] = cfg.reverse_checkpoint_penalty

        # 5. Penalizaciones por chocar con obstáculos físicos (Conos, Barreras, Muros)
        hit_cone = "cone" in car.active_collisions
        hit_barrier = "barrier" in car.active_collisions
        hit_wall = "wall" in car.active_collisions

        if hit_cone and not self._prev_cone_contact:
            total_reward += cfg.cone_collision_penalty
            breakdown["cone_collision"] = cfg.cone_collision_penalty
        self._prev_cone_contact = hit_cone

        if hit_barrier:
            if not self._prev_barrier_contact:
                total_reward += cfg.barrier_collision_penalty
                breakdown["barrier_collision"] = cfg.barrier_collision_penalty
            if cfg.terminate_on_barrier_crash:
                terminated = True
                termination_reason = "barrier_crash"
        self._prev_barrier_contact = hit_barrier

        if hit_wall:
            if not self._prev_wall_contact:
                total_reward += cfg.wall_collision_penalty
                breakdown["wall_collision"] = cfg.wall_collision_penalty
            if cfg.terminate_on_wall_crash:
                terminated = True
                termination_reason = "wall_crash"
        self._prev_wall_contact = hit_wall

        # 6. Penalización severa por quedarse estancado contra la pared vs Premio por retroceder y corregir
        front_blocked = (d_0 < 0.16) or hit_wall or hit_barrier
        if front_blocked:
            if v_fwd < -12.0 and d_0 >= self._prev_d0:
                # ¡Está retrocediendo activamente para despegarse del muro y abrir ángulo!
                self.wall_stuck_counter = max(0, self.wall_stuck_counter - 2)
                car.stuck_against_wall_steps = self.wall_stuck_counter
                total_reward += cfg.reverse_escape_reward
                breakdown["reverse_escape_bonus"] = cfg.reverse_escape_reward
            elif v_fwd < 45.0:
                # Está bloqueado contra la pared sin retroceder (haciendo micro-correcciones inútiles)
                self.wall_stuck_counter += 1
                car.stuck_against_wall_steps = self.wall_stuck_counter
                escalation = 1.0 + (self.wall_stuck_counter / 15.0) * (
                    cfg.wall_stuck_escalation_factor - 1.0
                )
                stuck_pen = cfg.wall_stuck_step_penalty * escalation
                total_reward += stuck_pen
                breakdown["wall_stuck_penalty"] = round(stuck_pen, 4)

                if self.wall_stuck_counter >= cfg.max_wall_stuck_steps:
                    total_reward += cfg.wall_stuck_timeout_penalty
                    breakdown["wall_stuck_timeout"] = cfg.wall_stuck_timeout_penalty
                    terminated = True
                    termination_reason = "wall_stuck_timeout"
        else:
            if d_0 > 0.25:
                self.wall_stuck_counter = 0
                car.stuck_against_wall_steps = 0

        self._prev_d0 = d_0

        # 7. Penalizaciones por obstáculos de terreno (Baches gigantes, Manchas de aceite, Césped)
        if car.current_surface_name == "pothole" or "pothole" in car.active_collisions:
            total_reward += cfg.pothole_step_penalty
            breakdown["pothole_penalty"] = cfg.pothole_step_penalty
        elif car.current_surface_name == "oil" or "oil_slick" in car.active_collisions:
            total_reward += cfg.oil_slick_step_penalty
            breakdown["oil_slick_penalty"] = cfg.oil_slick_step_penalty
        elif car.current_surface_name == "grass":
            total_reward += cfg.grass_step_penalty
            breakdown["grass_penalty"] = cfg.grass_step_penalty
        elif car.current_surface_name == "kerb":
            total_reward += cfg.kerb_step_penalty

        # 8. Penalización por ir en sentido contrario durante mucho tiempo (no aplica durante maniobra corta de reversa)
        is_reversing_to_unstuck = (v_fwd < -5.0) and (d_0 < 0.35)
        if (car.is_wrong_way or heading_alignment < -0.35) and not is_reversing_to_unstuck:
            self.wrong_way_Dynamic_counter += 1
            escalation = 1.0 + (self.wrong_way_Dynamic_counter / 30.0) * (
                cfg.wrong_way_escalation_factor - 1.0
            )
            ww_pen = cfg.wrong_way_step_penalty * escalation
            total_reward += ww_pen
            breakdown["wrong_way_penalty"] = round(ww_pen, 4)

            if self.wrong_way_Dynamic_counter >= cfg.max_wrong_way_steps:
                total_reward += cfg.wrong_way_timeout_penalty
                breakdown["wrong_way_timeout"] = cfg.wrong_way_timeout_penalty
                terminated = True
                termination_reason = "wrong_way_timeout"
        else:
            self.wrong_way_Dynamic_counter = max(0, self.wrong_way_Dynamic_counter - 2)

        # 9. Penalización por inactividad / quedarse detenido
        if car.velocity.magnitude() < cfg.idle_speed_threshold:
            self.idle_steps_counter += 1
            total_reward += cfg.idle_step_penalty
            if self.idle_steps_counter >= cfg.max_idle_steps:
                total_reward += cfg.idle_timeout_penalty
                breakdown["idle_timeout"] = cfg.idle_timeout_penalty
                terminated = True
                termination_reason = "idle_timeout"
        else:
            self.idle_steps_counter = 0

        breakdown["terminated"] = terminated
        breakdown["termination_reason"] = termination_reason
        return float(total_reward), terminated, breakdown
