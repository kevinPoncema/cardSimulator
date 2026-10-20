"""Renderizador 2D en Pygame-CE con soporte para obstáculos dinámicos, baches, aceite y 12 mapas."""

from __future__ import annotations

import math

import numpy as np
import pygame

from src.domain.entities.car import Car
from src.domain.entities.obstacle import Obstacle, ObstacleType
from src.domain.entities.sensor_reading import SensorReading
from src.domain.entities.track import Track
from src.domain.interfaces.renderer import IRenderer
from src.domain.value_objects.vector2d import Vector2D
from src.infrastructure.rendering.sprite_manager import SpriteManager


class PygameRenderer(IRenderer):
    """Implementación de IRenderer sobre Pygame-CE con renderizado de obstáculos y 12 mapas."""

    def __init__(
        self,
        width: int = 1280,
        height: int = 800,
        title: str = "Simulador de Carreras 2D - 12 Circuitos y Obstáculos Interactivos",
        headless: bool = False,
    ) -> None:
        self.viewport_width = int(width)
        self.viewport_height = int(height)
        self.title = title
        self.headless = headless

        if not pygame.get_init():
            pygame.init()
        if not pygame.font.get_init():
            pygame.font.init()

        flags = pygame.HIDDEN if self.headless else 0
        self.screen = pygame.display.set_mode(
            (self.viewport_width, self.viewport_height), flags
        )
        pygame.display.set_caption(self.title)

        self.sprite_manager = SpriteManager()
        self._track_surface_cache: pygame.Surface | None = None
        self._minimap_surface_cache: pygame.Surface | None = None
        self._cached_track_name: str | None = None

        self.camera_pos = Vector2D(0.0, 0.0)
        self._camera_initialized: bool = False

        self.font_hud = pygame.font.SysFont("monospace", 14, bold=True)
        self.font_title = pygame.font.SysFont("monospace", 15, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 12, bold=True)
        self.font_score = pygame.font.SysFont("monospace", 22, bold=True)

    def _build_track_surface(self, track: Track) -> pygame.Surface:
        """Pre-renderiza el mundo completo del circuito y los peligros estáticos de terreno."""
        world_w = max(self.viewport_width, int(track.width))
        world_h = max(self.viewport_height, int(track.height))
        surf = pygame.Surface((world_w, world_h))

        # 1. Césped exterior con franjas retro de 16 bits
        surf.fill((34, 112, 56))
        stripe_h = 64
        for y in range(0, world_h, stripe_h * 2):
            pygame.draw.rect(surf, (30, 102, 50), (0, y, world_w, stripe_h))

        # 2. Zona entre muros (escapatoria)
        outer_wall_pts = [pt.to_int_tuple() for pt in track.outer_wall_polygon]
        inner_wall_pts = [pt.to_int_tuple() for pt in track.inner_wall_polygon]
        if len(outer_wall_pts) >= 3:
            pygame.draw.polygon(surf, (46, 138, 70), outer_wall_pts)
        if len(inner_wall_pts) >= 3:
            pygame.draw.polygon(surf, (34, 112, 56), inner_wall_pts)

        # 3. Pianos / Bordes
        n_pts = len(track.centerline)
        kerb_outer = track.outer_kerb_polygon
        kerb_inner = track.inner_kerb_polygon
        for i in range(n_pts):
            nxt = (i + 1) % n_pts
            kerb_color = (225, 45, 45) if (i % 2 == 0) else (242, 242, 248)
            quad = [
                kerb_outer[i].to_int_tuple(),
                kerb_outer[nxt].to_int_tuple(),
                kerb_inner[nxt].to_int_tuple(),
                kerb_inner[i].to_int_tuple(),
            ]
            pygame.draw.polygon(surf, kerb_color, quad)

        # 4. Asfalto amplio
        asphalt_outer = track.outer_asphalt_polygon
        asphalt_inner = track.inner_asphalt_polygon
        for i in range(n_pts):
            nxt = (i + 1) % n_pts
            asphalt_color = (58, 62, 72) if (i % 2 == 0) else (53, 57, 66)
            quad = [
                asphalt_outer[i].to_int_tuple(),
                asphalt_outer[nxt].to_int_tuple(),
                asphalt_inner[nxt].to_int_tuple(),
                asphalt_inner[i].to_int_tuple(),
            ]
            pygame.draw.polygon(surf, asphalt_color, quad)

        # 5. Líneas de bordes de asfalto y centro
        for i in range(n_pts):
            nxt = (i + 1) % n_pts
            pygame.draw.line(surf, (210, 215, 225), asphalt_outer[i].to_int_tuple(), asphalt_outer[nxt].to_int_tuple(), 3)
            pygame.draw.line(surf, (210, 215, 225), asphalt_inner[i].to_int_tuple(), asphalt_inner[nxt].to_int_tuple(), 3)
            if i % 2 == 0:
                pygame.draw.line(surf, (150, 160, 175), track.centerline[i].to_int_tuple(), track.centerline[nxt].to_int_tuple(), 3)

        # 6. Obstáculos de terreno estáticos (Baches y Manchas de Aceite)
        for obs in track.obstacles:
            pos_int = obs.position.to_int_tuple()
            if obs.kind == ObstacleType.POTHOLE:
                # Bache gigante: Cráter oscuro con borde agrietado
                pygame.draw.circle(surf, (32, 34, 40), pos_int, int(obs.radius) + 3)
                pygame.draw.circle(surf, (20, 22, 26), pos_int, int(obs.radius))
                pygame.draw.circle(surf, (12, 14, 18), pos_int, int(obs.radius * 0.55))
            elif obs.kind == ObstacleType.OIL_SLICK:
                # Mancha de aceite: Charco iridiscente brillante
                pygame.draw.circle(surf, (22, 16, 32), pos_int, int(obs.radius))
                pygame.draw.circle(surf, (45, 25, 60), pos_int, int(obs.radius * 0.70))
                pygame.draw.circle(surf, (65, 40, 85), (pos_int[0] - 4, pos_int[1] - 4), int(obs.radius * 0.35))

        # 7. Líneas de Checkpoints intermedios
        for gate in track.gates:
            if gate.gate_id == 0:
                continue
            idx = (track.start_index + (gate.gate_id * n_pts) // len(track.gates)) % n_pts
            pygame.draw.line(
                surf,
                (70, 180, 230),
                asphalt_outer[idx].to_int_tuple(),
                asphalt_inner[idx].to_int_tuple(),
                4,
            )

        # 8. Línea de salida / meta ajedrezada
        start_idx = track.start_index % n_pts
        p_out = asphalt_outer[start_idx]
        p_in = asphalt_inner[start_idx]
        num_checks = 12
        for c in range(num_checks):
            t0 = c / float(num_checks)
            t1 = (c + 1) / float(num_checks)
            a = p_in + (p_out - p_in) * t0
            b = p_in + (p_out - p_in) * t1
            check_color = (250, 250, 250) if c % 2 == 0 else (20, 20, 20)
            pygame.draw.line(surf, check_color, a.to_int_tuple(), b.to_int_tuple(), 10)

        # 9. Barreras físicas estáticas
        for seg in track.outer_walls + track.inner_walls:
            pygame.draw.line(surf, (220, 225, 232), seg.start.to_int_tuple(), seg.end.to_int_tuple(), 8)
            pygame.draw.line(surf, (40, 46, 56), seg.start.to_int_tuple(), seg.end.to_int_tuple(), 2)

        return surf

    def _build_minimap_surface(
        self, track_surf: pygame.Surface, minimap_size: tuple[int, int] = (230, 155)
    ) -> pygame.Surface:
        """Crea una miniatura escalada del circuito completo para mostrar el minimapa."""
        scaled = pygame.transform.smoothscale(track_surf, minimap_size)
        pygame.draw.rect(
            scaled, (90, 180, 250), (0, 0, minimap_size[0], minimap_size[1]), 2
        )
        return scaled

    def _update_camera(self, target_pos: Vector2D, track: Track) -> None:
        """Actualiza suavemente la posición de la cámara centrada en el vehículo."""
        desired_x = target_pos.x - self.viewport_width * 0.5
        desired_y = target_pos.y - self.viewport_height * 0.5

        max_x = max(0.0, track.width - self.viewport_width)
        max_y = max(0.0, track.height - self.viewport_height)

        clamped_x = max(0.0, min(max_x, desired_x))
        clamped_y = max(0.0, min(max_y, desired_y))

        if not self._camera_initialized:
            self.camera_pos = Vector2D(clamped_x, clamped_y)
            self._camera_initialized = True
        else:
            lerp = 0.25
            self.camera_pos = Vector2D(
                self.camera_pos.x + (clamped_x - self.camera_pos.x) * lerp,
                self.camera_pos.y + (clamped_y - self.camera_pos.y) * lerp,
            )

    def world_to_screen(self, world_pt: Vector2D) -> tuple[int, int]:
        """Transforma coordenadas del mundo a coordenadas de píxeles en la ventana (cámara)."""
        return (
            int(round(world_pt.x - self.camera_pos.x)),
            int(round(world_pt.y - self.camera_pos.y)),
        )

    def _draw_active_checkpoint_highlight(self, track: Track, car: Car) -> None:
        """Resalta en pantalla el siguiente checkpoint o la meta."""
        if not track.gates:
            return
        target_gate_id = car.next_checkpoint_id % len(track.gates)
        gate = track.gates[target_gate_id]
        p_out = self.world_to_screen(gate.outer_point)
        p_in = self.world_to_screen(gate.inner_point)
        color = (255, 215, 0) if target_gate_id == 0 else (0, 230, 255)
        pygame.draw.line(self.screen, color, p_out, p_in, 3)

    def _draw_car(self, car: Car) -> None:
        """Dibuja el sprite del coche rotado en coordenadas de cámara."""
        base_sprite = self.sprite_manager.get_car_surface(
            car.color,
            length=int(car.config.length),
            width=int(car.config.width),
        )
        angle_deg = -math.degrees(car.angle.radians)
        rotated_sprite = pygame.transform.rotate(base_sprite, angle_deg)
        screen_center = self.world_to_screen(car.position)
        rect = rotated_sprite.get_rect(center=screen_center)
        self.screen.blit(rotated_sprite, rect)

    def _draw_physical_obstacles(
        self, dynamic_obstacles: list[tuple[Obstacle, Vector2D, float]] | None
    ) -> None:
        """Dibuja conos de tráfico y barreras de hormigón en sus posiciones físicas actuales."""
        if not dynamic_obstacles:
            return

        for obs, pos, ang_deg in dynamic_obstacles:
            screen_pos = self.world_to_screen(pos)

            if obs.kind == ObstacleType.CONE:
                # Cono de tráfico naranja brillante con anillo blanco
                r = int(obs.radius)
                pygame.draw.circle(self.screen, (35, 38, 45), screen_pos, r + 2)
                pygame.draw.circle(self.screen, (255, 95, 15), screen_pos, r)
                pygame.draw.circle(self.screen, (250, 250, 250), screen_pos, int(r * 0.65))
                pygame.draw.circle(self.screen, (255, 120, 30), screen_pos, int(r * 0.35))

            elif obs.kind == ObstacleType.BARRIER:
                # Barrera rectangular con franjas diagonales de señalización
                surf_b = pygame.Surface((int(obs.width), int(obs.height)), pygame.SRCALPHA)
                surf_b.fill((215, 45, 45))
                # Franjas blancas
                stripe_w = 8
                for sx in range(0, int(obs.width), stripe_w * 2):
                    pygame.draw.rect(surf_b, (245, 245, 250), (sx, 0, stripe_w, int(obs.height)))
                pygame.draw.rect(surf_b, (20, 24, 30), (0, 0, int(obs.width), int(obs.height)), 2)

                rotated = pygame.transform.rotate(surf_b, -ang_deg)
                rect = rotated.get_rect(center=screen_pos)
                self.screen.blit(rotated, rect)

    def _draw_lidar_overlay(self, reading: SensorReading) -> None:
        """Dibuja los rayos LIDAR en verde o rojo con puntos de impacto."""
        for ray in reading.rays:
            start_pos = self.world_to_screen(ray.origin)
            end_pos = self.world_to_screen(ray.end_point)
            if ray.hit:
                pygame.draw.line(self.screen, (255, 50, 50), start_pos, end_pos, 2)
                pygame.draw.circle(self.screen, (255, 220, 40), end_pos, 6)
                pygame.draw.circle(self.screen, (255, 40, 40), end_pos, 3)
            else:
                pygame.draw.line(self.screen, (0, 235, 100), start_pos, end_pos, 2)
                pygame.draw.circle(self.screen, (0, 235, 100), end_pos, 3)

        if reading.rays:
            emitter_pos = self.world_to_screen(reading.rays[0].origin)
            pygame.draw.circle(self.screen, (255, 255, 255), emitter_pos, 4)

    def _draw_minimap(self, track: Track, cars: list[Car]) -> None:
        """Dibuja el minimapa en la esquina superior derecha."""
        if self._minimap_surface_cache is None:
            return
        mm_w, mm_h = self._minimap_surface_cache.get_size()
        mm_x = self.viewport_width - mm_w - 14
        mm_y = 44

        self.screen.blit(self._minimap_surface_cache, (mm_x, mm_y))

        scale_x = mm_w / max(1.0, track.width)
        scale_y = mm_h / max(1.0, track.height)

        # Resaltar siguiente checkpoint en minimapa
        if cars and track.gates:
            next_gate = track.gates[cars[0].next_checkpoint_id % len(track.gates)]
            gx = mm_x + int(next_gate.center.x * scale_x)
            gy = mm_y + int(next_gate.center.y * scale_y)
            gate_color = (255, 215, 0) if next_gate.gate_id == 0 else (0, 235, 255)
            pygame.draw.circle(self.screen, gate_color, (gx, gy), 4)

        # Rectángulo de cámara
        cam_rect = pygame.Rect(
            mm_x + int(self.camera_pos.x * scale_x),
            mm_y + int(self.camera_pos.y * scale_y),
            max(10, int(self.viewport_width * scale_x)),
            max(10, int(self.viewport_height * scale_y)),
        )
        pygame.draw.rect(self.screen, (255, 255, 255), cam_rect, 1)

        # Posición de coches
        for car in cars:
            cx = mm_x + int(car.position.x * scale_x)
            cy = mm_y + int(car.position.y * scale_y)
            pygame.draw.circle(self.screen, (255, 255, 255), (cx, cy), 5)
            pygame.draw.circle(self.screen, car.color.to_rgb_tuple(), (cx, cy), 3)

        label_surf = self.font_small.render(
            "[M] Siguiente | [N] Anterior | [R] Reset", True, (220, 238, 255)
        )
        self.screen.blit(label_surf, (mm_x - 38, mm_y + mm_h + 6))

    def _draw_score_panel(self, car: Car, total_gates: int) -> None:
        """Dibuja el marcador de Puntos y progreso de sectores."""
        box_w, box_h = 330, 68
        box_x = (self.viewport_width - box_w) // 2
        box_y = 44

        panel = pygame.Surface((box_w, box_h), pygame.SRCALPHA)
        panel.fill((14, 22, 34, 230))
        border_col = (255, 210, 60) if car.score > 0 else (90, 190, 255)
        pygame.draw.rect(panel, border_col, (0, 0, box_w, box_h), 2, border_radius=8)
        self.screen.blit(panel, (box_x, box_y))

        score_str = f"PUNTOS: {car.score}"
        score_surf = self.font_score.render(score_str, True, (255, 230, 80))
        score_rect = score_surf.get_rect(centerx=box_x + box_w // 2, top=box_y + 8)
        self.screen.blit(score_surf, score_rect)

        total_cp = max(1, total_gates - 1)
        if car.checkpoints_passed_in_lap >= total_cp:
            cp_text = "¡VUELTA LISTA! CRUZA LA META -> +1 PTO"
            cp_color = (100, 255, 140)
        else:
            cp_text = f"Checkpoints: {car.checkpoints_passed_in_lap} / {total_cp}"
            cp_color = (190, 225, 255)

        cp_surf = self.font_small.render(cp_text, True, cp_color)
        cp_rect = cp_surf.get_rect(centerx=box_x + box_w // 2, top=box_y + 40)
        self.screen.blit(cp_surf, cp_rect)

        if car.is_wrong_way:
            warn_w, warn_h = 390, 36
            warn_x = (self.viewport_width - warn_w) // 2
            warn_y = box_y + box_h + 8
            warn_panel = pygame.Surface((warn_w, warn_h), pygame.SRCALPHA)
            warn_panel.fill((180, 20, 20, 235))
            pygame.draw.rect(warn_panel, (255, 220, 60), (0, 0, warn_w, warn_h), 2, border_radius=6)
            self.screen.blit(warn_panel, (warn_x, warn_y))
            warn_surf = self.font_title.render("¡SENTIDO CONTRARIO! DA LA VUELTA", True, (255, 255, 255))
            warn_rect = warn_surf.get_rect(center=(warn_x + warn_w // 2, warn_y + warn_h // 2))
            self.screen.blit(warn_surf, warn_rect)

    def _draw_hud(
        self,
        track: Track,
        car: Car,
        reading: SensorReading | None,
        obs: np.ndarray | None,
        debug_mode: bool,
        fps: float,
    ) -> None:
        """Dibuja la barra superior con información de pista y HUD de telemetría."""
        top_bar = pygame.Surface((self.viewport_width, 34), pygame.SRCALPHA)
        top_bar.fill((12, 16, 24, 220))
        self.screen.blit(top_bar, (0, 0))

        surf_display = car.current_surface_name.upper()
        if car.current_surface_name == "pothole":
            surf_display = "¡BACHE GIGANTE!"
        elif car.current_surface_name == "oil":
            surf_display = "¡MANCHA DE ACEITE!"

        status_text = (
            f"FPS: {fps:3.0f} | Pts: {car.score} | {track.name} [{track.category.upper()}] | "
            f"Suelo: {surf_display:14s} | Vel: {car.forward_speed():4.0f}px/s | "
            f"[M/N] Mapa [1-0,-,=] [R] Reset [F3] LIDAR"
        )
        txt_surf = self.font_small.render(status_text, True, (230, 240, 250))
        self.screen.blit(txt_surf, (12, 9))

        self._draw_score_panel(car, total_gates=len(track.gates))

        if not debug_mode or obs is None:
            return

        panel_w, panel_h = 380, 280
        panel_x, panel_y = 14, 44
        panel = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
        panel.fill((14, 20, 30, 225))
        pygame.draw.rect(panel, (70, 140, 210), (0, 0, panel_w, panel_h), 2, border_radius=6)
        self.screen.blit(panel, (panel_x, panel_y))

        title_surf = self.font_title.render("TELEMETRÍA LIDAR & VECTOR S (F3)", True, (90, 200, 255))
        self.screen.blit(title_surf, (panel_x + 12, panel_y + 10))

        labels = [
            ("d_0   (Frontal   0°)", 0),
            ("d_45  (Diag Izq +45°)", 1),
            ("d_-45 (Diag Der -45°)", 2),
            ("d_90  (Lat Izq  +90°)", 3),
            ("d_-90 (Lat Der  -90°)", 4),
            ("v / v_max   (Lineal)", 5),
            ("w / w_max   (Angular)", 6),
        ]

        y_offset = panel_y + 38
        for label_str, idx in labels:
            val = float(obs[idx]) if idx < len(obs) else 0.0
            raw_px_str = ""
            if reading is not None and idx < len(reading.rays):
                raw_px_str = f"({reading.rays[idx].distance:5.1f}px)"

            color_val = (120, 255, 150)
            if idx < 5 and val < 0.35:
                color_val = (255, 95, 95)
            elif idx < 5 and val < 0.65:
                color_val = (255, 215, 90)

            line_text = f"S[{idx}] {label_str}: {val:+6.3f} {raw_px_str}"
            line_surf = self.font_hud.render(line_text, True, color_val)
            self.screen.blit(line_surf, (panel_x + 12, y_offset))
            y_offset += 24

        vec_formatted = ", ".join(f"{float(x):.2f}" for x in obs)
        vec_line = f"S = [{vec_formatted}]"
        vec_surf = self.font_small.render(vec_line, True, (200, 220, 255))
        self.screen.blit(vec_surf, (panel_x + 12, y_offset + 8))

        drift_state = (
            "DERRAPE ACTIVO (FRENO MANO)"
            if car.current_action.handbrake
            else ("¡DERRAPE INVOLUNTARIO SOBRE ACEITE!" if car.current_surface_name == "oil" else "AGARRE ARCADE ASISTIDO")
        )
        drift_color = (255, 175, 60) if (car.current_action.handbrake or car.current_surface_name == "oil") else (140, 230, 180)
        drift_surf = self.font_small.render(drift_state, True, drift_color)
        self.screen.blit(drift_surf, (panel_x + 12, y_offset + 28))

    def render(
        self,
        track: Track,
        cars: list[Car],
        readings: dict[str, SensorReading],
        observations: dict[str, np.ndarray],
        debug_mode: bool = False,
        fps: float = 60.0,
        dynamic_obstacles: list[tuple[Obstacle, Vector2D, float]] | None = None,
    ) -> None:
        """Renderiza el mundo visible, obstáculos, coches, LIDAR, minimapa y HUD."""
        if self._track_surface_cache is None or self._cached_track_name != track.name:
            self._track_surface_cache = self._build_track_surface(track)
            self._minimap_surface_cache = self._build_minimap_surface(self._track_surface_cache)
            self._cached_track_name = track.name
            self._camera_initialized = False

        if cars:
            self._update_camera(cars[0].position, track)

        cam_rect = pygame.Rect(
            int(round(self.camera_pos.x)),
            int(round(self.camera_pos.y)),
            self.viewport_width,
            self.viewport_height,
        )
        self.screen.blit(self._track_surface_cache, (0, 0), area=cam_rect)

        # Dibujar obstáculos físicos dinámicos (conos y barreras)
        self._draw_physical_obstacles(dynamic_obstacles)

        if cars:
            self._draw_active_checkpoint_highlight(track, cars[0])

        if debug_mode:
            for car in cars:
                reading = readings.get(car.car_id)
                if reading is not None:
                    self._draw_lidar_overlay(reading)

        for car in cars:
            self._draw_car(car)

        self._draw_minimap(track, cars)

        if cars:
            primary_car = cars[0]
            self._draw_hud(
                track=track,
                car=primary_car,
                reading=readings.get(primary_car.car_id),
                obs=observations.get(primary_car.car_id),
                debug_mode=debug_mode,
                fps=fps,
            )

        pygame.display.flip()

    def close(self) -> None:
        """Cierra la ventana de Pygame y libera recursos."""
        if pygame.get_init():
            pygame.quit()
