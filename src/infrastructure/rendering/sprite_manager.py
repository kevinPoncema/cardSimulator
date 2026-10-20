"""Gestor y generador de sprites 2D de vehículos con soporte de paleta de colores."""

from __future__ import annotations

from pathlib import Path

import pygame

from src.domain.value_objects.color import CAR_PALETTE, Color


class SpriteManager:
    """Genera, tiñe y almacena en caché los sprites top-down de los coches."""

    def __init__(self, sprites_dir: str | Path | None = None) -> None:
        self.sprites_dir = Path(sprites_dir) if sprites_dir is not None else None
        self._base_cache: dict[tuple[str, int, int], pygame.Surface] = {}

    def get_car_surface(
        self, color: Color, length: int = 44, width: int = 24
    ) -> pygame.Surface:
        """Obtiene (o construye en caché) el sprite base orientado hacia +X (0 rad)."""
        key = (color.hex_code, int(length), int(width))
        if key not in self._base_cache:
            self._base_cache[key] = self._build_car_sprite(color, int(length), int(width))
        return self._base_cache[key]

    @staticmethod
    def _build_car_sprite(color: Color, length: int, width: int) -> pygame.Surface:
        """Dibuja un sprite estilo arcade 16-bits con ruedas, chasis, parabrisas y faros."""
        surf = pygame.Surface((length, width), pygame.SRCALPHA)

        wheel_color = (22, 24, 28)
        wheel_w = max(8, length // 5)
        wheel_h = max(4, width // 6)

        # 4 neumáticos sobresaliendo ligeramente
        wheel_positions = [
            (length - wheel_w - 5, 0),                   # Delantera izq
            (length - wheel_w - 5, width - wheel_h),     # Delantera der
            (5, 0),                                      # Trasera izq
            (5, width - wheel_h),                        # Trasera der
        ]
        for wx, wy in wheel_positions:
            pygame.draw.rect(surf, wheel_color, (wx, wy, wheel_w, wheel_h), border_radius=2)

        # Sombra / borde del chasis
        body_rect = pygame.Rect(2, 2, length - 4, width - 4)
        dark_rgb = color.darkened(0.55).to_rgb_tuple()
        main_rgb = color.to_rgb_tuple()
        light_rgb = color.lightened(0.40).to_rgb_tuple()

        pygame.draw.rect(surf, dark_rgb, body_rect, border_radius=6)
        inner_rect = body_rect.inflate(-2, -2)
        pygame.draw.rect(surf, main_rgb, inner_rect, border_radius=5)

        # Franja central deportiva
        stripe_h = max(2, width // 6)
        stripe_rect = pygame.Rect(4, (width - stripe_h) // 2, length - 8, stripe_h)
        pygame.draw.rect(surf, light_rgb, stripe_rect, border_radius=1)

        # Cabina / parabrisas
        cockpit_x = int(length * 0.30)
        cockpit_w = int(length * 0.40)
        cockpit_rect = pygame.Rect(cockpit_x, 5, cockpit_w, width - 10)
        pygame.draw.rect(surf, (30, 42, 58), cockpit_rect, border_radius=3)

        windshield_rect = pygame.Rect(
            cockpit_x + int(cockpit_w * 0.45), 6, int(cockpit_w * 0.48), width - 12
        )
        pygame.draw.rect(surf, (110, 175, 220), windshield_rect, border_radius=2)

        # Faros delanteros (derecha del sprite = +X)
        pygame.draw.rect(surf, (255, 248, 180), (length - 4, 4, 3, 4), border_radius=1)
        pygame.draw.rect(
            surf, (255, 248, 180), (length - 4, width - 8, 3, 4), border_radius=1
        )

        # Luces de freno traseras (izquierda del sprite = -X)
        pygame.draw.rect(surf, (235, 40, 40), (2, 4, 3, 4), border_radius=1)
        pygame.draw.rect(surf, (235, 40, 40), (2, width - 8, 3, 4), border_radius=1)

        return surf

    def export_default_palette_sprites(self, output_dir: str | Path) -> list[Path]:
        """Genera y guarda en disco los sprites PNG de la paleta de colores en assets/sprites/."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        saved_files: list[Path] = []
        for name, color in CAR_PALETTE.items():
            surf = self.get_car_surface(color)
            file_path = out_path / f"car_{name}.png"
            pygame.image.save(surf, str(file_path))
            saved_files.append(file_path)
        return saved_files
