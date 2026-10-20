"""Objeto de Valor Color para parametrizar la paleta de los vehículos."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Color:
    """Representa un color RGB inmutable inicializable por componentes o código hexadecimal."""

    r: int
    g: int
    b: int

    def __post_init__(self) -> None:
        for channel_name, channel_val in (("r", self.r), ("g", self.g), ("b", self.b)):
            if not (0 <= channel_val <= 255):
                raise ValueError(
                    f"Canal de color {channel_name}={channel_val} fuera de rango [0, 255]."
                )

    @classmethod
    def from_hex(cls, hex_code: str) -> Color:
        """Crea una instancia Color a partir de una cadena hexadecimal '#RRGGBB' o 'RRGGBB'."""
        cleaned = hex_code.strip().lstrip("#")
        if len(cleaned) != 6:
            raise ValueError(f"Formato hexadecimal inválido: '{hex_code}'. Se espera #RRGGBB.")
        r = int(cleaned[0:2], 16)
        g = int(cleaned[2:4], 16)
        b = int(cleaned[4:6], 16)
        return cls(r=r, g=g, b=b)

    @property
    def hex_code(self) -> str:
        """Retorna la representación hexadecimal en mayúsculas '#RRGGBB'."""
        return f"#{self.r:02X}{self.g:02X}{self.b:02X}"

    def to_rgb_tuple(self) -> tuple[int, int, int]:
        """Retorna la tupla (R, G, B)."""
        return (self.r, self.g, self.b)

    def darkened(self, factor: float = 0.65) -> Color:
        """Retorna una variante más oscura del color multiplicando por factor en [0, 1]."""
        factor = max(0.0, min(1.0, factor))
        return Color(
            r=int(round(self.r * factor)),
            g=int(round(self.g * factor)),
            b=int(round(self.b * factor)),
        )

    def lightened(self, factor: float = 0.35) -> Color:
        """Retorna una variante más clara mezclando hacia blanco."""
        factor = max(0.0, min(1.0, factor))
        return Color(
            r=int(round(self.r + (255 - self.r) * factor)),
            g=int(round(self.g + (255 - self.g) * factor)),
            b=int(round(self.b + (255 - self.b) * factor)),
        )


# Paleta predefinida para vehículos (jugador y futuros agentes IA/CPU)
CAR_PALETTE: dict[str, Color] = {
    "crimson_red": Color.from_hex("#E63946"),
    "electric_blue": Color.from_hex("#00B4D8"),
    "emerald_green": Color.from_hex("#2EC4B6"),
    "amber_gold": Color.from_hex("#FFBE0B"),
    "royal_purple": Color.from_hex("#8338EC"),
}
