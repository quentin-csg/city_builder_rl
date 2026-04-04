"""Types de terrain, génération procédurale (Perlin noise)."""

from enum import Enum


class TerrainType(str, Enum):
    """Types de terrain disponibles sur la grille 32x32."""

    PLAIN = "plain"
    FOREST = "forest"
    HILL = "hill"
    WATER = "water"
    MARSH = "marsh"
