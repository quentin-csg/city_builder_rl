"""Types de terrain, génération procédurale (Perlin noise)."""

from __future__ import annotations

import logging
from enum import Enum

import numpy as np
import vnoise

logger = logging.getLogger(__name__)

_MAX_GENERATION_ATTEMPTS = 100
_PERLIN_SCALE = 0.10
_MIN_PLAIN = 350
_MIN_HILL = 30
_MIN_HILL_BLOCKS_3X3 = 3


class TerrainType(str, Enum):
    """Types de terrain disponibles sur la grille 32x32."""

    PLAIN = "plain"
    FOREST = "forest"
    HILL = "hill"
    WATER = "water"
    MARSH = "marsh"


def _count_hill_blocks_3x3(terrain: list[list[TerrainType]], size: int) -> int:
    """Compte le nombre de blocs 3×3 entièrement HILL (non chevauchants)."""
    count = 0
    used = [[False] * size for _ in range(size)]
    for y in range(size - 2):
        for x in range(size - 2):
            if used[y][x]:
                continue
            all_hill = all(
                terrain[y + dy][x + dx] == TerrainType.HILL
                for dy in range(3)
                for dx in range(3)
            )
            if all_hill:
                count += 1
                for dy in range(3):
                    for dx in range(3):
                        used[y + dy][x + dx] = True
    return count


def _generate_river(
    terrain: list[list[TerrainType]], size: int, rng: np.random.Generator
) -> None:
    """Trace une rivière continue du bord gauche au bord droit (random walk).

    Garantit la 4-connexité : quand y change de ligne, une case de liaison est posée.
    """
    y = int(rng.integers(1, size - 1))
    for x in range(size):
        terrain[y][x] = TerrainType.WATER
        if x < size - 1:
            step = int(rng.choice([-1, 0, 0, 0, 1]))
            new_y = max(1, min(size - 2, y + step))
            if new_y != y:
                # Liaison verticale pour garantir la 4-connexité
                terrain[new_y][x] = TerrainType.WATER
            y = new_y


def _generate_terrain_attempt(
    size: int, seed: int
) -> list[list[TerrainType]] | None:
    """Tente de générer un terrain viable. Retourne None si les contraintes ne sont pas respectées."""
    noise = vnoise.Noise(seed=seed % (2**31))
    rng = np.random.default_rng(seed)

    terrain: list[list[TerrainType]] = []
    for y in range(size):
        row: list[TerrainType] = []
        for x in range(size):
            val = noise.noise2(x * _PERLIN_SCALE, y * _PERLIN_SCALE, grid_mode=False)
            # val dans [-1, 1] typiquement, centré autour de 0
            if val < -0.35:
                row.append(TerrainType.MARSH)
            elif val < 0.25:
                row.append(TerrainType.PLAIN)
            elif val < 0.45:
                row.append(TerrainType.FOREST)
            else:
                row.append(TerrainType.HILL)
        terrain.append(row)

    _generate_river(terrain, size, rng)

    # Validation viabilité
    plain_count = sum(
        1 for y in range(size) for x in range(size) if terrain[y][x] == TerrainType.PLAIN
    )
    hill_count = sum(
        1 for y in range(size) for x in range(size) if terrain[y][x] == TerrainType.HILL
    )

    if plain_count < _MIN_PLAIN:
        return None
    if hill_count < _MIN_HILL:
        return None
    if _count_hill_blocks_3x3(terrain, size) < _MIN_HILL_BLOCKS_3X3:
        return None

    return terrain


def generate_terrain(size: int = 32, seed: int = 42) -> list[list[TerrainType]]:
    """Génère un terrain procédural viable.

    Args:
        size: Taille de la grille (carrée).
        seed: Seed pour la reproductibilité.

    Returns:
        Grille 2D de TerrainType.

    Raises:
        RuntimeError: Si aucun terrain viable n'est trouvé après 100 tentatives.
    """
    for attempt in range(_MAX_GENERATION_ATTEMPTS):
        result = _generate_terrain_attempt(size, seed + attempt)
        if result is not None:
            if attempt > 0:
                logger.info("Terrain viable trouvé au seed %d (tentative %d)", seed + attempt, attempt + 1)
            return result

    raise RuntimeError(
        f"Impossible de générer un terrain viable après {_MAX_GENERATION_ATTEMPTS} "
        f"tentatives (seed initial={seed}, size={size})."
    )
