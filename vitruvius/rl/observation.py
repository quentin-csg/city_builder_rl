"""Encodage de l'état du jeu en tenseur : grille multi-canal (32×32×31) + features globales."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from vitruvius.engine.buildings import is_aqueduct_connected
from vitruvius.engine.services import compute_coverage_grid

if TYPE_CHECKING:
    from vitruvius.config import GameConfig
    from vitruvius.engine.game_state import GameState

# Canaux de couverture de services, dans l'ordre (indices 22-27)
_SERVICE_ORDER = ["water", "food", "religion", "hygiene", "entertainment", "security"]

# Valeurs terrain normalisées (canal 0)
_TERRAIN_VALUES: dict[str, float] = {
    "plain": 0.0,
    "forest": 0.25,
    "hill": 0.5,
    "water": 0.75,
    "marsh": 1.0,
}

_NUM_BUILDINGS = 20
_GRID_SIZE = 32
_GRID_CHANNELS = 31   # 1 terrain + 20 one-hot bâtiment + 1 level + 6 services + 1 pop + 1 aqueduct + 1 famine
_GLOBAL_FEATURES = 18  # 15 existants + 3 flags victoire (forum, obelisque, prefecture)
_MAX_POP = 70  # max_population niveau 6

# Layout des canaux grille :
#   0      : terrain
#   1-20   : one-hot building type (index 0-19 → canal 1-20)
#   21     : house level / 6
#   22-27  : couverture services (water/food/religion/hygiene/entertainment/security)
#   28     : pop / _MAX_POP
#   29     : aqueduc connecté WATER
#   30     : famine flag


def build_observation(
    gs: GameState,
    cfg: GameConfig,
    building_index_map: dict[str, int],
    dynamics: dict[str, float] | None = None,
) -> dict[str, np.ndarray]:
    """Construit l'observation RL depuis l'état courant.

    Args:
        gs: État courant du jeu.
        cfg: Configuration du jeu.
        building_index_map: Mapping building_id → index (0-indexed, 20 bâtiments).
        dynamics: Métriques inter-tour calculées par l'env :
            - growth_rate  : clampé [-1, 1]
            - wheat_conso_ratio : clampé [0, 1]
            - net_income   : clampé [-1, 1]
            Si None → 0.0 pour ces trois features.

    Returns:
        Dict avec clés "grid" (32,32,31 float32) et "global_features" (18, float32).
    """
    grid = gs.grid
    rs = gs.resource_state
    bldgs = cfg.buildings.buildings

    # ------------------------------------------------------------------
    # Pré-calculs (une seule fois)
    # ------------------------------------------------------------------
    coverage_grid = compute_coverage_grid(grid, bldgs, rs)  # dict[svc, set[(x,y)]]

    # Cache is_aqueduct_connected par tile aqueduct (évite 1024 BFS)
    aqueduct_connected: dict[tuple[int, int], bool] = {}
    for (ox, oy), pb in grid.placed_buildings.items():
        if pb.building_id == "aqueduct":
            for dy in range(pb.size[1]):
                for dx in range(pb.size[0]):
                    tile = (ox + dx, oy + dy)
                    if tile not in aqueduct_connected:
                        aqueduct_connected[tile] = is_aqueduct_connected(
                            grid, tile[0], tile[1], bldgs
                        )

    # Index maison par toutes les tiles qu'elle occupe (pour canaux 21, 28, 30)
    house_at: dict[tuple[int, int], object] = {}
    for origin, house in gs.houses.items():
        pb = grid.placed_buildings.get(origin)
        if pb is None:
            continue
        for dy in range(pb.size[1]):
            for dx in range(pb.size[0]):
                house_at[(origin[0] + dx, origin[1] + dy)] = house

    # ------------------------------------------------------------------
    # Grille 32×32×31
    # ------------------------------------------------------------------
    obs_grid = np.zeros((_GRID_SIZE, _GRID_SIZE, _GRID_CHANNELS), dtype=np.float32)

    for y in range(_GRID_SIZE):
        for x in range(_GRID_SIZE):
            tile = (x, y)

            # Canal 0 : terrain
            terrain_name = grid.terrain[y][x].name.lower()
            obs_grid[y, x, 0] = _TERRAIN_VALUES.get(terrain_name, 0.0)

            # Canaux 1-20 : one-hot type de bâtiment
            pb = grid.get_building_at(x, y)
            if pb is not None:
                idx = building_index_map.get(pb.building_id, -1)
                if idx >= 0:
                    obs_grid[y, x, 1 + idx] = 1.0

            # Canaux 22-27 : couverture de services
            for ch, svc in enumerate(_SERVICE_ORDER):
                if tile in coverage_grid[svc]:
                    obs_grid[y, x, 22 + ch] = 1.0

            # Canal 29 : aqueduc connecté WATER
            if tile in aqueduct_connected:
                obs_grid[y, x, 29] = 1.0 if aqueduct_connected[tile] else 0.0

            # Canaux 21, 28, 30 : spécifiques aux maisons
            house = house_at.get(tile)
            if house is not None:
                obs_grid[y, x, 21] = house.level / 6.0
                obs_grid[y, x, 28] = house.population / _MAX_POP
                obs_grid[y, x, 30] = 1.0 if house.famine else 0.0

    # ------------------------------------------------------------------
    # Features globales (18)
    # ------------------------------------------------------------------
    total_pop = sum(h.population for h in gs.houses.values())

    # Capacités de stockage
    from vitruvius.engine.resources import compute_storage_cap
    cap_wheat = compute_storage_cap("wheat", grid.placed_buildings, bldgs) or 0
    cap_wood = compute_storage_cap("wood", grid.placed_buildings, bldgs) or 0
    cap_marble = compute_storage_cap("marble", grid.placed_buildings, bldgs) or 0

    # Sécheresse active
    drought_active = any(e.event_type == "drought" for e in gs.active_events)
    drought_turns = next(
        (e.turns_remaining for e in gs.active_events if e.event_type == "drought"), 0
    )

    dyn = dynamics or {}
    growth_rate = float(np.clip(dyn.get("growth_rate", 0.0), -1.0, 1.0))
    wheat_conso_ratio = float(np.clip(dyn.get("wheat_conso_ratio", 0.0), 0.0, 1.0))
    net_income = float(np.clip(dyn.get("net_income", 0.0), -1.0, 1.0))

    # Flags victoire (O(1) via Counter)
    ids = grid._placed_ids
    has_forum = 1.0 if ids["forum"] > 0 else 0.0
    has_obelisk = 1.0 if ids["obelisk"] > 0 else 0.0
    has_prefecture = 1.0 if ids["prefecture"] > 0 else 0.0

    global_features = np.clip(np.array([
        rs.denarii / 10_000.0,                            # [0]
        rs.wheat / 5_000.0,                               # [1]
        rs.wood / 5_000.0,                                # [2]
        rs.marble / 500.0,                                # [3] — 200 marble = 0.40 (visible)
        total_pop / 5_000.0,                              # [4]
        gs.global_satisfaction,                           # [5]
        gs.city_level / 5.0,                              # [6]
        gs.turn / 1_000.0,                                # [7]
        cap_wheat / 5_000.0,                              # [8]
        (cap_wood + cap_marble) / 10_000.0,               # [9]
        growth_rate,                                       # [10]
        wheat_conso_ratio,                                 # [11]
        net_income,                                        # [12]
        1.0 if drought_active else 0.0,                   # [13]
        drought_turns / 3.0,                              # [14]
        has_forum,                                         # [15]
        has_obelisk,                                       # [16]
        has_prefecture,                                    # [17]
    ], dtype=np.float32), -1.0, 1.0)

    return {"grid": obs_grid, "global_features": global_features}
