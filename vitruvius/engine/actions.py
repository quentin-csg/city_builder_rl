"""Espace d'actions : énumération et encodage de toutes les actions légales par tour."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from vitruvius.engine.resources import can_afford
from vitruvius.engine.turn import Action

if TYPE_CHECKING:
    from vitruvius.config import GameConfig
    from vitruvius.engine.game_state import GameState


# ---------------------------------------------------------------------------
# Constantes d'encodage
# ---------------------------------------------------------------------------

GRID_SIZE: int = 32
CELLS: int = GRID_SIZE * GRID_SIZE          # 1024
NUM_BUILDINGS: int = 20
DEMOLISH_OFFSET: int = NUM_BUILDINGS * CELLS  # 20480
DO_NOTHING: int = (NUM_BUILDINGS + 1) * CELLS # 21504
TOTAL_ACTIONS: int = DO_NOTHING + 1           # 21505


# ---------------------------------------------------------------------------
# Ordre des bâtiments
# ---------------------------------------------------------------------------


def get_building_order(config: GameConfig) -> tuple[list[str], dict[str, int]]:
    """Retourne la liste ordonnée des bâtiments et leur index inverse.

    L'ordre est celui du dict `config.buildings.buildings` (insertion order YAML).
    Stable : le même config donne toujours le même ordre.

    Args:
        config: Configuration unifiée du jeu.

    Returns:
        Tuple (building_list, building_index_map) où :
        - building_list[i] = identifiant du bâtiment d'index i
        - building_index_map[building_id] = index du bâtiment
    """
    building_list = list(config.buildings.buildings.keys())
    assert len(building_list) == NUM_BUILDINGS, (
        f"Attendu {NUM_BUILDINGS} bâtiments, obtenu {len(building_list)}"
    )
    building_index_map = {bid: i for i, bid in enumerate(building_list)}
    return building_list, building_index_map


# ---------------------------------------------------------------------------
# Encodage / Décodage
# ---------------------------------------------------------------------------


def encode_action(action: Action, building_index_map: dict[str, int]) -> int:
    """Convertit une Action en entier discret.

    Args:
        action: Action à encoder.
        building_index_map: Map building_id -> index (depuis get_building_order).

    Returns:
        Entier dans [0, TOTAL_ACTIONS).

    Raises:
        ValueError: Si building_id absent ou None pour une action "place".
    """
    if action.type == "place":
        if action.building_id is None or action.building_id not in building_index_map:
            raise ValueError(
                f"building_id invalide pour place : {action.building_id!r}"
            )
        idx = building_index_map[action.building_id]
        return idx * CELLS + action.y * GRID_SIZE + action.x
    elif action.type == "demolish":
        return DEMOLISH_OFFSET + action.y * GRID_SIZE + action.x
    else:
        return DO_NOTHING


def decode_action(action_int: int, building_list: list[str]) -> Action:
    """Convertit un entier discret en Action.

    Args:
        action_int: Entier dans [0, TOTAL_ACTIONS).
        building_list: Liste ordonnée des bâtiments (depuis get_building_order).

    Returns:
        Action correspondante.

    Raises:
        ValueError: Si action_int hors bornes.
    """
    if action_int < 0 or action_int >= TOTAL_ACTIONS:
        raise ValueError(
            f"action_int hors bornes : {action_int} (doit être dans [0, {TOTAL_ACTIONS}))"
        )
    if action_int == DO_NOTHING:
        return Action("do_nothing")
    if action_int >= DEMOLISH_OFFSET:
        offset = action_int - DEMOLISH_OFFSET
        y, x = divmod(offset, GRID_SIZE)
        return Action("demolish", x=x, y=y)
    # Place
    building_idx, cell = divmod(action_int, CELLS)
    y, x = divmod(cell, GRID_SIZE)
    return Action("place", building_id=building_list[building_idx], x=x, y=y)


# ---------------------------------------------------------------------------
# Masque d'actions légales
# ---------------------------------------------------------------------------


def compute_action_mask(
    game_state: GameState,
    config: GameConfig,
    building_list: list[str],
) -> np.ndarray:
    """Calcule le masque booléen des actions légales pour un état donné.

    Utilisé par MaskablePPO (stable-baselines3) à chaque step.

    Args:
        game_state: État courant du jeu.
        config: Configuration unifiée du jeu.
        building_list: Liste ordonnée des bâtiments (depuis get_building_order).

    Returns:
        np.ndarray de forme (TOTAL_ACTIONS,) et dtype np.bool_.
        True = action légale, False = action illégale.
    """
    mask = np.zeros(TOTAL_ACTIONS, dtype=np.bool_)
    grid = game_state.grid
    rs = game_state.resource_state
    bldg = config.buildings.buildings

    # ------------------------------------------------------------------
    # PLACE actions (indices 0 à 20479)
    # ------------------------------------------------------------------
    for i, building_id in enumerate(building_list):
        cfg = bldg[building_id]
        base = i * CELLS

        # Optimisation 1 : bâtiment non abordable → skip les 1024 cellules
        if not can_afford(rs, cfg.cost):
            continue

        # Optimisation 2 : bâtiment unique déjà posé → skip
        if cfg.unique and grid._placed_ids[building_id] > 0:
            continue

        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                if grid.can_place(building_id, x, y, cfg):
                    mask[base + y * GRID_SIZE + x] = True

    # ------------------------------------------------------------------
    # DEMOLISH actions (indices 20480 à 21503)
    # ------------------------------------------------------------------
    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            if grid.get_building_at(x, y) is not None:
                mask[DEMOLISH_OFFSET + y * GRID_SIZE + x] = True

    # ------------------------------------------------------------------
    # DO_NOTHING (index 21504) — toujours légal
    # ------------------------------------------------------------------
    mask[DO_NOTHING] = True

    return mask
