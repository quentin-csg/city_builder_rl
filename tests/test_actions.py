"""Tests pour vitruvius.engine.actions."""

from __future__ import annotations

import numpy as np
import pytest

from vitruvius.config import load_config
from vitruvius.engine.actions import (
    CELLS,
    DEMOLISH_OFFSET,
    DO_NOTHING,
    GRID_SIZE,
    NUM_BUILDINGS,
    TOTAL_ACTIONS,
    compute_action_mask,
    decode_action,
    encode_action,
    get_building_order,
)
from vitruvius.engine.game_state import init_game_state
from vitruvius.engine.terrain import TerrainType
from vitruvius.engine.turn import Action, step


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def building_order(cfg):
    return get_building_order(cfg)


@pytest.fixture(scope="module")
def building_list(building_order):
    return building_order[0]


@pytest.fixture(scope="module")
def building_index_map(building_order):
    return building_order[1]


@pytest.fixture
def gs(cfg):
    return init_game_state(cfg, seed=42)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_water_tile(grid) -> tuple[int, int] | None:
    for y in range(grid.SIZE):
        for x in range(grid.SIZE):
            if grid.terrain[y][x] == TerrainType.WATER:
                return (x, y)
    return None


def _find_plain_tile(grid) -> tuple[int, int] | None:
    for y in range(grid.SIZE):
        for x in range(grid.SIZE):
            if grid.terrain[y][x] == TerrainType.PLAIN and grid._origin[y][x] is None:
                return (x, y)
    return None


# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------


def test_total_actions_value():
    assert TOTAL_ACTIONS == 21505


def test_demolish_offset_value():
    assert DEMOLISH_OFFSET == 20480


def test_do_nothing_value():
    assert DO_NOTHING == 21504


def test_cells_value():
    assert CELLS == GRID_SIZE * GRID_SIZE == 1024


# ---------------------------------------------------------------------------
# get_building_order
# ---------------------------------------------------------------------------


def test_building_order_length(building_list):
    assert len(building_list) == NUM_BUILDINGS == 20


def test_building_order_first_last(building_list, building_index_map):
    assert building_list[0] == "road"
    assert building_list[19] == "obelisk"
    assert building_index_map["road"] == 0
    assert building_index_map["obelisk"] == 19


def test_building_order_index_map_consistent(building_list, building_index_map):
    for i, bid in enumerate(building_list):
        assert building_index_map[bid] == i


# ---------------------------------------------------------------------------
# encode_action / decode_action
# ---------------------------------------------------------------------------


def test_encode_decode_place_origin(building_list, building_index_map):
    """road à (0,0) → 0."""
    action = Action("place", "road", 0, 0)
    encoded = encode_action(action, building_index_map)
    assert encoded == 0
    decoded = decode_action(encoded, building_list)
    assert decoded == action


def test_encode_decode_place_last(building_list, building_index_map):
    """obelisk à (31,31) → 20479."""
    action = Action("place", "obelisk", 31, 31)
    encoded = encode_action(action, building_index_map)
    assert encoded == 19 * 1024 + 31 * 32 + 31  # 20479
    decoded = decode_action(encoded, building_list)
    assert decoded == action


def test_encode_decode_place_arbitrary(building_list, building_index_map):
    """housing (index 4) à (7, 3) → 4*1024 + 3*32 + 7 = 4199."""
    action = Action("place", "housing", 7, 3)
    encoded = encode_action(action, building_index_map)
    assert encoded == 4 * 1024 + 3 * 32 + 7
    decoded = decode_action(encoded, building_list)
    assert decoded == action


def test_encode_decode_demolish_origin(building_list, building_index_map):
    """demolish (0,0) → 20480."""
    action = Action("demolish", x=0, y=0)
    encoded = encode_action(action, building_index_map)
    assert encoded == DEMOLISH_OFFSET
    decoded = decode_action(encoded, building_list)
    assert decoded == action


def test_encode_decode_demolish_corner(building_list, building_index_map):
    """demolish (31,31) → 21503."""
    action = Action("demolish", x=31, y=31)
    encoded = encode_action(action, building_index_map)
    assert encoded == 21503
    decoded = decode_action(encoded, building_list)
    assert decoded == action


def test_encode_decode_do_nothing(building_list, building_index_map):
    action = Action("do_nothing")
    encoded = encode_action(action, building_index_map)
    assert encoded == DO_NOTHING
    decoded = decode_action(encoded, building_list)
    assert decoded == action


def test_decode_invalid_negative(building_list):
    with pytest.raises(ValueError):
        decode_action(-1, building_list)


def test_decode_invalid_too_large(building_list):
    with pytest.raises(ValueError):
        decode_action(TOTAL_ACTIONS, building_list)


def test_encode_place_invalid_building_id(building_index_map):
    with pytest.raises(ValueError):
        encode_action(Action("place", "nonexistent", 0, 0), building_index_map)


def test_encode_place_none_building_id(building_index_map):
    with pytest.raises(ValueError):
        encode_action(Action("place", None, 0, 0), building_index_map)


# ---------------------------------------------------------------------------
# compute_action_mask
# ---------------------------------------------------------------------------


def test_mask_shape_dtype(cfg, gs, building_list):
    mask = compute_action_mask(gs, cfg, building_list)
    assert mask.shape == (TOTAL_ACTIONS,)
    assert mask.dtype == np.bool_


def test_mask_do_nothing_always_true(cfg, gs, building_list):
    mask = compute_action_mask(gs, cfg, building_list)
    assert mask[DO_NOTHING] is np.bool_(True)


def test_mask_demolish_empty_grid(cfg, gs, building_list):
    """Grille vide → aucune demolition possible."""
    mask = compute_action_mask(gs, cfg, building_list)
    demolish_slice = mask[DEMOLISH_OFFSET:DO_NOTHING]
    assert not demolish_slice.any()


def test_mask_demolish_occupied(cfg, gs, building_list, building_index_map):
    """Après placement d'une route, demolish à cet endroit est True."""
    pos = _find_plain_tile(gs.grid)
    assert pos is not None
    x, y = pos
    step(gs, cfg, Action("place", "road", x, y))

    mask = compute_action_mask(gs, cfg, building_list)
    demolish_idx = DEMOLISH_OFFSET + y * GRID_SIZE + x
    assert mask[demolish_idx] is np.bool_(True)


def test_mask_place_blocked_water(cfg, gs, building_list):
    """Tile water → tous les 20 bâtiments bloqués à cet endroit."""
    pos = _find_water_tile(gs.grid)
    if pos is None:
        pytest.skip("Pas de tile WATER sur ce seed")
    x, y = pos
    mask = compute_action_mask(gs, cfg, building_list)
    for i in range(NUM_BUILDINGS):
        idx = i * CELLS + y * GRID_SIZE + x
        assert mask[idx] is np.bool_(False), f"Bâtiment {i} placé sur eau en ({x},{y})"


def test_mask_place_blocked_no_resources(cfg, gs, building_list):
    """Ressources à 0 → aucun placement possible."""
    gs.resource_state.denarii = 0.0
    gs.resource_state.wheat = 0
    gs.resource_state.wood = 0
    gs.resource_state.marble = 0

    mask = compute_action_mask(gs, cfg, building_list)
    place_slice = mask[:DEMOLISH_OFFSET]
    assert not place_slice.any()


def test_mask_unique_already_placed(cfg, gs, building_list, building_index_map):
    """Obelisk déjà posé → ses 1024 actions de placement sont toutes False."""
    gs.resource_state.denarii = 5000.0  # obelisk coûte 1000
    pos = _find_plain_tile(gs.grid)
    assert pos is not None
    x, y = pos
    step(gs, cfg, Action("place", "obelisk", x, y))

    mask = compute_action_mask(gs, cfg, building_list)
    obelisk_idx = building_index_map["obelisk"]
    obelisk_slice = mask[obelisk_idx * CELLS: (obelisk_idx + 1) * CELLS]
    assert not obelisk_slice.any()


def test_mask_place_valid_cell(cfg, building_list):
    """État initial avec ressources → au moins une route plaçable."""
    gs = init_game_state(cfg, seed=42)
    mask = compute_action_mask(gs, cfg, building_list)
    road_idx = 0  # road = index 0
    road_slice = mask[road_idx * CELLS: (road_idx + 1) * CELLS]
    assert road_slice.any(), "Aucune route plaçable en début de partie"
