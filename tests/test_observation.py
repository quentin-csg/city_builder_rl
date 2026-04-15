"""Tests pour vitruvius/rl/observation.py."""

from __future__ import annotations

import numpy as np
import pytest

from vitruvius.config import load_config
from vitruvius.engine.actions import get_building_order
from vitruvius.engine.game_state import init_game_state
from vitruvius.engine.turn import Action, step
from vitruvius.rl.observation import _GRID_CHANNELS, _GLOBAL_FEATURES, build_observation


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def building_index_map(cfg):
    _, idx_map = get_building_order(cfg)
    return idx_map


@pytest.fixture
def gs(cfg):
    return init_game_state(cfg, seed=42)


def _find_plain(grid, w, h):
    for y in range(grid.SIZE - h + 1):
        for x in range(grid.SIZE - w + 1):
            if all(
                grid.terrain[y + dy][x + dx].name.lower() == "plain"
                and grid._origin[y + dy][x + dx] is None
                for dy in range(h)
                for dx in range(w)
            ):
                return x, y
    return None


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------


def test_obs_keys(cfg, gs, building_index_map):
    obs = build_observation(gs, cfg, building_index_map)
    assert set(obs.keys()) == {"grid", "global_features"}


def test_obs_shapes(cfg, gs, building_index_map):
    obs = build_observation(gs, cfg, building_index_map)
    assert obs["grid"].shape == (32, 32, _GRID_CHANNELS)
    assert obs["global_features"].shape == (_GLOBAL_FEATURES,)


def test_obs_shapes_constants():
    """Les constantes correspondent aux shapes attendues."""
    assert _GRID_CHANNELS == 31
    assert _GLOBAL_FEATURES == 18


def test_obs_dtype(cfg, gs, building_index_map):
    obs = build_observation(gs, cfg, building_index_map)
    assert obs["grid"].dtype == np.float32
    assert obs["global_features"].dtype == np.float32


# ---------------------------------------------------------------------------
# Canal 0 : terrain
# ---------------------------------------------------------------------------


def test_obs_terrain_channel(cfg, gs, building_index_map):
    """Les valeurs du canal 0 correspondent aux valeurs terrain attendues."""
    obs = build_observation(gs, cfg, building_index_map)
    grid = gs.grid
    expected = {"plain": 0.0, "forest": 0.25, "hill": 0.5, "water": 0.75, "marsh": 1.0}
    for y in range(32):
        for x in range(32):
            name = grid.terrain[y][x].name.lower()
            assert obs["grid"][y, x, 0] == pytest.approx(expected[name])


# ---------------------------------------------------------------------------
# Canaux 1-20 : one-hot type de bâtiment
# ---------------------------------------------------------------------------


def test_obs_building_channels_empty(cfg, gs, building_index_map):
    """Canaux 1-20 = 0 sur toute la grille vide."""
    obs = build_observation(gs, cfg, building_index_map)
    assert (obs["grid"][:, :, 1:21] == 0.0).all()


def test_obs_building_one_hot_after_place(cfg, gs, building_index_map):
    """Après placement d'une road, le canal correspondant vaut 1.0, les autres 0."""
    pos = _find_plain(gs.grid, 1, 1)
    assert pos is not None
    x, y = pos
    step(gs, cfg, Action("place", "road", x, y))
    obs = build_observation(gs, cfg, building_index_map)

    road_idx = building_index_map["road"]  # 0
    # Canal road doit être 1.0
    assert obs["grid"][y, x, 1 + road_idx] == pytest.approx(1.0)
    # Tous les autres canaux bâtiment doivent être 0
    for ch in range(1, 21):
        if ch != 1 + road_idx:
            assert obs["grid"][y, x, ch] == pytest.approx(0.0)


def test_obs_building_one_hot_sum(cfg, gs, building_index_map):
    """Sur chaque tile, la somme des canaux 1-20 vaut 0 (vide) ou 1 (occupé)."""
    obs = build_observation(gs, cfg, building_index_map)
    one_hot_sum = obs["grid"][:, :, 1:21].sum(axis=2)
    assert np.all((one_hot_sum == 0.0) | (one_hot_sum == 1.0))


# ---------------------------------------------------------------------------
# Canal 21 : niveau maison
# ---------------------------------------------------------------------------


def test_obs_house_level_channel(cfg, gs, building_index_map):
    """Canal 21 = level/6 sur le tile d'origine d'un housing."""
    pos = _find_plain(gs.grid, 2, 2)
    assert pos is not None
    x, y = pos
    step(gs, cfg, Action("place", "housing", x, y))
    obs = build_observation(gs, cfg, building_index_map)
    assert obs["grid"][y, x, 21] == pytest.approx(0.0)  # level=0 → 0/6

    gs.houses[(x, y)].level = 1
    obs = build_observation(gs, cfg, building_index_map)
    assert obs["grid"][y, x, 21] == pytest.approx(1 / 6.0)


# ---------------------------------------------------------------------------
# Canaux 22-27 : couverture services
# ---------------------------------------------------------------------------


def test_obs_coverage_channels_empty(cfg, gs, building_index_map):
    """Canaux 22-27 = 0 sur grille sans bâtiments de service."""
    obs = build_observation(gs, cfg, building_index_map)
    assert (obs["grid"][:, :, 22:28] == 0.0).all()


def test_obs_coverage_water_after_well(cfg, gs, building_index_map):
    """Canal 22 (water) = 1 dans le rayon d'un puits."""
    pos = _find_plain(gs.grid, 1, 1)
    assert pos is not None
    x, y = pos
    step(gs, cfg, Action("place", "well", x, y))
    obs = build_observation(gs, cfg, building_index_map)
    assert obs["grid"][y, x, 22] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Features globales
# ---------------------------------------------------------------------------


def test_obs_global_features_initial(cfg, building_index_map):
    """Vérifie quelques valeurs des features globales à l'état initial."""
    gs_fresh = init_game_state(cfg, seed=42)
    obs = build_observation(gs_fresh, cfg, building_index_map)
    gf = obs["global_features"]

    assert gf[0] == pytest.approx(1000.0 / 10_000.0, abs=1e-4)  # denarii
    assert gf[5] == pytest.approx(0.5, abs=1e-4)               # satisfaction
    assert gf[6] == pytest.approx(0.2, abs=1e-4)               # city_level/5
    assert gf[7] == pytest.approx(0.0, abs=1e-4)               # turn
    assert gf[13] == pytest.approx(0.0, abs=1e-4)              # drought inactive


def test_obs_global_features_victory_flags_initial(cfg, building_index_map):
    """Indices 15-17 = 0 sur grille vide (aucun bâtiment de victoire)."""
    gs_fresh = init_game_state(cfg, seed=42)
    obs = build_observation(gs_fresh, cfg, building_index_map)
    gf = obs["global_features"]
    assert gf[15] == pytest.approx(0.0)  # has_forum
    assert gf[16] == pytest.approx(0.0)  # has_obelisk
    assert gf[17] == pytest.approx(0.0)  # has_prefecture


def test_obs_global_features_forum_flag(cfg, building_index_map):
    """Indice 15 passe à 1.0 après placement d'un forum (si abordable)."""
    gs_fresh = init_game_state(cfg, seed=42)
    # Forcer le placement direct via placed_buildings pour éviter contraintes économiques
    from vitruvius.engine.grid import PlacedBuilding
    pos = _find_plain(gs_fresh.grid, 3, 3)
    if pos is None:
        pytest.skip("Pas assez de plaines libres 3×3 sur ce seed")
    x, y = pos
    gs_fresh.grid.placed_buildings[(x, y)] = PlacedBuilding("forum", x, y, (3, 3))
    gs_fresh.grid._placed_ids["forum"] += 1

    obs = build_observation(gs_fresh, cfg, building_index_map)
    assert obs["global_features"][15] == pytest.approx(1.0)


def test_obs_global_features_dynamics_none(cfg, building_index_map):
    """Avec dynamics=None, les indices 10/11/12 valent 0.0."""
    gs_fresh = init_game_state(cfg, seed=42)
    obs = build_observation(gs_fresh, cfg, building_index_map, dynamics=None)
    gf = obs["global_features"]
    assert gf[10] == pytest.approx(0.0)
    assert gf[11] == pytest.approx(0.0)
    assert gf[12] == pytest.approx(0.0)


def test_obs_global_features_clamping(cfg, building_index_map):
    """growth_rate > 1 est clampé à 1, net_income < -1 clampé à -1."""
    gs_fresh = init_game_state(cfg, seed=42)
    obs = build_observation(
        gs_fresh, cfg, building_index_map,
        dynamics={"growth_rate": 5.0, "wheat_conso_ratio": 0.5, "net_income": -3.0}
    )
    gf = obs["global_features"]
    assert gf[10] == pytest.approx(1.0)
    assert gf[12] == pytest.approx(-1.0)


def test_obs_is_pure(cfg, building_index_map):
    """build_observation ne modifie pas gs."""
    gs_fresh = init_game_state(cfg, seed=42)
    turn_before = gs_fresh.turn
    wheat_before = gs_fresh.resource_state.wheat
    build_observation(gs_fresh, cfg, building_index_map)
    assert gs_fresh.turn == turn_before
    assert gs_fresh.resource_state.wheat == wheat_before
