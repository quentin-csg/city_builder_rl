"""Tests pour vitruvius/rl/observation.py."""

from __future__ import annotations

import numpy as np
import pytest

from vitruvius.config import load_config
from vitruvius.engine.actions import get_building_order
from vitruvius.engine.game_state import init_game_state
from vitruvius.engine.turn import Action, step
from vitruvius.rl.observation import build_observation


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
    assert obs["grid"].shape == (32, 32, 12)
    assert obs["global_features"].shape == (15,)


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
# Canal 1 : type de bâtiment
# ---------------------------------------------------------------------------


def test_obs_building_channel_empty(cfg, gs, building_index_map):
    """Canal 1 = 0 sur toute la grille vide."""
    obs = build_observation(gs, cfg, building_index_map)
    assert (obs["grid"][:, :, 1] == 0.0).all()


def test_obs_building_channel_after_place(cfg, gs, building_index_map):
    """Canal 1 != 0 sur les tiles occupées par une road."""
    pos = _find_plain(gs.grid, 1, 1)
    assert pos is not None
    x, y = pos
    step(gs, cfg, Action("place", "road", x, y))
    obs = build_observation(gs, cfg, building_index_map)
    assert obs["grid"][y, x, 1] > 0.0


# ---------------------------------------------------------------------------
# Canal 2 : niveau maison
# ---------------------------------------------------------------------------


def test_obs_house_level_channel(cfg, gs, building_index_map):
    """Canal 2 = level/6 sur le tile d'origine d'un housing."""
    pos = _find_plain(gs.grid, 2, 2)
    assert pos is not None
    x, y = pos
    step(gs, cfg, Action("place", "housing", x, y))
    # housing posé à level=0 → 0.0
    obs = build_observation(gs, cfg, building_index_map)
    assert obs["grid"][y, x, 2] == pytest.approx(0.0)

    # Forcer level=1
    gs.houses[(x, y)].level = 1
    obs = build_observation(gs, cfg, building_index_map)
    assert obs["grid"][y, x, 2] == pytest.approx(1 / 6.0)


# ---------------------------------------------------------------------------
# Canaux 3-8 : couverture services
# ---------------------------------------------------------------------------


def test_obs_coverage_channels_empty(cfg, gs, building_index_map):
    """Canaux 3-8 = 0 sur grille sans bâtiments de service."""
    obs = build_observation(gs, cfg, building_index_map)
    assert (obs["grid"][:, :, 3:9] == 0.0).all()


def test_obs_coverage_water_after_well(cfg, gs, building_index_map):
    """Canal 3 (water) = 1 dans le rayon d'un puits."""
    pos = _find_plain(gs.grid, 1, 1)
    assert pos is not None
    x, y = pos
    step(gs, cfg, Action("place", "well", x, y))
    obs = build_observation(gs, cfg, building_index_map)
    # Au moins la tile du puits lui-même est couverte
    assert obs["grid"][y, x, 3] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Features globales
# ---------------------------------------------------------------------------


def test_obs_global_features_initial(cfg, building_index_map):
    """Vérifie quelques valeurs des features globales à l'état initial."""
    gs_fresh = init_game_state(cfg, seed=42)
    obs = build_observation(gs_fresh, cfg, building_index_map)
    gf = obs["global_features"]

    # denarii=800 / 10000 = 0.08
    assert gf[0] == pytest.approx(800.0 / 10_000.0, abs=1e-4)
    # satisfaction = 0.5
    assert gf[5] == pytest.approx(0.5, abs=1e-4)
    # city_level=1 / 5 = 0.2
    assert gf[6] == pytest.approx(0.2, abs=1e-4)
    # turn=0
    assert gf[7] == pytest.approx(0.0, abs=1e-4)
    # drought inactive
    assert gf[13] == pytest.approx(0.0, abs=1e-4)


def test_obs_global_features_dynamics_none(cfg, building_index_map):
    """Avec dynamics=None, les indices 10/11/12 valent 0.0."""
    gs_fresh = init_game_state(cfg, seed=42)
    obs = build_observation(gs_fresh, cfg, building_index_map, dynamics=None)
    gf = obs["global_features"]
    assert gf[10] == pytest.approx(0.0)
    assert gf[11] == pytest.approx(0.0)
    assert gf[12] == pytest.approx(0.0)


def test_obs_global_features_clamping(cfg, building_index_map):
    """growth_rate > 1 est clampé à 1, < -1 clampé à -1."""
    gs_fresh = init_game_state(cfg, seed=42)
    obs = build_observation(
        gs_fresh, cfg, building_index_map,
        dynamics={"growth_rate": 5.0, "wheat_conso_ratio": 0.5, "net_income": -3.0}
    )
    gf = obs["global_features"]
    assert gf[10] == pytest.approx(1.0)   # clampé
    assert gf[12] == pytest.approx(-1.0)  # clampé


def test_obs_is_pure(cfg, building_index_map):
    """build_observation ne modifie pas gs."""
    gs_fresh = init_game_state(cfg, seed=42)
    turn_before = gs_fresh.turn
    wheat_before = gs_fresh.resource_state.wheat
    build_observation(gs_fresh, cfg, building_index_map)
    assert gs_fresh.turn == turn_before
    assert gs_fresh.resource_state.wheat == wheat_before
