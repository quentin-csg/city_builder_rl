"""Tests pour vitruvius.engine.population."""

import math
import pytest

from vitruvius.config import load_config
from vitruvius.engine.buildings import BuildingConfig, try_place_building
from vitruvius.engine.grid import Grid
from vitruvius.engine.population import (
    HouseState,
    HouseLevelConfig,
    apply_exodus,
    apply_famine_loss,
    apply_growth,
    apply_immigration,
    compute_global_satisfaction,
    compute_house_satisfaction,
    compute_house_taxes,
    evolve_houses,
    init_houses,
)
from vitruvius.engine.resources import ResourceState
from vitruvius.engine.terrain import TerrainType


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def bldg(cfg):
    return cfg.buildings.buildings


@pytest.fixture(scope="module")
def house_levels(cfg):
    return cfg.needs.house_levels


@pytest.fixture
def grid():
    return Grid(seed=42)


@pytest.fixture
def state():
    return ResourceState(denarii=5000.0, wheat=200, wood=500, marble=500)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_plain_spot(grid: Grid, w: int, h: int) -> tuple[int, int]:
    """Trouve le premier spot PLAIN libre pour un bâtiment w×h."""
    for oy in range(grid.SIZE - h + 1):
        for ox in range(grid.SIZE - w + 1):
            if all(
                grid.terrain[oy + dy][ox + dx] == TerrainType.PLAIN
                and grid._origin[oy + dy][ox + dx] is None
                for dy in range(h)
                for dx in range(w)
            ):
                return ox, oy
    raise RuntimeError(f"Pas de spot PLAIN libre pour {w}×{h}")


def _place_housing(grid: Grid, bldg: dict, state: ResourceState) -> tuple[int, int]:
    """Place un housing sur la grille et retourne son origin."""
    ox, oy = _find_plain_spot(grid, 2, 2)
    assert try_place_building(grid, state, "housing", ox, oy, bldg)
    return ox, oy


def _make_house(level: int = 1, population: int = 3, origin: tuple = (0, 0)) -> HouseState:
    return HouseState(origin=origin, level=level, population=population)


# ---------------------------------------------------------------------------
# HouseState
# ---------------------------------------------------------------------------


def test_house_state_defaults():
    h = HouseState(origin=(2, 3), level=0, population=0)
    assert h.origin == (2, 3)
    assert h.level == 0
    assert h.population == 0
    assert h.famine is False


# ---------------------------------------------------------------------------
# init_houses
# ---------------------------------------------------------------------------


def test_init_houses_empty_grid(bldg, house_levels):
    g = Grid(seed=42)
    state = ResourceState(denarii=5000.0, wheat=0, wood=0, marble=0)
    houses = init_houses(g, bldg)
    assert houses == {}


def test_init_houses_creates_entries_for_each_housing(bldg):
    g = Grid(seed=42)
    state = ResourceState(denarii=5000.0, wheat=0, wood=500, marble=0)
    ox1, oy1 = _find_plain_spot(g, 2, 2)
    assert try_place_building(g, state, "housing", ox1, oy1, bldg)
    # Place a second one
    state2 = ResourceState(denarii=5000.0, wheat=0, wood=500, marble=0)
    ox2, oy2 = _find_plain_spot(g, 2, 2)
    assert try_place_building(g, state2, "housing", ox2, oy2, bldg)

    houses = init_houses(g, bldg)
    assert len(houses) == 2
    for house in houses.values():
        assert house.level == 0
        assert house.population == 0
        assert house.famine is False


def test_init_houses_ignores_non_housing(bldg, grid, state):
    # Place non-housing buildings, check they don't appear
    ox, oy = _find_plain_spot(grid, 1, 1)
    assert try_place_building(grid, state, "well", ox, oy, bldg)
    houses = init_houses(grid, bldg)
    assert all(
        grid.placed_buildings[origin].building_id == "housing"
        for origin in houses
    )


# ---------------------------------------------------------------------------
# compute_house_satisfaction
# ---------------------------------------------------------------------------


def test_satisfaction_level_0_returns_0(house_levels, grid):
    h = HouseState(origin=(0, 0), level=0, population=0)
    assert compute_house_satisfaction(h, {"water"}, house_levels, grid) == 0.0


def test_satisfaction_all_needs_met(bldg, house_levels):
    g = Grid(seed=42)
    state = ResourceState(denarii=5000.0, wheat=0, wood=500, marble=0)
    ox, oy = _place_housing(g, bldg, state)
    h = HouseState(origin=(ox, oy), level=1, population=3)  # niveau 1 = tente, besoin: water
    sat = compute_house_satisfaction(h, {"water"}, house_levels, g)
    assert sat == pytest.approx(1.0)


def test_satisfaction_partial_needs(bldg, house_levels):
    g = Grid(seed=42)
    state = ResourceState(denarii=5000.0, wheat=0, wood=500, marble=0)
    ox, oy = _place_housing(g, bldg, state)
    h = HouseState(origin=(ox, oy), level=2, population=5)  # niveau 2 = hutte, besoins: water+food
    sat = compute_house_satisfaction(h, {"water"}, house_levels, g)  # food manquant
    assert sat == pytest.approx(0.5)


def test_satisfaction_no_needs_met(bldg, house_levels):
    g = Grid(seed=42)
    state = ResourceState(denarii=5000.0, wheat=0, wood=500, marble=0)
    ox, oy = _place_housing(g, bldg, state)
    h = HouseState(origin=(ox, oy), level=1, population=3)
    sat = compute_house_satisfaction(h, set(), house_levels, g)
    assert sat == pytest.approx(0.0)


def test_satisfaction_road_bonus(bldg, house_levels):
    g = Grid(seed=42)
    state = ResourceState(denarii=5000.0, wheat=0, wood=500, marble=0)
    ox, oy = _place_housing(g, bldg, state)
    # Niveau 2 (hutte) besoins: water+food, seulement water → base=0.5
    h = HouseState(origin=(ox, oy), level=2, population=5)

    # Satisfaction sans route
    sat_no_road = compute_house_satisfaction(h, {"water"}, house_levels, g)
    assert sat_no_road == pytest.approx(0.5)

    # Placer une route adjacente (à droite du housing 2×2)
    road_x, road_y = ox + 2, oy
    if 0 <= road_x < g.SIZE and g._origin[road_y][road_x] is None and g.terrain[road_y][road_x] != TerrainType.WATER:
        assert try_place_building(g, state, "road", road_x, road_y, bldg)
        sat_with_road = compute_house_satisfaction(h, {"water"}, house_levels, g)
        assert sat_with_road == pytest.approx(0.5 + 0.03)


def test_satisfaction_capped_at_1(bldg, house_levels):
    g = Grid(seed=42)
    state = ResourceState(denarii=5000.0, wheat=0, wood=500, marble=0)
    ox, oy = _place_housing(g, bldg, state)
    h = HouseState(origin=(ox, oy), level=1, population=3)
    # Placer plusieurs routes adjacentes
    for dx, dy in [(2, 0), (-1, 0), (0, 2), (0, -1)]:
        rx, ry = ox + dx, oy + dy
        if 0 <= rx < g.SIZE and 0 <= ry < g.SIZE and g._origin[ry][rx] is None:
            if g.terrain[ry][rx] != TerrainType.WATER:
                try_place_building(g, state, "road", rx, ry, bldg)
    sat = compute_house_satisfaction(h, {"water"}, house_levels, g)
    assert sat <= 1.0


# ---------------------------------------------------------------------------
# compute_global_satisfaction
# ---------------------------------------------------------------------------


def test_global_satisfaction_no_pop_returns_half(bldg, house_levels):
    g = Grid(seed=42)
    state = ResourceState(denarii=5000.0, wheat=0, wood=500, marble=0)
    ox, oy = _place_housing(g, bldg, state)
    houses = {(ox, oy): HouseState(origin=(ox, oy), level=0, population=0)}
    coverage = {}
    sat = compute_global_satisfaction(houses, coverage, house_levels, g)
    assert sat == pytest.approx(0.5)


def test_global_satisfaction_weighted(bldg, house_levels):
    g = Grid(seed=42)
    state = ResourceState(denarii=5000.0, wheat=0, wood=500, marble=0)
    ox1, oy1 = _place_housing(g, bldg, state)
    ox2, oy2 = _place_housing(g, bldg, state)

    # Maison 1 : level 1, pop=10, water couvert → sat=1.0
    # Maison 2 : level 1, pop=10, pas water → sat=0.0
    houses = {
        (ox1, oy1): HouseState(origin=(ox1, oy1), level=1, population=10),
        (ox2, oy2): HouseState(origin=(ox2, oy2), level=1, population=10),
    }
    coverage = {(ox1, oy1): {"water"}}
    sat = compute_global_satisfaction(houses, coverage, house_levels, g)
    # (1.0 × 10 + 0.0 × 10) / 20 = 0.5
    assert sat == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# compute_house_taxes
# ---------------------------------------------------------------------------


def test_compute_house_taxes_floor(house_levels):
    # Niveau 1 (tente) : tax=0.2, pop=7 → floor(7×0.2) = floor(1.4) = 1
    h = HouseState(origin=(0, 0), level=1, population=7)
    houses = {(0, 0): h}
    taxes = compute_house_taxes(houses, house_levels)
    assert taxes == [math.floor(7 * 0.2)]


def test_compute_house_taxes_level_0_excluded(house_levels):
    h = HouseState(origin=(0, 0), level=0, population=0)
    houses = {(0, 0): h}
    taxes = compute_house_taxes(houses, house_levels)
    assert taxes == [0.0]


def test_compute_house_taxes_multiple_levels(house_levels):
    # Niveau 1 (0.2) + niveau 3 (0.3)
    houses = {
        (0, 0): HouseState(origin=(0, 0), level=1, population=5),
        (2, 2): HouseState(origin=(2, 2), level=3, population=15),
    }
    taxes = compute_house_taxes(houses, house_levels)
    expected = [math.floor(5 * 0.2), math.floor(15 * 0.3)]
    assert taxes == expected


# ---------------------------------------------------------------------------
# apply_famine_loss
# ---------------------------------------------------------------------------


def test_apply_famine_loss_ceil_10_percent():
    h = HouseState(origin=(0, 0), level=1, population=10, famine=True)
    houses = {(0, 0): h}
    lost = apply_famine_loss(houses)
    assert lost == 1  # ceil(10 * 0.10) = 1
    assert h.population == 9


def test_apply_famine_loss_ceil_rounds_up():
    h = HouseState(origin=(0, 0), level=1, population=7, famine=True)
    houses = {(0, 0): h}
    lost = apply_famine_loss(houses)
    assert lost == math.ceil(7 * 0.10)  # ceil(0.7) = 1
    assert h.population == 7 - lost


def test_apply_famine_loss_resets_flag():
    h = HouseState(origin=(0, 0), level=1, population=10, famine=True)
    houses = {(0, 0): h}
    apply_famine_loss(houses)
    assert h.famine is False


def test_apply_famine_loss_no_famine_unchanged():
    h = HouseState(origin=(0, 0), level=1, population=10, famine=False)
    houses = {(0, 0): h}
    lost = apply_famine_loss(houses)
    assert lost == 0
    assert h.population == 10


# ---------------------------------------------------------------------------
# evolve_houses
# ---------------------------------------------------------------------------


def test_evolve_level1_to_2_when_needs_met(house_levels):
    # Niveau 1 (tente) besoins: [water] → niveau 2 (hutte) besoins: [water, food]
    h = HouseState(origin=(0, 0), level=1, population=3)
    houses = {(0, 0): h}
    # Couvrir water ET food (besoins de niveau 2)
    coverage = {(0, 0): {"water", "food"}}
    evolved, regressed = evolve_houses(houses, coverage, house_levels)
    assert evolved == 1
    assert regressed == 0
    assert h.level == 2


def test_evolve_level0_unchanged(house_levels):
    h = HouseState(origin=(0, 0), level=0, population=0)
    houses = {(0, 0): h}
    evolved, regressed = evolve_houses(houses, {(0, 0): {"water"}}, house_levels)
    assert evolved == 0
    assert regressed == 0
    assert h.level == 0


def test_evolve_no_evolution_at_level6(house_levels):
    h = HouseState(origin=(0, 0), level=6, population=50)
    houses = {(0, 0): h}
    all_needs = {"water", "food", "religion", "hygiene", "entertainment", "security"}
    coverage = {(0, 0): all_needs}
    evolved, regressed = evolve_houses(houses, coverage, house_levels)
    assert evolved == 0
    assert h.level == 6


def test_evolve_regression_level2_to_1(house_levels):
    # Niveau 2 (hutte) besoins: [water, food] — perd food
    h = HouseState(origin=(0, 0), level=2, population=8)
    houses = {(0, 0): h}
    coverage = {(0, 0): {"water"}}  # food manquant
    evolved, regressed = evolve_houses(houses, coverage, house_levels)
    assert regressed == 1
    assert h.level == 1
    # pop_max niveau 1 = 5, pop initiale=8 → clampée à 5
    assert h.population == 5


def test_evolve_regression_tent_loses_water(house_levels):
    # Tente (level 1) perd water → level 0, tous perdus
    h = HouseState(origin=(0, 0), level=1, population=4)
    houses = {(0, 0): h}
    coverage = {(0, 0): set()}  # aucun service
    evolved, regressed = evolve_houses(houses, coverage, house_levels)
    assert regressed == 1
    assert h.level == 0
    assert h.population == 0


def test_evolve_regression_excess_pop_clipped(house_levels):
    # Niveau 3 (casa, pop_max=20) régresse vers niveau 2 (hutte, pop_max=10)
    h = HouseState(origin=(0, 0), level=3, population=18)
    houses = {(0, 0): h}
    coverage = {(0, 0): {"water"}}  # food et religion manquants
    evolve_houses(houses, coverage, house_levels)
    assert h.level == 2
    assert h.population == min(18, 10)  # clampé à pop_max de niveau 2


# ---------------------------------------------------------------------------
# apply_growth
# ---------------------------------------------------------------------------


def test_apply_growth_adds_pop(house_levels):
    h = HouseState(origin=(0, 0), level=1, population=0)
    houses = {(0, 0): h}
    added = apply_growth(houses, global_satisfaction=0.8, house_levels=house_levels)
    assert added > 0
    assert h.population == added


def test_apply_growth_no_growth_below_half(house_levels):
    h = HouseState(origin=(0, 0), level=1, population=2)
    houses = {(0, 0): h}
    added = apply_growth(houses, global_satisfaction=0.4, house_levels=house_levels)
    assert added == 0
    assert h.population == 2


def test_apply_growth_capped_at_pop_max(house_levels):
    # pop_max niveau 1 (tente) = 5
    h = HouseState(origin=(0, 0), level=1, population=4)
    houses = {(0, 0): h}
    apply_growth(houses, global_satisfaction=1.0, house_levels=house_levels)
    assert h.population <= 5


def test_apply_growth_skips_level_0(house_levels):
    h = HouseState(origin=(0, 0), level=0, population=0)
    houses = {(0, 0): h}
    added = apply_growth(houses, global_satisfaction=0.9, house_levels=house_levels)
    assert added == 0
    assert h.level == 0


# ---------------------------------------------------------------------------
# apply_exodus
# ---------------------------------------------------------------------------


def test_apply_exodus_triggers_below_30_percent(house_levels):
    h = HouseState(origin=(0, 0), level=2, population=100)
    houses = {(0, 0): h}
    lost = apply_exodus(houses, global_satisfaction=0.2)
    assert lost > 0
    assert h.population < 100


def test_apply_exodus_no_exodus_above_30(house_levels):
    h = HouseState(origin=(0, 0), level=2, population=100)
    houses = {(0, 0): h}
    lost = apply_exodus(houses, global_satisfaction=0.5)
    assert lost == 0
    assert h.population == 100


def test_apply_exodus_loses_5_percent(house_levels):
    h = HouseState(origin=(0, 0), level=2, population=100)
    houses = {(0, 0): h}
    lost = apply_exodus(houses, global_satisfaction=0.1)
    assert lost == math.ceil(100 * 0.05)  # ceil(5) = 5


def test_apply_exodus_no_negative_pop():
    h = HouseState(origin=(0, 0), level=1, population=1)
    houses = {(0, 0): h}
    apply_exodus(houses, global_satisfaction=0.0)
    assert h.population >= 0


# ---------------------------------------------------------------------------
# apply_immigration
# ---------------------------------------------------------------------------


def test_apply_immigration_fills_houses(house_levels):
    h = HouseState(origin=(0, 0), level=1, population=0)
    houses = {(0, 0): h}
    settled = apply_immigration(houses, amount=3, house_levels=house_levels)
    assert settled == 3
    assert h.population == 3


def test_apply_immigration_upgrades_level0(house_levels):
    h = HouseState(origin=(0, 0), level=0, population=0)
    houses = {(0, 0): h}
    settled = apply_immigration(houses, amount=2, house_levels=house_levels)
    assert h.level == 1
    assert settled == 2
    assert h.population == 2


def test_apply_immigration_returns_settled_when_full(house_levels):
    # pop_max niveau 1 (tente) = 5, maison déjà à 4
    h = HouseState(origin=(0, 0), level=1, population=4)
    houses = {(0, 0): h}
    settled = apply_immigration(houses, amount=10, house_levels=house_levels)
    assert settled == 1  # seulement 1 place restante
    assert h.population == 5


def test_apply_immigration_returns_0_when_no_space(house_levels):
    h = HouseState(origin=(0, 0), level=1, population=5)  # plein
    houses = {(0, 0): h}
    settled = apply_immigration(houses, amount=5, house_levels=house_levels)
    assert settled == 0


# ---------------------------------------------------------------------------
# Test d'intégration
# ---------------------------------------------------------------------------


def test_full_cycle(bldg, house_levels):
    """init → immigration → evolve → growth → taxes."""
    g = Grid(seed=42)
    state = ResourceState(denarii=5000.0, wheat=200, wood=500, marble=500)
    ox, oy = _place_housing(g, bldg, state)
    houses = init_houses(g, bldg)

    assert (ox, oy) in houses
    assert houses[(ox, oy)].level == 0

    # Immigration initiale
    apply_immigration(houses, amount=3, house_levels=house_levels)
    assert houses[(ox, oy)].level == 1
    assert houses[(ox, oy)].population == 3

    # Évolution si water couvert
    coverage = {(ox, oy): {"water", "food"}}
    evolve_houses(houses, coverage, house_levels)
    assert houses[(ox, oy)].level == 2

    # Croissance
    apply_growth(houses, global_satisfaction=0.8, house_levels=house_levels)
    assert houses[(ox, oy)].population >= 3

    # Taxes cohérentes (floor)
    taxes = compute_house_taxes(houses, house_levels)
    assert all(t >= 0 for t in taxes)
