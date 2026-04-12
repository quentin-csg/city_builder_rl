"""Tests pour vitruvius.engine.buildings (runtime)."""

import pytest

from vitruvius.config import load_config
from vitruvius.engine.buildings import (
    get_connected_aqueducts,
    get_functional_fountains,
    is_aqueduct_connected,
    is_fountain_functional,
    try_demolish,
    try_place_building,
)
from vitruvius.engine.grid import Grid, PlacedBuilding
from vitruvius.engine.resources import (
    ResourceState,
    clamp_stocks_to_capacity,
    refund_cost,
)
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


@pytest.fixture
def grid():
    return Grid(seed=42)


@pytest.fixture
def state():
    return ResourceState(denarii=5000.0, wheat=0, wood=200, marble=500)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_plain_spot(grid: Grid, w: int, h: int) -> tuple[int, int]:
    """Trouve une zone w×h entièrement PLAIN et vide."""
    size = Grid.SIZE
    for y in range(size - h + 1):
        for x in range(size - w + 1):
            if all(
                grid.terrain[y + dy][x + dx] == TerrainType.PLAIN
                and grid._origin[y + dy][x + dx] is None
                for dy in range(h)
                for dx in range(w)
            ):
                return x, y
    raise RuntimeError("Pas de zone PLAIN disponible")


def _find_water_adjacent_plain(grid: Grid) -> tuple[int, int]:
    """Trouve une tile PLAIN adjacente (4-connexité) à une tile WATER, vide."""
    size = Grid.SIZE
    for y in range(size):
        for x in range(size):
            if grid.terrain[y][x] == TerrainType.PLAIN and grid._origin[y][x] is None:
                for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                    if 0 <= nx < size and 0 <= ny < size and grid.terrain[ny][nx] == TerrainType.WATER:
                        return x, y
    raise RuntimeError("Pas de tile PLAIN adjacente à WATER")


def _find_plain_far_from_water(grid: Grid) -> tuple[int, int]:
    """Trouve une tile PLAIN sans aucun voisin WATER dans un rayon de 3 tiles."""
    size = Grid.SIZE
    for y in range(size):
        for x in range(size):
            if grid.terrain[y][x] != TerrainType.PLAIN or grid._origin[y][x] is not None:
                continue
            far = True
            for dy in range(-3, 4):
                for dx in range(-3, 4):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < size and 0 <= ny < size and grid.terrain[ny][nx] == TerrainType.WATER:
                        far = False
                        break
                if not far:
                    break
            if far:
                return x, y
    raise RuntimeError("Pas de tile PLAIN éloignée de WATER")


# ---------------------------------------------------------------------------
# refund_cost
# ---------------------------------------------------------------------------


def test_refund_denarii_only():
    state = ResourceState(denarii=0.0, wheat=0, wood=0, marble=0)
    refund_cost(state, {"denarii": 100})
    assert state.denarii == 50.0


def test_refund_multi_resources():
    state = ResourceState(denarii=0.0, wheat=0, wood=0, marble=0)
    refund_cost(state, {"denarii": 200, "marble": 100})
    assert state.denarii == 100.0
    assert state.marble == 50


def test_refund_odd_floor():
    state = ResourceState(denarii=0.0, wheat=0, wood=0, marble=0)
    refund_cost(state, {"denarii": 31})
    assert state.denarii == 15.0  # floor(15.5) = 15


def test_refund_wood_only():
    state = ResourceState(denarii=0.0, wheat=0, wood=0, marble=0)
    refund_cost(state, {"wood": 10})
    assert state.wood == 5


def test_refund_custom_ratio():
    state = ResourceState(denarii=0.0, wheat=0, wood=0, marble=0)
    refund_cost(state, {"denarii": 100}, ratio=0.75)
    assert state.denarii == 75.0


# ---------------------------------------------------------------------------
# clamp_stocks_to_capacity
# ---------------------------------------------------------------------------


def test_clamp_within_cap(bldg):
    state = ResourceState(denarii=0.0, wheat=100, wood=0, marble=0)
    placed = {(0, 0): PlacedBuilding(building_id="granary", x=0, y=0, size=(3, 3))}
    lost = clamp_stocks_to_capacity(state, placed, bldg)
    assert state.wheat == 100
    assert lost == {}


def test_clamp_no_storage(bldg):
    state = ResourceState(denarii=0.0, wheat=200, wood=0, marble=0)
    placed = {}
    lost = clamp_stocks_to_capacity(state, placed, bldg)
    assert state.wheat == 0
    assert lost == {"wheat": 200}


def test_clamp_partial(bldg):
    state = ResourceState(denarii=0.0, wheat=3000, wood=0, marble=0)
    placed = {(0, 0): PlacedBuilding(building_id="granary", x=0, y=0, size=(3, 3))}
    lost = clamp_stocks_to_capacity(state, placed, bldg)
    assert state.wheat == 2400
    assert lost == {"wheat": 600}


def test_clamp_denarii_unaffected(bldg):
    state = ResourceState(denarii=999999.0, wheat=0, wood=0, marble=0)
    placed = {}
    lost = clamp_stocks_to_capacity(state, placed, bldg)
    assert state.denarii == 999999.0
    assert "denarii" not in lost


def test_clamp_multiple_resources(bldg):
    state = ResourceState(denarii=0.0, wheat=5000, wood=5000, marble=0)
    placed = {
        (0, 0): PlacedBuilding(building_id="granary", x=0, y=0, size=(3, 3)),
        (3, 0): PlacedBuilding(building_id="warehouse_wood", x=3, y=0, size=(3, 3)),
    }
    lost = clamp_stocks_to_capacity(state, placed, bldg)
    assert state.wheat == 2400
    assert state.wood == 3200
    assert lost == {"wheat": 2600, "wood": 1800}


# ---------------------------------------------------------------------------
# try_place_building
# ---------------------------------------------------------------------------


def test_place_road_success(grid, state, bldg):
    x, y = _find_plain_spot(grid, 1, 1)
    denarii_before = state.denarii
    result = try_place_building(grid, state, "road", x, y, bldg)
    assert result is True
    assert grid.get_building_at(x, y) is not None
    assert state.denarii == denarii_before - 2


def test_place_no_money(grid, bldg):
    s = ResourceState(denarii=0.0, wheat=0, wood=0, marble=0)
    x, y = _find_plain_spot(grid, 1, 1)
    result = try_place_building(grid, s, "road", x, y, bldg)
    assert result is False
    assert grid.get_building_at(x, y) is None


def test_place_housing_wood(grid, bldg):
    s = ResourceState(denarii=5000.0, wheat=0, wood=10, marble=0)
    x, y = _find_plain_spot(grid, 2, 2)
    denarii_before = s.denarii
    result = try_place_building(grid, s, "housing", x, y, bldg)
    assert result is True
    assert s.wood == 0
    assert s.denarii == denarii_before  # denarii inchangés


def test_place_on_water(grid, state, bldg):
    size = Grid.SIZE
    wx, wy = next(
        (x, y)
        for y in range(size)
        for x in range(size)
        if grid.terrain[y][x] == TerrainType.WATER
    )
    result = try_place_building(grid, state, "road", wx, wy, bldg)
    assert result is False


def test_place_overlap(grid, bldg):
    s = ResourceState(denarii=5000.0, wheat=0, wood=0, marble=0)
    x, y = _find_plain_spot(grid, 1, 1)
    try_place_building(grid, s, "road", x, y, bldg)
    denarii_after_first = s.denarii
    result = try_place_building(grid, s, "road", x, y, bldg)
    assert result is False
    assert s.denarii == denarii_after_first


def test_place_unique_second(grid, bldg):
    s = ResourceState(denarii=50000.0, wheat=0, wood=0, marble=1000)
    x1, y1 = _find_plain_spot(grid, 4, 4)
    try_place_building(grid, s, "forum", x1, y1, bldg)
    x2, y2 = _find_plain_spot(grid, 4, 4)
    result = try_place_building(grid, s, "forum", x2, y2, bldg)
    assert result is False


def test_place_multi_resource(grid, bldg):
    s = ResourceState(denarii=5000.0, wheat=0, wood=0, marble=500)
    x, y = _find_plain_spot(grid, 3, 3)
    result = try_place_building(grid, s, "temple", x, y, bldg)
    assert result is True
    assert s.denarii == 5000.0 - 800
    assert s.marble == 500 - 100


def test_place_insufficient_one_resource(grid, bldg):
    s = ResourceState(denarii=5000.0, wheat=0, wood=0, marble=50)
    x, y = _find_plain_spot(grid, 3, 3)
    result = try_place_building(grid, s, "temple", x, y, bldg)
    assert result is False
    assert s.denarii == 5000.0
    assert s.marble == 50


# ---------------------------------------------------------------------------
# try_demolish
# ---------------------------------------------------------------------------


def test_demolish_road_refund(grid, bldg):
    s = ResourceState(denarii=5000.0, wheat=0, wood=0, marble=0)
    x, y = _find_plain_spot(grid, 1, 1)
    grid.place_building("road", x, y, bldg["road"])
    denarii_before = s.denarii
    pb = try_demolish(grid, s, x, y, bldg)
    assert pb is not None
    assert pb.building_id == "road"
    assert grid.get_building_at(x, y) is None
    assert s.denarii == denarii_before + 1  # road coûte 2, rembourse 1


def test_demolish_empty_none(grid, bldg):
    s = ResourceState(denarii=5000.0, wheat=0, wood=0, marble=0)
    x, y = _find_plain_spot(grid, 1, 1)
    result = try_demolish(grid, s, x, y, bldg)
    assert result is None


def test_demolish_housing_wood(grid, bldg):
    s = ResourceState(denarii=5000.0, wheat=0, wood=0, marble=0)
    x, y = _find_plain_spot(grid, 2, 2)
    grid.place_building("housing", x, y, bldg["housing"])
    pb = try_demolish(grid, s, x, y, bldg)
    assert pb is not None
    assert s.wood == 5  # housing coûte 10 wood, rembourse 5


def test_demolish_granary_clamp(grid, bldg):
    s = ResourceState(denarii=5000.0, wheat=1000, wood=0, marble=0)
    x, y = _find_plain_spot(grid, 3, 3)
    grid.place_building("granary", x, y, bldg["granary"])
    try_demolish(grid, s, x, y, bldg)
    assert s.wheat == 0  # cap = 0 → excédent perdu


def test_demolish_one_of_two_granaries(grid, bldg):
    s = ResourceState(denarii=5000.0, wheat=3000, wood=0, marble=0)
    x1, y1 = _find_plain_spot(grid, 3, 3)
    grid.place_building("granary", x1, y1, bldg["granary"])
    x2, y2 = _find_plain_spot(grid, 3, 3)
    grid.place_building("granary", x2, y2, bldg["granary"])
    try_demolish(grid, s, x1, y1, bldg)
    assert s.wheat == 2400  # cap restante = 2400


def test_demolish_refund_multi(grid, bldg):
    s = ResourceState(denarii=0.0, wheat=0, wood=0, marble=0)
    x, y = _find_plain_spot(grid, 4, 4)
    grid.place_building("forum", x, y, bldg["forum"])
    pb = try_demolish(grid, s, x, y, bldg)
    assert pb is not None
    assert s.denarii == 1000.0   # floor(2000 * 0.5)
    assert s.marble == 50        # floor(100 * 0.5)


# ---------------------------------------------------------------------------
# get_connected_aqueducts
# ---------------------------------------------------------------------------


def test_connected_adjacent_water(grid, bldg):
    x, y = _find_water_adjacent_plain(grid)
    grid.place_building("aqueduct", x, y, bldg["aqueduct"])
    connected = get_connected_aqueducts(grid, bldg)
    assert (x, y) in connected


def test_disconnected(grid, bldg):
    x, y = _find_plain_far_from_water(grid)
    grid.place_building("aqueduct", x, y, bldg["aqueduct"])
    connected = get_connected_aqueducts(grid, bldg)
    assert (x, y) not in connected


def test_chain(grid, bldg):
    # A adjacent à WATER, B adjacent à A mais pas à WATER → les deux connectés
    ax, ay = _find_water_adjacent_plain(grid)
    grid.place_building("aqueduct", ax, ay, bldg["aqueduct"])

    size = Grid.SIZE
    bx, by = None, None
    for nx, ny in ((ax - 1, ay), (ax + 1, ay), (ax, ay - 1), (ax, ay + 1)):
        if (
            0 <= nx < size
            and 0 <= ny < size
            and grid.terrain[ny][nx] == TerrainType.PLAIN
            and grid._origin[ny][nx] is None
        ):
            bx, by = nx, ny
            break
    assert bx is not None, "Pas de voisin PLAIN de A disponible"

    grid.place_building("aqueduct", bx, by, bldg["aqueduct"])
    connected = get_connected_aqueducts(grid, bldg)
    assert (ax, ay) in connected
    assert (bx, by) in connected


def test_no_aqueducts(grid, bldg):
    connected = get_connected_aqueducts(grid, bldg)
    assert connected == set()


# ---------------------------------------------------------------------------
# is_aqueduct_connected
# ---------------------------------------------------------------------------


def test_is_connected_true(grid, bldg):
    x, y = _find_water_adjacent_plain(grid)
    grid.place_building("aqueduct", x, y, bldg["aqueduct"])
    assert is_aqueduct_connected(grid, x, y, bldg) is True


def test_is_connected_false(grid, bldg):
    x, y = _find_plain_far_from_water(grid)
    grid.place_building("aqueduct", x, y, bldg["aqueduct"])
    assert is_aqueduct_connected(grid, x, y, bldg) is False


# ---------------------------------------------------------------------------
# is_fountain_functional / get_functional_fountains
# ---------------------------------------------------------------------------


def test_functional_with_chain(grid, bldg):
    ax, ay = _find_water_adjacent_plain(grid)
    grid.place_building("aqueduct", ax, ay, bldg["aqueduct"])

    size = Grid.SIZE
    fx, fy = None, None
    for nx, ny in ((ax - 1, ay), (ax + 1, ay), (ax, ay - 1), (ax, ay + 1)):
        if (
            0 <= nx < size
            and 0 <= ny < size
            and grid.terrain[ny][nx] == TerrainType.PLAIN
            and grid._origin[ny][nx] is None
        ):
            fx, fy = nx, ny
            break
    assert fx is not None, "Pas de voisin PLAIN de l'aqueduc pour la fontaine"

    grid.place_building("fountain", fx, fy, bldg["fountain"])
    assert is_fountain_functional(grid, fx, fy, bldg) is True


def test_not_functional_no_aqueduct(grid, bldg):
    x, y = _find_plain_spot(grid, 1, 1)
    grid.place_building("fountain", x, y, bldg["fountain"])
    assert is_fountain_functional(grid, x, y, bldg) is False


def test_not_functional_disconnected_aqueduct(grid, bldg):
    ax, ay = _find_plain_far_from_water(grid)
    grid.place_building("aqueduct", ax, ay, bldg["aqueduct"])

    size = Grid.SIZE
    fx, fy = None, None
    for nx, ny in ((ax - 1, ay), (ax + 1, ay), (ax, ay - 1), (ax, ay + 1)):
        if (
            0 <= nx < size
            and 0 <= ny < size
            and grid.terrain[ny][nx] == TerrainType.PLAIN
            and grid._origin[ny][nx] is None
        ):
            fx, fy = nx, ny
            break
    assert fx is not None, "Pas de voisin PLAIN de l'aqueduc pour la fontaine"

    grid.place_building("fountain", fx, fy, bldg["fountain"])
    assert is_fountain_functional(grid, fx, fy, bldg) is False


def test_get_functional_mixed(grid, bldg):
    ax, ay = _find_water_adjacent_plain(grid)
    grid.place_building("aqueduct", ax, ay, bldg["aqueduct"])

    size = Grid.SIZE
    f1x, f1y = None, None
    for nx, ny in ((ax - 1, ay), (ax + 1, ay), (ax, ay - 1), (ax, ay + 1)):
        if (
            0 <= nx < size
            and 0 <= ny < size
            and grid.terrain[ny][nx] == TerrainType.PLAIN
            and grid._origin[ny][nx] is None
        ):
            f1x, f1y = nx, ny
            break
    assert f1x is not None
    grid.place_building("fountain", f1x, f1y, bldg["fountain"])

    f2x, f2y = _find_plain_spot(grid, 1, 1)
    grid.place_building("fountain", f2x, f2y, bldg["fountain"])

    functional = get_functional_fountains(grid, bldg)
    assert (f1x, f1y) in functional
    assert (f2x, f2y) not in functional
