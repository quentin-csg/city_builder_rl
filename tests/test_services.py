"""Tests pour vitruvius.engine.services."""

import pytest

from vitruvius.config import load_config
from vitruvius.engine.buildings import BuildingConfig
from vitruvius.engine.grid import Grid
from vitruvius.engine.resources import ResourceState
from vitruvius.engine.services import (
    _min_manhattan_distance,
    compute_coverage,
    compute_coverage_grid,
)


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
    return ResourceState(denarii=5000.0, wheat=200, wood=200, marble=500)


@pytest.fixture
def state_no_wheat():
    return ResourceState(denarii=5000.0, wheat=0, wood=200, marble=500)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_plain_spot(grid: Grid, w: int, h: int) -> tuple[int, int]:
    """Trouve une zone w×h entièrement PLAIN et vide."""
    from vitruvius.engine.terrain import TerrainType

    for y in range(grid.SIZE - h + 1):
        for x in range(grid.SIZE - w + 1):
            if all(
                grid.terrain[y + dy][x + dx] == TerrainType.PLAIN
                and grid._origin[y + dy][x + dx] is None
                for dy in range(h)
                for dx in range(w)
            ):
                return x, y
    raise RuntimeError(f"Pas de spot PLAIN {w}×{h} disponible")


def _find_water_adjacent_plain(grid: Grid) -> tuple[int, int]:
    """Trouve une tile PLAIN adjacente (4-connexité) à WATER."""
    from vitruvius.engine.terrain import TerrainType

    for y in range(grid.SIZE):
        for x in range(grid.SIZE):
            if grid.terrain[y][x] != TerrainType.PLAIN:
                continue
            if grid._origin[y][x] is not None:
                continue
            for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                if 0 <= nx < grid.SIZE and 0 <= ny < grid.SIZE:
                    if grid.terrain[ny][nx] == TerrainType.WATER:
                        return x, y
    raise RuntimeError("Pas de tile PLAIN adjacente à WATER")


def _find_plain_spot_near(
    grid: Grid, w: int, h: int, cx: int, cy: int, max_search: int = 20
) -> tuple[int, int]:
    """Trouve la zone w×h PLAIN et vide la plus proche de (cx, cy)."""
    from vitruvius.engine.terrain import TerrainType

    for r in range(max_search):
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if abs(dx) != r and abs(dy) != r:
                    continue
                ox, oy = cx + dx, cy + dy
                if ox < 0 or oy < 0 or ox + w > grid.SIZE or oy + h > grid.SIZE:
                    continue
                if all(
                    grid.terrain[oy + ddy][ox + ddx] == TerrainType.PLAIN
                    and grid._origin[oy + ddy][ox + ddx] is None
                    for ddy in range(h)
                    for ddx in range(w)
                ):
                    return ox, oy
    raise RuntimeError(f"Pas de spot PLAIN {w}×{h} près de ({cx},{cy}) dans {max_search} cases")


# ---------------------------------------------------------------------------
# Tests _min_manhattan_distance
# ---------------------------------------------------------------------------


def test_min_manhattan_same_point():
    assert _min_manhattan_distance(5, 5, 1, 1, 5, 5, 1, 1) == 0


def test_min_manhattan_adjacent():
    # Tiles (5,5) et (6,5) : voisines → distance Manhattan = 1
    assert _min_manhattan_distance(5, 5, 1, 1, 6, 5, 1, 1) == 1


def test_min_manhattan_gap_1():
    # Tiles (5,5) et (7,5) : une tile d'écart → distance Manhattan = 2
    assert _min_manhattan_distance(5, 5, 1, 1, 7, 5, 1, 1) == 2


def test_min_manhattan_diagonal():
    # Tiles (5,5) et (7,7) : dx=2, dy=2 → distance Manhattan = 4
    assert _min_manhattan_distance(5, 5, 1, 1, 7, 7, 1, 1) == 4


def test_min_manhattan_rect_to_rect():
    # Temple 3×3 en (5,5) [tiles x:5-7, y:5-7], housing 2×2 en (10,5) [tiles x:10-11]
    # dx = max(0, 5-(10+2-1), 10-(5+3-1)) = max(0,-6,3) = 3, dy=0
    assert _min_manhattan_distance(5, 5, 3, 3, 10, 5, 2, 2) == 3


def test_min_manhattan_overlapping():
    # Rectangles qui se chevauchent → distance = 0
    assert _min_manhattan_distance(5, 5, 3, 3, 6, 6, 2, 2) == 0


# ---------------------------------------------------------------------------
# Tests basic coverage
# ---------------------------------------------------------------------------


def test_no_service_no_coverage(grid, bldg, state):
    # Seulement housing, aucun bâtiment de service → set vide
    hx, hy = _find_plain_spot(grid, 2, 2)
    grid.place_building("housing", hx, hy, bldg["housing"])

    cov = compute_coverage(grid, bldg, state)
    assert (hx, hy) in cov
    assert cov[(hx, hy)] == set()


def test_no_housing_empty_dict(grid, bldg, state):
    # Seulement service buildings, aucun housing → dict vide
    wx, wy = _find_plain_spot(grid, 1, 1)
    grid.place_building("well", wx, wy, bldg["well"])

    cov = compute_coverage(grid, bldg, state)
    assert cov == {}


def test_coverage_exact_boundary(grid, bldg, state):
    # Well r=3 (1×1 en wx,wy), housing 2×2 en (wx+3,wy)
    # dx = max(0, wx-(wx+3+2-1), (wx+3)-(wx+1-1)) = max(0,-4,3) = 3 ≤ 3 → couvert
    wx, wy = _find_plain_spot(grid, 1, 1)
    grid.place_building("well", wx, wy, bldg["well"])
    hx, hy = wx + 3, wy
    grid.place_building("housing", hx, hy, bldg["housing"])

    cov = compute_coverage(grid, bldg, state)
    assert "water" in cov[(hx, hy)]


def test_coverage_one_beyond_boundary(grid, bldg, state):
    # Well r=3, housing 2×2 en (wx+4,wy) → dx=4 > 3 → pas couvert
    wx, wy = _find_plain_spot(grid, 1, 1)
    grid.place_building("well", wx, wy, bldg["well"])
    hx, hy = wx + 4, wy
    grid.place_building("housing", hx, hy, bldg["housing"])

    cov = compute_coverage(grid, bldg, state)
    assert "water" not in cov[(hx, hy)]


def test_well_does_not_cover_distant_housing(grid, bldg, state):
    # Well r=3, housing à distance 6 → pas couvert
    wx, wy = _find_plain_spot(grid, 1, 1)
    grid.place_building("well", wx, wy, bldg["well"])
    hx, hy = wx + 6, wy
    grid.place_building("housing", hx, hy, bldg["housing"])

    cov = compute_coverage(grid, bldg, state)
    assert "water" not in cov.get((hx, hy), set())


# ---------------------------------------------------------------------------
# Tests multi-tile distance
# ---------------------------------------------------------------------------


def test_temple_3x3_covers_nearby(grid, bldg, state):
    # Temple 3×3 r=8, housing 2×2 à distance 2 → religion couverte
    tx, ty = _find_plain_spot(grid, 3, 3)
    grid.place_building("temple", tx, ty, bldg["temple"])
    # distance = max(0, tx-(hx+1), hx-(tx+2)) avec hx=tx+4 → max(0,-5,2)=2 ≤ 8
    hx, hy = tx + 4, ty
    grid.place_building("housing", hx, hy, bldg["housing"])

    cov = compute_coverage(grid, bldg, state)
    assert "religion" in cov[(hx, hy)]


def test_housing_2x2_nearest_tile(grid, bldg, state):
    # Well r=3, housing 2×2 à distance exacte 3 → couvert
    wx, wy = _find_plain_spot(grid, 1, 1)
    grid.place_building("well", wx, wy, bldg["well"])
    hx, hy = wx + 3, wy
    grid.place_building("housing", hx, hy, bldg["housing"])

    cov = compute_coverage(grid, bldg, state)
    assert "water" in cov[(hx, hy)]


def test_multi_to_multi_boundary(grid, bldg, state):
    # Temple 3×3 r=8, housing 2×2 à distance exacte 8 → couvert
    # Avec temple en (tx,ty) et housing en (tx+10, ty) :
    # dx = max(0, tx-(tx+10+1), (tx+10)-(tx+2)) = max(0,-11,8) = 8 ≤ 8 → couvert
    tx, ty = _find_plain_spot(grid, 3, 3)
    grid.place_building("temple", tx, ty, bldg["temple"])
    hx, hy = tx + 10, ty
    grid.place_building("housing", hx, hy, bldg["housing"])

    cov = compute_coverage(grid, bldg, state)
    assert "religion" in cov[(hx, hy)]


def test_prefecture_large_radius(grid, bldg, state):
    # Préfecture r=10, placer housing quelque part → au moins une housing couverte
    px, py = _find_plain_spot(grid, 2, 2)
    grid.place_building("prefecture", px, py, bldg["prefecture"])
    hx, hy = _find_plain_spot(grid, 2, 2)
    grid.place_building("housing", hx, hy, bldg["housing"])

    cov = compute_coverage(grid, bldg, state)
    assert any("security" in v for v in cov.values())


# ---------------------------------------------------------------------------
# Tests fountain gating
# ---------------------------------------------------------------------------


def test_functional_fountain_covers(grid, bldg, state):
    # Fontaine avec aqueduc connecté → fournit water (r=6) au housing adjacent
    from vitruvius.engine.terrain import TerrainType

    ax, ay = _find_water_adjacent_plain(grid)
    grid.place_building("aqueduct", ax, ay, bldg["aqueduct"])

    # Chercher une tile PLAIN adjacente à l'aqueduc pour la fontaine
    placed_fountain = False
    for nx, ny in ((ax + 1, ay), (ax - 1, ay), (ax, ay + 1), (ax, ay - 1)):
        if 0 <= nx < grid.SIZE and 0 <= ny < grid.SIZE:
            if grid.terrain[ny][nx] == TerrainType.PLAIN and grid._origin[ny][nx] is None:
                grid.place_building("fountain", nx, ny, bldg["fountain"])
                fx, fy = nx, ny
                placed_fountain = True
                break

    if not placed_fountain:
        pytest.skip("Pas de place pour la fontaine adjacente à l'aqueduc")

    # Housing dans le rayon r=6 de la fontaine (chercher près d'elle)
    hx, hy = _find_plain_spot_near(grid, 2, 2, fx, fy)
    grid.place_building("housing", hx, hy, bldg["housing"])

    cov = compute_coverage(grid, bldg, state)
    assert any("water" in v for v in cov.values())


def test_non_functional_fountain_no_coverage(grid, bldg, state):
    # Fontaine SANS aqueduc adjacent → pas de water (sauf si un well est là)
    fx, fy = _find_plain_spot(grid, 1, 1)
    grid.place_building("fountain", fx, fy, bldg["fountain"])
    hx, hy = fx + 1, fy
    grid.place_building("housing", hx, hy, bldg["housing"])

    cov = compute_coverage(grid, bldg, state)
    assert "water" not in cov.get((hx, hy), set())


def test_well_always_functional(grid, bldg, state):
    # Well fonctionne sans aqueduc
    wx, wy = _find_plain_spot(grid, 1, 1)
    grid.place_building("well", wx, wy, bldg["well"])
    hx, hy = wx + 1, wy
    grid.place_building("housing", hx, hy, bldg["housing"])

    cov = compute_coverage(grid, bldg, state)
    assert "water" in cov[(hx, hy)]


def test_fountain_and_well_combined(grid, bldg, state):
    # Well proche couvre housing, fontaine sans aqueduc non-fonctionnelle → water vient du well
    wx, wy = _find_plain_spot(grid, 1, 1)
    grid.place_building("well", wx, wy, bldg["well"])
    fx, fy = _find_plain_spot(grid, 1, 1)
    grid.place_building("fountain", fx, fy, bldg["fountain"])
    hx, hy = wx + 1, wy
    grid.place_building("housing", hx, hy, bldg["housing"])

    cov = compute_coverage(grid, bldg, state)
    assert "water" in cov[(hx, hy)]


# ---------------------------------------------------------------------------
# Tests market gating
# ---------------------------------------------------------------------------


def test_market_with_wheat(grid, bldg, state):
    # wheat=200 > 0 → market fournit food
    mx, my = _find_plain_spot(grid, 2, 2)
    grid.place_building("market", mx, my, bldg["market"])
    hx, hy = mx + 2, my
    grid.place_building("housing", hx, hy, bldg["housing"])

    cov = compute_coverage(grid, bldg, state)
    assert "food" in cov[(hx, hy)]


def test_market_without_wheat(grid, bldg, state_no_wheat):
    # wheat=0 → market inactif
    mx, my = _find_plain_spot(grid, 2, 2)
    grid.place_building("market", mx, my, bldg["market"])
    hx, hy = mx + 2, my
    grid.place_building("housing", hx, hy, bldg["housing"])

    cov = compute_coverage(grid, bldg, state_no_wheat)
    assert "food" not in cov.get((hx, hy), set())


def test_market_wheat_exactly_zero(grid, bldg):
    # wheat=0 strict → pas couvert
    state_zero = ResourceState(denarii=5000.0, wheat=0, wood=0, marble=0)
    mx, my = _find_plain_spot(grid, 2, 2)
    grid.place_building("market", mx, my, bldg["market"])
    hx, hy = mx + 2, my
    grid.place_building("housing", hx, hy, bldg["housing"])

    cov = compute_coverage(grid, bldg, state_zero)
    assert "food" not in cov.get((hx, hy), set())


# ---------------------------------------------------------------------------
# Tests multiple services
# ---------------------------------------------------------------------------


def test_housing_multiple_services(grid, bldg, state):
    # Housing couverte par well + market + small_altar → {water, food, religion}
    # Placer housing d'abord, puis chaque service proche d'elle
    hx, hy = _find_plain_spot(grid, 2, 2)
    grid.place_building("housing", hx, hy, bldg["housing"])
    wx, wy = _find_plain_spot_near(grid, 1, 1, hx, hy)
    grid.place_building("well", wx, wy, bldg["well"])
    mx, my = _find_plain_spot_near(grid, 2, 2, hx, hy)
    grid.place_building("market", mx, my, bldg["market"])
    ax, ay = _find_plain_spot_near(grid, 1, 1, hx, hy)
    grid.place_building("small_altar", ax, ay, bldg["small_altar"])

    cov = compute_coverage(grid, bldg, state)
    covered = cov.get((hx, hy), set())
    assert "water" in covered
    assert "food" in covered
    assert "religion" in covered


def test_housing_missing_one_service(grid, bldg, state):
    # Housing près de well + market, mais pas d'autel → pas de religion
    hx, hy = _find_plain_spot(grid, 2, 2)
    grid.place_building("housing", hx, hy, bldg["housing"])
    wx, wy = _find_plain_spot(grid, 1, 1)
    grid.place_building("well", wx, wy, bldg["well"])
    mx, my = _find_plain_spot(grid, 2, 2)
    grid.place_building("market", mx, my, bldg["market"])

    cov = compute_coverage(grid, bldg, state)
    assert "religion" not in cov.get((hx, hy), set())


def test_all_six_services(grid, bldg, state):
    # Tous les types de service proches d'une housing → 6 besoins couverts
    hx, hy = _find_plain_spot(grid, 2, 2)
    grid.place_building("housing", hx, hy, bldg["housing"])
    # Placer chaque service près du housing (dans son rayon respectif)
    for bid in ["well", "market", "small_altar", "baths", "theater", "prefecture"]:
        w, h = bldg[bid].size
        sx, sy = _find_plain_spot_near(grid, w, h, hx, hy)
        grid.place_building(bid, sx, sy, bldg[bid])

    cov = compute_coverage(grid, bldg, state)
    assert len(cov.get((hx, hy), set())) == 6


# ---------------------------------------------------------------------------
# Tests compute_coverage_grid
# ---------------------------------------------------------------------------


def test_coverage_grid_well_tiles(grid, bldg, state):
    # Well 1×1 r=3 : tiles couvertes respectent la distance Manhattan
    wx, wy = _find_plain_spot(grid, 1, 1)
    grid.place_building("well", wx, wy, bldg["well"])

    cg = compute_coverage_grid(grid, bldg, state)
    water_tiles = cg["water"]

    # Tile du well elle-même couverte (distance 0)
    assert (wx, wy) in water_tiles
    # Tile à distance 3 → couverte
    if wx + 3 < grid.SIZE:
        assert (wx + 3, wy) in water_tiles
    # Tile à distance 4 → pas couverte
    if wx + 4 < grid.SIZE:
        assert (wx + 4, wy) not in water_tiles


def test_coverage_grid_all_six_keys(grid, bldg, state):
    # Les 6 clés doivent toujours être présentes
    cg = compute_coverage_grid(grid, bldg, state)
    assert set(cg.keys()) == {"water", "food", "religion", "hygiene", "entertainment", "security"}


def test_coverage_grid_empty_no_services(grid, bldg, state):
    # Aucun service placé → tous les sets vides
    cg = compute_coverage_grid(grid, bldg, state)
    for svc_type, tiles in cg.items():
        assert tiles == set(), f"Set non-vide pour {svc_type}"


def test_coverage_grid_overlapping(grid, bldg, state):
    # Deux wells → union des tiles couvertes
    wx1, wy1 = _find_plain_spot(grid, 1, 1)
    grid.place_building("well", wx1, wy1, bldg["well"])
    wx2, wy2 = _find_plain_spot(grid, 1, 1)
    grid.place_building("well", wx2, wy2, bldg["well"])

    cg = compute_coverage_grid(grid, bldg, state)
    water_tiles = cg["water"]
    assert (wx1, wy1) in water_tiles
    assert (wx2, wy2) in water_tiles


# ---------------------------------------------------------------------------
# Tests edge cases
# ---------------------------------------------------------------------------


def test_service_at_grid_edge(grid, bldg, state):
    # Bâtiment en bord de grille : les tiles couvertes doivent rester dans les bornes
    from vitruvius.engine.terrain import TerrainType

    placed = False
    for x, y in [(0, 0), (1, 0), (0, 1), (2, 0), (0, 2)]:
        if grid.terrain[y][x] == TerrainType.PLAIN and grid._origin[y][x] is None:
            grid.place_building("well", x, y, bldg["well"])
            placed = True
            break

    if not placed:
        pytest.skip("Pas de tile PLAIN au bord de la grille")

    cg = compute_coverage_grid(grid, bldg, state)
    for tx, ty in cg["water"]:
        assert 0 <= tx < grid.SIZE
        assert 0 <= ty < grid.SIZE


def test_housing_at_grid_edge(grid, bldg, state):
    # Housing au bord inférieur-droit, service proche → apparaît dans cov
    from vitruvius.engine.terrain import TerrainType

    hx, hy = None, None
    for y in range(grid.SIZE - 2, grid.SIZE - 1):
        for x in range(grid.SIZE - 3, grid.SIZE - 1):
            if all(
                grid.terrain[y + dy][x + dx] == TerrainType.PLAIN
                and grid._origin[y + dy][x + dx] is None
                for dy in range(2) for dx in range(2)
            ):
                hx, hy = x, y
                break
        if hx is not None:
            break

    if hx is None:
        pytest.skip("Pas de spot 2×2 PLAIN au bord")

    grid.place_building("housing", hx, hy, bldg["housing"])
    wx, wy = _find_plain_spot(grid, 1, 1)
    grid.place_building("well", wx, wy, bldg["well"])

    cov = compute_coverage(grid, bldg, state)
    assert (hx, hy) in cov


# ---------------------------------------------------------------------------
# Test de cohérence compute_coverage ↔ compute_coverage_grid
# ---------------------------------------------------------------------------


def test_coverage_vs_grid_consistent(grid, bldg, state):
    # Placer well et housing, vérifier la cohérence entre les deux fonctions
    wx, wy = _find_plain_spot(grid, 1, 1)
    grid.place_building("well", wx, wy, bldg["well"])
    hx, hy = _find_plain_spot(grid, 2, 2)
    grid.place_building("housing", hx, hy, bldg["housing"])

    cov = compute_coverage(grid, bldg, state)
    cg = compute_coverage_grid(grid, bldg, state)

    all_needs = {"water", "food", "religion", "hygiene", "entertainment", "security"}
    hw, hh = bldg["housing"].size

    for (ox, oy), covered_needs in cov.items():
        housing_tiles = {(ox + dx, oy + dy) for dy in range(hh) for dx in range(hw)}
        for need in covered_needs:
            assert housing_tiles & cg[need], (
                f"Incohérence : {need} couvert pour ({ox},{oy}) "
                f"mais aucune tile de la housing dans coverage_grid"
            )
        for need in all_needs - covered_needs:
            assert not (housing_tiles & cg[need]), (
                f"Incohérence : {need} pas couvert pour ({ox},{oy}) "
                f"mais des tiles de la housing sont dans coverage_grid"
            )
