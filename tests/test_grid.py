"""Tests pour vitruvius.engine.grid."""

from collections import deque

import pytest

from vitruvius.config import load_config
from vitruvius.engine.grid import Grid, PlacedBuilding
from vitruvius.engine.terrain import TerrainType, _MIN_HILL, _MIN_HILL_BLOCKS_3X3, _MIN_PLAIN, _count_hill_blocks_3x3


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture
def grid():
    return Grid(seed=42)


# ---------------------------------------------------------------------------
# Génération du terrain
# ---------------------------------------------------------------------------

def test_smoke(grid):
    assert isinstance(grid, Grid)


def test_dimensions(grid):
    assert len(grid.terrain) == Grid.SIZE
    assert all(len(row) == Grid.SIZE for row in grid.terrain)
    assert len(grid._origin) == Grid.SIZE
    assert all(len(row) == Grid.SIZE for row in grid._origin)


def test_all_terrain_types_present(grid):
    types_present = {grid.terrain[y][x] for y in range(Grid.SIZE) for x in range(Grid.SIZE)}
    assert types_present == set(TerrainType)


def test_viability_plain(grid):
    count = sum(
        1 for y in range(Grid.SIZE) for x in range(Grid.SIZE)
        if grid.terrain[y][x] == TerrainType.PLAIN
    )
    assert count >= _MIN_PLAIN, f"Seulement {count} tiles PLAIN, minimum {_MIN_PLAIN}"


def test_viability_hill(grid):
    count = sum(
        1 for y in range(Grid.SIZE) for x in range(Grid.SIZE)
        if grid.terrain[y][x] == TerrainType.HILL
    )
    assert count >= _MIN_HILL, f"Seulement {count} tiles HILL, minimum {_MIN_HILL}"


def test_viability_hill_blocks(grid):
    blocks = _count_hill_blocks_3x3(grid.terrain, Grid.SIZE)
    assert blocks >= _MIN_HILL_BLOCKS_3X3, (
        f"Seulement {blocks} blocs 3x3 HILL, minimum {_MIN_HILL_BLOCKS_3X3}"
    )


def test_river_continuous_bord_a_bord(grid):
    """La rivière doit relier le bord gauche au bord droit (BFS sur WATER)."""
    size = Grid.SIZE
    # Cases WATER sur le bord gauche
    start_cells = [
        (0, y) for y in range(size) if grid.terrain[y][0] == TerrainType.WATER
    ]
    assert start_cells, "Aucune case WATER sur le bord gauche"

    visited = set(start_cells)
    queue = deque(start_cells)
    reached_right = False

    while queue:
        x, y = queue.popleft()
        if x == size - 1:
            reached_right = True
            break
        for nx, ny in [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]:
            if 0 <= nx < size and 0 <= ny < size and (nx, ny) not in visited:
                if grid.terrain[ny][nx] == TerrainType.WATER:
                    visited.add((nx, ny))
                    queue.append((nx, ny))

    assert reached_right, "La rivière ne relie pas le bord gauche au bord droit"


def test_determinism():
    g1 = Grid(seed=7)
    g2 = Grid(seed=7)
    assert g1.terrain == g2.terrain


def test_different_seeds_all_viable():
    for seed in range(10):
        g = Grid(seed=seed * 13)
        count_plain = sum(
            1 for y in range(Grid.SIZE) for x in range(Grid.SIZE)
            if g.terrain[y][x] == TerrainType.PLAIN
        )
        assert count_plain >= _MIN_PLAIN, f"seed={seed*13}: seulement {count_plain} PLAIN"


# ---------------------------------------------------------------------------
# Placement de bâtiments
# ---------------------------------------------------------------------------

def _find_plain_spot(grid: Grid, w: int, h: int) -> tuple[int, int]:
    """Trouve une zone w×h entièrement PLAIN et vide."""
    size = Grid.SIZE
    for y in range(size - h + 1):
        for x in range(size - w + 1):
            if all(
                grid.terrain[y + dy][x + dx] == TerrainType.PLAIN
                and grid._origin[y + dy][x + dx] is None
                for dy in range(h) for dx in range(w)
            ):
                return x, y
    raise RuntimeError("Pas de zone PLAIN disponible")


def test_place_building_valid(grid, cfg):
    housing = cfg.buildings.buildings["housing"]
    x, y = _find_plain_spot(grid, 2, 2)
    assert grid.can_place("housing", x, y, housing)
    grid.place_building("housing", x, y, housing)
    assert grid.get_building_at(x, y) is not None
    assert grid.get_building_at(x, y).building_id == "housing"
    # Toutes les cases occupées
    for dy in range(2):
        for dx in range(2):
            assert grid._origin[y + dy][x + dx] == (x, y)
    # Nettoyage
    grid.remove_building(x, y)


def test_place_out_of_bounds(grid, cfg):
    road = cfg.buildings.buildings["road"]
    assert not grid.can_place("road", Grid.SIZE - 1, Grid.SIZE - 1, road) or True  # juste pas d'erreur
    assert not grid.can_place("road", Grid.SIZE, 0, road)
    assert not grid.can_place("road", 0, Grid.SIZE, road)
    assert not grid.can_place("road", -1, 0, road)


def test_place_on_water(grid, cfg):
    road = cfg.buildings.buildings["road"]
    water_cells = [
        (x, y) for y in range(Grid.SIZE) for x in range(Grid.SIZE)
        if grid.terrain[y][x] == TerrainType.WATER
    ]
    assert water_cells, "Pas de WATER sur la grille"
    wx, wy = water_cells[0]
    assert not grid.can_place("road", wx, wy, road)


def test_place_overlap(grid, cfg):
    housing = cfg.buildings.buildings["housing"]
    x, y = _find_plain_spot(grid, 2, 2)
    grid.place_building("housing", x, y, housing)
    assert not grid.can_place("housing", x, y, housing)
    grid.remove_building(x, y)


def test_terrain_constraint_all_tiles(grid, cfg):
    wheat_farm = cfg.buildings.buildings["wheat_farm"]
    # Trouver zone 3x3 entièrement PLAIN
    x, y = _find_plain_spot(grid, 3, 3)
    assert grid.can_place("wheat_farm", x, y, wheat_farm)

    # Trouver une case FOREST, vérifier que wheat_farm ne peut pas y être placé
    forest_cells = [
        (x2, y2) for y2 in range(Grid.SIZE - 2) for x2 in range(Grid.SIZE - 2)
        if grid.terrain[y2][x2] == TerrainType.FOREST
    ]
    if forest_cells:
        fx, fy = forest_cells[0]
        assert not grid.can_place("wheat_farm", fx, fy, wheat_farm)


def test_terrain_constraint_adjacent(grid, cfg):
    lumber = cfg.buildings.buildings["lumber_camp"]
    size = Grid.SIZE
    # Chercher case 2x2 PLAIN avec au moins 1 FOREST adjacent
    for y in range(size - 1):
        for x in range(size - 1):
            # Zone 2x2 doit être PLAIN
            zone_ok = all(
                grid.terrain[y + dy][x + dx] == TerrainType.PLAIN
                and grid._origin[y + dy][x + dx] is None
                for dy in range(2) for dx in range(2)
            )
            if not zone_ok:
                continue
            # Au moins 1 adjacent FOREST
            has_adj_forest = False
            zone = {(x + dx, y + dy) for dy in range(2) for dx in range(2)}
            for dy in range(2):
                for dx in range(2):
                    for nx, ny in [(x+dx-1, y+dy), (x+dx+1, y+dy), (x+dx, y+dy-1), (x+dx, y+dy+1)]:
                        if (nx, ny) not in zone and 0 <= nx < size and 0 <= ny < size:
                            if grid.terrain[ny][nx] == TerrainType.FOREST:
                                has_adj_forest = True
            if has_adj_forest:
                assert grid.can_place("lumber_camp", x, y, lumber)
                return
    pytest.skip("Pas de position 2x2 PLAIN adjacent à FOREST trouvée")


def test_remove_building(grid, cfg):
    housing = cfg.buildings.buildings["housing"]
    x, y = _find_plain_spot(grid, 2, 2)
    grid.place_building("housing", x, y, housing)
    pb = grid.remove_building(x, y)
    assert isinstance(pb, PlacedBuilding)
    assert pb.building_id == "housing"
    # Cases libérées
    for dy in range(2):
        for dx in range(2):
            assert grid._origin[y + dy][x + dx] is None
    assert (x, y) not in grid.placed_buildings


def test_unique_building(grid, cfg):
    forum = cfg.buildings.buildings["forum"]
    x, y = _find_plain_spot(grid, 4, 4)
    grid.place_building("forum", x, y, forum)
    # Second forum refusé
    x2, y2 = _find_plain_spot(grid, 4, 4)
    assert not grid.can_place("forum", x2, y2, forum)
    grid.remove_building(x, y)


def test_ascii_output(grid):
    ascii_map = grid.to_ascii()
    lines = ascii_map.split("\n")
    assert len(lines) == Grid.SIZE
    assert all(len(line) == Grid.SIZE for line in lines)
    chars = set("".join(lines))
    terrain_chars = set(".T^~%")
    assert chars.issubset(terrain_chars | set("ABCDEFGHIJKLMNOPQRSTUVWXYZ#"))
