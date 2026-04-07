"""Définitions des 20 bâtiments, validation de placement."""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel

from vitruvius.engine.resources import (
    ResourceState,
    can_afford,
    clamp_stocks_to_capacity,
    pay_cost,
    refund_cost,
)
from vitruvius.engine.terrain import TerrainType

if TYPE_CHECKING:
    from vitruvius.engine.grid import Grid, PlacedBuilding


class TerrainConstraint(BaseModel):
    """Contrainte de terrain pour le placement d'un bâtiment."""

    type: Literal["all_tiles", "adjacent"]
    terrain: TerrainType


class ProductionConfig(BaseModel):
    """Production de ressource par tour."""

    resource: str
    amount: int


class StorageConfig(BaseModel):
    """Capacité de stockage d'une ressource."""

    resource: str
    capacity: int


class ServiceConfig(BaseModel):
    """Service fourni par un bâtiment dans son rayon d'influence."""

    type: str
    radius: int


class SpecialEffect(BaseModel):
    """Effets spéciaux d'un bâtiment (champs optionnels, tous à leur valeur neutre par défaut)."""

    requires_aqueduct: bool = False
    is_aqueduct: bool = False
    is_housing: bool = False
    tax_bonus: float = 0.0
    fire_risk_divisor: int = 1


class BuildingConfig(BaseModel):
    """Configuration complète d'un bâtiment."""

    display_name: str
    category: str
    size: tuple[int, int]
    cost: dict[str, int]
    maintenance: int
    unique: bool
    terrain_constraint: TerrainConstraint | None = None
    production: ProductionConfig | None = None
    storage: StorageConfig | None = None
    service: ServiceConfig | None = None
    special_effect: SpecialEffect | None = None


class BuildingsConfig(BaseModel):
    """Configuration complète des bâtiments chargée depuis buildings.yaml."""

    buildings: dict[str, BuildingConfig]


# ---------------------------------------------------------------------------
# Runtime — placement et démolition
# ---------------------------------------------------------------------------


def try_place_building(
    grid: Grid,
    state: ResourceState,
    building_id: str,
    x: int,
    y: int,
    building_configs: dict[str, BuildingConfig],
) -> bool:
    """Tente de placer un bâtiment : vérifie terrain + ressources, puis pose et paie.

    Args:
        grid: Grille de jeu.
        state: État des ressources.
        building_id: Identifiant du bâtiment à placer.
        x: Colonne du coin haut-gauche.
        y: Ligne du coin haut-gauche.
        building_configs: Configs des bâtiments (depuis le YAML).

    Returns:
        True si le placement a réussi, False sinon (terrain invalide ou ressources insuffisantes).
    """
    config = building_configs[building_id]
    if not grid.can_place(building_id, x, y, config):
        return False
    if not can_afford(state, config.cost):
        return False
    pay_cost(state, config.cost)
    grid.place_building(building_id, x, y, config)
    return True


def try_demolish(
    grid: Grid,
    state: ResourceState,
    x: int,
    y: int,
    building_configs: dict[str, BuildingConfig],
) -> PlacedBuilding | None:
    """Démolis le bâtiment occupant la case (x, y), rembourse 50% du coût (floor).

    Si la démolition réduit la capacité de stockage sous le stock actuel,
    l'excédent est perdu (granary/warehouse démolit).

    Args:
        grid: Grille de jeu.
        state: État des ressources.
        x: Colonne (n'importe quelle case du bâtiment).
        y: Ligne.
        building_configs: Configs des bâtiments.

    Returns:
        Le PlacedBuilding supprimé, ou None si la case était vide.
    """
    pb = grid.remove_building(x, y)
    if pb is None:
        return None
    # Clamp uniquement si le bâtiment démoli avait du stockage — sinon cap==0
    # zérerait des ressources indépendantes de ce bâtiment.
    demolished_cfg = building_configs[pb.building_id]
    if demolished_cfg.storage is not None:
        clamp_stocks_to_capacity(state, grid.placed_buildings, building_configs)
    # Remboursement ensuite : toujours reçu, indépendamment de la capacité
    refund_cost(state, building_configs[pb.building_id].cost)
    return pb


# ---------------------------------------------------------------------------
# Runtime — connectivité des aqueducs
# ---------------------------------------------------------------------------


def get_connected_aqueducts(
    grid: Grid,
    building_configs: dict[str, BuildingConfig],
) -> set[tuple[int, int]]:
    """Retourne les origins de tous les aqueducs connectés à une tile WATER.

    Un aqueduc est connecté s'il est adjacent (4-connexité) à une tile WATER
    ou à un autre aqueduc lui-même connecté. BFS depuis toutes les tiles WATER.

    Args:
        grid: Grille de jeu.
        building_configs: Configs des bâtiments.

    Returns:
        Set de (ox, oy) — coins haut-gauche des aqueducs connectés à WATER.
    """
    # Construire reverse map : tile (x, y) → origin (ox, oy) pour les aqueducs
    tile_to_aqueduct_origin: dict[tuple[int, int], tuple[int, int]] = {}
    for (ox, oy), pb in grid.placed_buildings.items():
        cfg = building_configs[pb.building_id]
        if cfg.special_effect is not None and cfg.special_effect.is_aqueduct:
            w, h = pb.size
            for dy in range(h):
                for dx in range(w):
                    tile_to_aqueduct_origin[(ox + dx, oy + dy)] = (ox, oy)

    connected_origins: set[tuple[int, int]] = set()
    # Tiles visitées dans le BFS (inclut les tiles water ET les tiles d'aqueducs visités)
    visited: set[tuple[int, int]] = set()

    # Amorcer le BFS depuis toutes les tiles WATER (cache précalculé, pas de scan O(SIZE²))
    frontier: deque[tuple[int, int]] = deque(grid.water_tiles)
    visited.update(grid.water_tiles)
    size = grid.SIZE

    while frontier:
        cx, cy = frontier.popleft()
        for nx, ny in ((cx - 1, cy), (cx + 1, cy), (cx, cy - 1), (cx, cy + 1)):
            if (nx, ny) in visited:
                continue
            if not (0 <= nx < size and 0 <= ny < size):
                continue
            if (nx, ny) not in tile_to_aqueduct_origin:
                continue
            # Voisin est une tile d'aqueduc non encore visitée
            origin = tile_to_aqueduct_origin[(nx, ny)]
            if origin in connected_origins:
                continue
            connected_origins.add(origin)
            # Ajouter toutes les tiles de cet aqueduc au frontier
            pb = grid.placed_buildings[origin]
            ox, oy = origin
            w, h = pb.size
            for dy in range(h):
                for dx in range(w):
                    tile = (ox + dx, oy + dy)
                    if tile not in visited:
                        visited.add(tile)
                        frontier.append(tile)

    return connected_origins


def is_aqueduct_connected(
    grid: Grid,
    x: int,
    y: int,
    building_configs: dict[str, BuildingConfig],
) -> bool:
    """Vérifie si l'aqueduc dont la case (x, y) fait partie est connecté à WATER.

    Args:
        grid: Grille de jeu.
        x: Colonne d'une tile de l'aqueduc.
        y: Ligne.
        building_configs: Configs des bâtiments.

    Returns:
        True si l'aqueduc est dans la chaîne connectée à WATER.
    """
    origin = grid._origin[y][x]
    if origin is None:
        return False
    connected = get_connected_aqueducts(grid, building_configs)
    return origin in connected


def _has_connected_aqueduct_neighbor(
    grid: Grid,
    ox: int,
    oy: int,
    w: int,
    h: int,
    connected: set[tuple[int, int]],
) -> bool:
    """Vérifie si un bâtiment (ox, oy, w×h) a un voisin aqueduc connecté à WATER."""
    size = grid.SIZE
    for dy in range(h):
        for dx in range(w):
            tx, ty = ox + dx, oy + dy
            for nx, ny in ((tx - 1, ty), (tx + 1, ty), (tx, ty - 1), (tx, ty + 1)):
                if not (0 <= nx < size and 0 <= ny < size):
                    continue
                origin = grid._origin[ny][nx]
                if origin is not None and origin in connected:
                    return True
    return False


def is_fountain_functional(
    grid: Grid,
    x: int,
    y: int,
    building_configs: dict[str, BuildingConfig],
) -> bool:
    """Vérifie si la fontaine en (x, y) est fonctionnelle (aqueduc connecté adjacent).

    Une fontaine est fonctionnelle si au moins un de ses voisins 4-connexes
    est un aqueduc lui-même connecté à WATER.

    Args:
        grid: Grille de jeu.
        x: Colonne du coin haut-gauche de la fontaine.
        y: Ligne.
        building_configs: Configs des bâtiments.

    Returns:
        True si la fontaine peut fournir son service eau.
    """
    pb = grid.placed_buildings.get((x, y))
    if pb is None:
        return False
    cfg = building_configs[pb.building_id]
    if cfg.special_effect is None or not cfg.special_effect.requires_aqueduct:
        return False
    connected = get_connected_aqueducts(grid, building_configs)
    w, h = pb.size
    return _has_connected_aqueduct_neighbor(grid, x, y, w, h, connected)


def get_functional_fountains(
    grid: Grid,
    building_configs: dict[str, BuildingConfig],
) -> set[tuple[int, int]]:
    """Retourne les origins de toutes les fontaines fonctionnelles.

    Calcule la connectivité des aqueducs une seule fois, puis vérifie chaque fontaine.
    Utilisé par services.py pour déterminer la couverture eau effective.

    Args:
        grid: Grille de jeu.
        building_configs: Configs des bâtiments.

    Returns:
        Set de (ox, oy) — origins des fontaines fonctionnelles.
    """
    connected = get_connected_aqueducts(grid, building_configs)
    functional: set[tuple[int, int]] = set()

    for (ox, oy), pb in grid.placed_buildings.items():
        cfg = building_configs[pb.building_id]
        if cfg.special_effect is None or not cfg.special_effect.requires_aqueduct:
            continue
        w, h = pb.size
        if _has_connected_aqueduct_neighbor(grid, ox, oy, w, h, connected):
            functional.add((ox, oy))


    return functional
