"""Système de rayon d'influence des bâtiments de service (pas de walkers)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from vitruvius.engine.buildings import BuildingConfig, get_functional_fountains
from vitruvius.engine.resources import ResourceState, get_stock

if TYPE_CHECKING:
    from vitruvius.engine.grid import Grid

_SERVICE_TYPES = frozenset(["water", "food", "religion", "hygiene", "entertainment", "security"])


def _min_manhattan_distance(
    ox1: int, oy1: int, w1: int, h1: int,
    ox2: int, oy2: int, w2: int, h2: int,
) -> int:
    """Distance de Manhattan minimale entre deux rectangles axis-aligned.

    Args:
        ox1, oy1: Coin haut-gauche du rectangle 1.
        w1, h1: Taille du rectangle 1.
        ox2, oy2: Coin haut-gauche du rectangle 2.
        w2, h2: Taille du rectangle 2.

    Returns:
        Distance de Manhattan minimale entre les deux rectangles (0 si chevauchement ou coïncidence).
    """
    dx = max(0, ox1 - (ox2 + w2 - 1), ox2 - (ox1 + w1 - 1))
    dy = max(0, oy1 - (oy2 + h2 - 1), oy2 - (oy1 + h1 - 1))
    return dx + dy


def _get_functional_service_buildings(
    grid: Grid,
    building_configs: dict[str, BuildingConfig],
    resource_state: ResourceState,
) -> list[tuple[tuple[int, int], str, int, tuple[int, int]]]:
    """Collecte tous les bâtiments de service fonctionnels avec leurs paramètres.

    Gère les règles de fonctionnement spéciales :
    - Fontaine : fonctionnelle seulement si adjacente à un aqueduc connecté à WATER.
    - Service food : fonctionnel seulement si blé > 0 en stock.

    Args:
        grid: Grille de jeu.
        building_configs: Configs des bâtiments.
        resource_state: État des ressources (pour vérifier le blé).

    Returns:
        Liste de (origin, service_type, radius, size) pour chaque bâtiment fonctionnel.
    """
    result: list[tuple[tuple[int, int], str, int, tuple[int, int]]] = []

    # Calcul unique de la connectivité des fontaines (BFS interne à get_functional_fountains)
    functional_fountains: set[tuple[int, int]] | None = None
    has_wheat = get_stock(resource_state, "wheat") > 0

    for (ox, oy), pb in grid.placed_buildings.items():
        cfg = building_configs[pb.building_id]
        if cfg.service is None:
            continue

        service_type = cfg.service.type
        radius = cfg.service.radius

        # Gating fontaine : fonctionnelle seulement si aqueduc connecté adjacent
        if cfg.special_effect is not None and cfg.special_effect.requires_aqueduct:
            if functional_fountains is None:
                functional_fountains = get_functional_fountains(grid, building_configs)
            if (ox, oy) not in functional_fountains:
                continue

        # Gating food : marché inactif sans blé
        if service_type == "food" and not has_wheat:
            continue

        result.append(((ox, oy), service_type, radius, pb.size))

    return result


def compute_coverage(
    grid: Grid,
    building_configs: dict[str, BuildingConfig],
    resource_state: ResourceState,
) -> dict[tuple[int, int], set[str]]:
    """Calcule la couverture de service pour chaque bâtiment de type housing.

    Pour chaque housing, retourne l'ensemble des types de service qui le couvrent.
    La couverture est déterminée par la distance de Manhattan entre les rectangles.
    Un housing est couvert si la distance minimale entre ses tiles et celles du
    bâtiment de service est inférieure ou égale au rayon du service.

    Toutes les housing apparaissent dans le résultat, même avec un set vide.

    Args:
        grid: Grille de jeu.
        building_configs: Configs des bâtiments.
        resource_state: État des ressources (pour gating marché).

    Returns:
        Dict mapping (ox, oy) de chaque housing → set des besoins couverts.
    """
    service_buildings = _get_functional_service_buildings(grid, building_configs, resource_state)

    # Collecter toutes les housing
    coverage: dict[tuple[int, int], set[str]] = {}
    housing_list: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for (ox, oy), pb in grid.placed_buildings.items():
        cfg = building_configs[pb.building_id]
        if cfg.special_effect is not None and cfg.special_effect.is_housing:
            coverage[(ox, oy)] = set()
            housing_list.append(((ox, oy), pb.size))

    # Pour chaque paire (service, housing), calculer la distance O(1)
    for (sox, soy), svc_type, radius, (sw, sh) in service_buildings:
        for (hox, hoy), (hw, hh) in housing_list:
            dist = _min_manhattan_distance(sox, soy, sw, sh, hox, hoy, hw, hh)
            if dist <= radius:
                coverage[(hox, hoy)].add(svc_type)

    return coverage


def compute_coverage_grid(
    grid: Grid,
    building_configs: dict[str, BuildingConfig],
    resource_state: ResourceState,
) -> dict[str, set[tuple[int, int]]]:
    """Calcule la couverture par tile pour les 6 types de service.

    Pour chaque type de service, retourne l'ensemble de toutes les tiles (x, y)
    couvertes par au moins un bâtiment fonctionnel de ce type. Utilisé par
    observation.py pour remplir les canaux 3–8 de l'observation RL.

    Les 6 clés sont toujours présentes même si leur set est vide.

    Args:
        grid: Grille de jeu.
        building_configs: Configs des bâtiments.
        resource_state: État des ressources (pour gating marché).

    Returns:
        Dict mapping service_type → set de tiles (x, y) couvertes.
    """
    result: dict[str, set[tuple[int, int]]] = {svc: set() for svc in _SERVICE_TYPES}
    size = grid.SIZE

    for (ox, oy), svc_type, radius, (w, h) in _get_functional_service_buildings(
        grid, building_configs, resource_state
    ):
        # Itérer sur le diamant Manhattan autour du rectangle, clippé à [0, size-1]
        # Pour un rectangle w×h, la zone couverte en y va de oy-r à oy+h-1+r
        y_min = max(0, oy - radius)
        y_max = min(size - 1, oy + h - 1 + radius)
        covered = result[svc_type]

        for ty in range(y_min, y_max + 1):
            # Distance verticale au rectangle [oy, oy+h-1]
            dy = max(0, oy - ty, ty - (oy + h - 1))
            remaining = radius - dy
            # Distance horizontale restante disponible
            x_min = max(0, ox - remaining)
            x_max = min(size - 1, ox + w - 1 + remaining)
            for tx in range(x_min, x_max + 1):
                covered.add((tx, ty))

    return result
