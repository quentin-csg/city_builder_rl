"""Croissance, besoins, satisfaction, niveaux de maison."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pydantic import BaseModel, model_validator

if TYPE_CHECKING:
    from vitruvius.engine.buildings import BuildingConfig
    from vitruvius.engine.grid import Grid

VALID_NEEDS = frozenset(["water", "food", "religion", "hygiene", "entertainment", "security"])


class HouseLevelConfig(BaseModel):
    """Configuration d'un niveau de maison."""

    level: int
    id: str
    display_name: str
    max_population: int
    required_needs: list[str]
    tax_per_inhabitant: float

    @model_validator(mode="after")
    def validate_needs(self) -> "HouseLevelConfig":
        invalid = set(self.required_needs) - VALID_NEEDS
        if invalid:
            raise ValueError(f"Besoins inconnus pour le niveau {self.level} : {invalid}")
        return self


class NeedsConfig(BaseModel):
    """Configuration des niveaux de maison chargée depuis needs.yaml."""

    house_levels: list[HouseLevelConfig]

    @model_validator(mode="after")
    def validate_levels(self) -> "NeedsConfig":
        levels = [hl.level for hl in self.house_levels]
        if sorted(levels) != list(range(1, len(levels) + 1)):
            raise ValueError(f"Les niveaux de maison doivent être 1..{len(levels)} sans trous.")
        return self


# ---------------------------------------------------------------------------
# Runtime state
# ---------------------------------------------------------------------------


@dataclass
class HouseState:
    """État mutable d'un plot housing pendant la partie."""

    origin: tuple[int, int]   # (ox, oy) coin haut-gauche du bâtiment housing
    level: int                # 0 = plot vide, 1–6 = niveau de maison habité
    population: int           # habitants actuels
    famine: bool = field(default=False)  # positionné par turn.py, consommé par apply_famine_loss


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


def init_houses(
    grid: Grid,
    building_configs: dict[str, BuildingConfig],
) -> dict[tuple[int, int], HouseState]:
    """Crée un HouseState vide pour chaque plot housing présent dans la grille.

    Args:
        grid: Grille de jeu.
        building_configs: Configs des bâtiments.

    Returns:
        Dict mapping (ox, oy) → HouseState(level=0, population=0).
    """
    houses: dict[tuple[int, int], HouseState] = {}
    for (ox, oy), pb in grid.placed_buildings.items():
        cfg = building_configs[pb.building_id]
        if cfg.special_effect is not None and cfg.special_effect.is_housing:
            houses[(ox, oy)] = HouseState(origin=(ox, oy), level=0, population=0)
    return houses


# ---------------------------------------------------------------------------
# Satisfaction
# ---------------------------------------------------------------------------


def _count_adjacent_roads(origin: tuple[int, int], grid: Grid) -> int:
    """Compte les tiles de route adjacentes (4-connexité) au footprint du housing.

    Args:
        origin: (ox, oy) coin haut-gauche du housing.
        grid: Grille de jeu.

    Returns:
        Nombre de tiles uniques adjacentes au footprint occupées par une route.
    """
    ox, oy = origin
    pb = grid.placed_buildings.get(origin)
    if pb is None:
        return 0
    w, h = pb.size
    size = grid.SIZE

    footprint = {(ox + dx, oy + dy) for dy in range(h) for dx in range(w)}
    neighbors: set[tuple[int, int]] = set()
    for tx, ty in footprint:
        for nx, ny in ((tx - 1, ty), (tx + 1, ty), (tx, ty - 1), (tx, ty + 1)):
            if (nx, ny) not in footprint and 0 <= nx < size and 0 <= ny < size:
                neighbors.add((nx, ny))

    road_count = 0
    for nx, ny in neighbors:
        nb_origin = grid._origin[ny][nx]
        if nb_origin is not None and grid.placed_buildings[nb_origin].building_id == "road":
            road_count += 1
    return road_count


def compute_house_satisfaction(
    house: HouseState,
    covered_needs: set[str],
    house_levels: list[HouseLevelConfig],
    grid: Grid,
) -> float:
    """Calcule la satisfaction d'une maison.

    satisfaction = besoins_satisfaits / besoins_requis + 0.03 × routes_adjacentes, capé à 1.0.
    Level 0 (plot vide) → 0.0.

    Args:
        house: État de la maison.
        covered_needs: Ensemble des besoins couverts pour cette maison (depuis compute_coverage).
        house_levels: Liste des configs de niveaux (issues de NeedsConfig.house_levels).
        grid: Grille de jeu (pour le comptage des routes adjacentes).

    Returns:
        Satisfaction dans [0.0, 1.0].
    """
    if house.level == 0:
        return 0.0
    required = house_levels[house.level - 1].required_needs
    base = sum(1 for n in required if n in covered_needs) / len(required) if required else 1.0
    road_bonus = 0.03 * _count_adjacent_roads(house.origin, grid)
    return min(1.0, base + road_bonus)


def compute_global_satisfaction(
    houses: dict[tuple[int, int], HouseState],
    coverage: dict[tuple[int, int], set[str]],
    house_levels: list[HouseLevelConfig],
    grid: Grid,
) -> float:
    """Calcule la satisfaction globale (moyenne pondérée par population).

    Args:
        houses: États de toutes les maisons.
        coverage: Dict (ox, oy) → set de besoins couverts (depuis compute_coverage).
        house_levels: Configs des niveaux de maison.
        grid: Grille de jeu.

    Returns:
        Satisfaction globale dans [0.0, 1.0]. Retourne 0.5 si population totale = 0.
    """
    total_pop = 0
    weighted_sum = 0.0
    for house in houses.values():
        if house.population <= 0 or house.level == 0:
            continue
        covered = coverage.get(house.origin, set())
        sat = compute_house_satisfaction(house, covered, house_levels, grid)
        weighted_sum += sat * house.population
        total_pop += house.population
    if total_pop == 0:
        return 0.5
    return weighted_sum / total_pop


# ---------------------------------------------------------------------------
# Taxes
# ---------------------------------------------------------------------------


def compute_house_taxes(
    houses: dict[tuple[int, int], HouseState],
    house_levels: list[HouseLevelConfig],
) -> list[float]:
    """Calcule les taxes par maison (pour apply_taxes() dans resources.py).

    Args:
        houses: États des maisons.
        house_levels: Configs des niveaux de maison.

    Returns:
        Liste de floor(pop × tax_per_inhabitant) par maison. Level 0 → 0.0.
    """
    result: list[float] = []
    for house in houses.values():
        if house.level == 0 or house.population <= 0:
            result.append(0.0)
        else:
            tax_rate = house_levels[house.level - 1].tax_per_inhabitant
            result.append(math.floor(house.population * tax_rate))
    return result


# ---------------------------------------------------------------------------
# Famine
# ---------------------------------------------------------------------------


def apply_famine_loss(
    houses: dict[tuple[int, int], HouseState],
) -> int:
    """Applique la perte de population aux maisons marquées famine.

    Chaque maison avec famine=True perd ceil(10% de sa population).
    Remet famine=False après traitement.

    Args:
        houses: États des maisons (modifiés en place).

    Returns:
        Population totale perdue.
    """
    total_lost = 0
    for house in houses.values():
        if house.famine:
            loss = math.ceil(house.population * 0.10)
            house.population = max(0, house.population - loss)
            total_lost += loss
            house.famine = False
    return total_lost


# ---------------------------------------------------------------------------
# Évolution / Régression
# ---------------------------------------------------------------------------


def evolve_houses(
    houses: dict[tuple[int, int], HouseState],
    coverage: dict[tuple[int, int], set[str]],
    house_levels: list[HouseLevelConfig],
) -> tuple[int, int]:
    """Fait évoluer ou régresser les maisons selon la couverture de services.

    Règles :
    - Level 0 : ignoré (la montée 0→1 se fait via apply_immigration).
    - Régression : si un besoin du niveau actuel n'est plus satisfait.
      - Level 1 (tente) sans eau → level=0, population=0 (tous perdus).
      - Level 2+ → level-=1, excédent de pop perdu immédiatement.
    - Évolution : si tous les besoins du niveau supérieur sont satisfaits.
      Un seul mouvement par tour (évolution OU régression, pas les deux).

    Args:
        houses: États des maisons (modifiés en place).
        coverage: Dict (ox, oy) → set de besoins couverts.
        house_levels: Configs des niveaux.

    Returns:
        (nb_évolutions, nb_régressions) pour le tour.
    """
    evolved = 0
    regressed = 0
    for house in houses.values():
        if house.level == 0:
            # Level 0 peut évoluer vers level 1 si les besoins de level 1 sont satisfaits
            covered = coverage.get(house.origin, set())
            next_needs = house_levels[0].required_needs  # level 1 = ["water"]
            if all(n in covered for n in next_needs):
                house.level = 1
                evolved += 1
            continue
        covered = coverage.get(house.origin, set())
        current_needs = house_levels[house.level - 1].required_needs

        if not all(n in covered for n in current_needs):
            # Régression
            if house.level == 1:
                house.level = 0
                house.population = 0
            else:
                house.level -= 1
                cap = house_levels[house.level - 1].max_population
                house.population = min(house.population, cap)
            regressed += 1
        elif house.level < len(house_levels):
            # Tentative d'évolution
            next_needs = house_levels[house.level].required_needs
            if all(n in covered for n in next_needs):
                house.level += 1
                evolved += 1

    return evolved, regressed


# ---------------------------------------------------------------------------
# Croissance / Exode
# ---------------------------------------------------------------------------


def apply_growth(
    houses: dict[tuple[int, int], HouseState],
    global_satisfaction: float,
    house_levels: list[HouseLevelConfig],
) -> int:
    """Ajoute des habitants dans les maisons qui ont de la place (satisfaction ≥ 0.5).

    Taux de croissance = 0.05 × satisfaction_globale.
    Chaque maison reçoit min(place_libre, ceil(pop_max × taux)).

    Args:
        houses: États des maisons (modifiés en place).
        global_satisfaction: Satisfaction globale du tour.
        house_levels: Configs des niveaux.

    Returns:
        Nombre total d'habitants ajoutés.
    """
    if global_satisfaction < 0.5:
        return 0
    rate = 0.05 * global_satisfaction
    total_added = 0
    for house in houses.values():
        if house.level == 0:
            continue
        cap = house_levels[house.level - 1].max_population
        space = cap - house.population
        if space <= 0:
            continue
        add = min(space, math.ceil(cap * rate))
        house.population += add
        total_added += add
    return total_added


def apply_exodus(
    houses: dict[tuple[int, int], HouseState],
    global_satisfaction: float,
) -> int:
    """Applique l'exode si la satisfaction globale est inférieure à 30%.

    5% de la population totale part, répartis proportionnellement entre les maisons.

    Args:
        houses: États des maisons (modifiés en place).
        global_satisfaction: Satisfaction globale du tour.

    Returns:
        Population totale perdue par exode. 0 si pas d'exode.
    """
    if global_satisfaction >= 0.3:
        return 0
    total_pop = sum(h.population for h in houses.values())
    if total_pop == 0:
        return 0
    to_lose = math.ceil(total_pop * 0.05)

    # Répartition proportionnelle : floor par maison, excédent absorbé en ordre décroissant
    losses: dict[tuple[int, int], int] = {}
    assigned = 0
    for house in houses.values():
        if house.population <= 0:
            losses[house.origin] = 0
            continue
        loss = math.floor(house.population / total_pop * to_lose)
        losses[house.origin] = loss
        assigned += loss

    # Distribuer l'excédent restant (to_lose - assigned) aux maisons les plus peuplées
    remainder = to_lose - assigned
    if remainder > 0:
        sorted_houses = sorted(
            [h for h in houses.values() if h.population > 0],
            key=lambda h: h.population,
            reverse=True,
        )
        for house in sorted_houses:
            if remainder <= 0:
                break
            extra = min(1, house.population - losses[house.origin])
            if extra > 0:
                losses[house.origin] += extra
                remainder -= extra

    total_lost = 0
    for house in houses.values():
        loss = losses.get(house.origin, 0)
        house.population = max(0, house.population - loss)
        total_lost += loss

    return total_lost


# ---------------------------------------------------------------------------
# Immigration
# ---------------------------------------------------------------------------


def apply_immigration(
    houses: dict[tuple[int, int], HouseState],
    amount: int,
    house_levels: list[HouseLevelConfig],
) -> int:
    """Distribue des immigrants dans les maisons ayant de la place disponible.

    Un plot vide (level=0) est automatiquement promu au level 1 (tente) à l'arrivée
    des premiers immigrants. Les immigrants non placés (toutes maisons pleines) sont perdus.

    Args:
        houses: États des maisons (modifiés en place).
        amount: Nombre d'immigrants à distribuer.
        house_levels: Configs des niveaux.

    Returns:
        Nombre d'immigrants réellement installés.
    """
    remaining = amount
    for house in houses.values():
        if remaining <= 0:
            break
        if house.level == 0:
            house.level = 1
        cap = house_levels[house.level - 1].max_population
        space = cap - house.population
        if space <= 0:
            continue
        add = min(space, remaining)
        house.population += add
        remaining -= add
    return amount - remaining
