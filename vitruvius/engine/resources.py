"""Gestion des 4 ressources : Denarii, Blé, Bois, Marbre."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import BaseModel, field_validator

if TYPE_CHECKING:
    from vitruvius.engine.buildings import BuildingConfig
    from vitruvius.engine.grid import PlacedBuilding


class ResourceConfig(BaseModel):
    """Configuration d'une ressource."""

    display_name: str
    starting_amount: int
    storage_building: str | None
    max_storage: int | None


class PassiveIncomeConfig(BaseModel):
    """Revenu passif par tour, indépendant des bâtiments."""

    denarii: int

    @field_validator("denarii")
    @classmethod
    def must_be_positive(cls, v: int) -> int:
        if v < 0:
            raise ValueError("Le revenu passif en denarii ne peut pas être négatif.")
        return v


class ResourcesConfig(BaseModel):
    """Configuration complète des ressources chargée depuis resources.yaml."""

    resources: dict[str, ResourceConfig]
    passive_income: PassiveIncomeConfig


# ---------------------------------------------------------------------------
# Runtime state
# ---------------------------------------------------------------------------

_RESOURCE_KEYS = frozenset({"denarii", "wheat", "wood", "marble"})


@dataclass
class ResourceState:
    """État mutable des 4 ressources pendant la partie."""

    denarii: float
    wheat: int
    wood: int
    marble: int


def init_resources(config: ResourcesConfig) -> ResourceState:
    """Crée l'état initial depuis les montants de départ de la config.

    Les stocks de départ existent sans bâtiment de stockage.
    """
    r = config.resources
    return ResourceState(
        denarii=float(r["denarii"].starting_amount),
        wheat=int(r["wheat"].starting_amount),
        wood=int(r["wood"].starting_amount),
        marble=int(r["marble"].starting_amount),
    )


def get_stock(state: ResourceState, key: str) -> float | int:
    """Retourne le stock actuel d'une ressource par clé string.

    Raises:
        ValueError: Si la clé est inconnue.
    """
    if key not in _RESOURCE_KEYS:
        raise ValueError(f"Ressource inconnue : {key!r}")
    return getattr(state, key)


def set_stock(state: ResourceState, key: str, value: float | int) -> None:
    """Définit le stock d'une ressource par clé string.

    Raises:
        ValueError: Si la clé est inconnue.
    """
    if key not in _RESOURCE_KEYS:
        raise ValueError(f"Ressource inconnue : {key!r}")
    setattr(state, key, value)


def compute_storage_cap(
    resource_key: str,
    placed_buildings: dict[tuple[int, int], PlacedBuilding],
    building_configs: dict[str, BuildingConfig],
) -> int | None:
    """Somme la capacité de stockage pour une ressource depuis les bâtiments posés.

    Args:
        resource_key: Clé de la ressource ("wheat", "wood", "marble", "denarii").
        placed_buildings: Bâtiments actuellement posés sur la grille.
        building_configs: Configs des bâtiments (depuis le YAML).

    Returns:
        None si la ressource est illimitée (denarii).
        Somme des capacités de tous les bâtiments de stockage correspondants (0 si aucun).
    """
    if resource_key == "denarii":
        return None
    total = 0
    for pb in placed_buildings.values():
        cfg = building_configs[pb.building_id]
        if cfg.storage is not None and cfg.storage.resource == resource_key:
            total += cfg.storage.capacity
    return total


def apply_production(
    state: ResourceState,
    placed_buildings: dict[tuple[int, int], PlacedBuilding],
    building_configs: dict[str, BuildingConfig],
    resources_config: ResourcesConfig,
) -> dict[str, int]:
    """Applique la production de tous les bâtiments producteurs.

    Pour les ressources non-denarii : vérifie qu'au moins un bâtiment de
    stockage du bon type est posé. Sans stockage → production perdue (0).
    La production est capée par la capacité totale de stockage disponible.

    Args:
        state: État des ressources à modifier.
        placed_buildings: Bâtiments posés sur la grille.
        building_configs: Configs des bâtiments.
        resources_config: Config des ressources (pour connaître le storage_building requis).

    Returns:
        Dict {resource_key: montant_effectivement_ajouté}.
    """
    produced: dict[str, int] = {}

    for pb in placed_buildings.values():
        cfg = building_configs[pb.building_id]
        if cfg.production is None:
            continue

        resource = cfg.production.resource
        amount = cfg.production.amount

        # Vérification : stockage requis pour les ressources non-denarii
        storage_building = resources_config.resources[resource].storage_building
        if storage_building is not None:
            has_storage = any(
                building_configs[pb2.building_id].storage is not None
                and building_configs[pb2.building_id].storage.resource == resource
                for pb2 in placed_buildings.values()
            )
            if not has_storage:
                produced.setdefault(resource, 0)
                continue

        cap = compute_storage_cap(resource, placed_buildings, building_configs)
        current = get_stock(state, resource)

        if cap is None:
            added = amount
        else:
            added = max(0, min(amount, cap - int(current)))

        set_stock(state, resource, current + added)
        produced[resource] = produced.get(resource, 0) + added

    return produced


def apply_passive_income(
    state: ResourceState,
    passive_income_config: PassiveIncomeConfig,
) -> float:
    """Ajoute le revenu passif en denarii.

    Returns:
        Montant ajouté.
    """
    amount = float(passive_income_config.denarii)
    state.denarii += amount
    return amount


def apply_maintenance(
    state: ResourceState,
    placed_buildings: dict[tuple[int, int], PlacedBuilding],
    building_configs: dict[str, BuildingConfig],
) -> float:
    """Soustrait les coûts de maintenance (denarii) de tous les bâtiments posés.

    Denarii peut devenir négatif (faillite traquée par consecutive_bankrupt_turns).

    Returns:
        Coût total de maintenance soustrait.
    """
    total = sum(
        building_configs[pb.building_id].maintenance
        for pb in placed_buildings.values()
    )
    state.denarii -= total
    return float(total)


def apply_taxes(
    state: ResourceState,
    houses_tax_data: list[float],
) -> float:
    """Ajoute les taxes pré-calculées aux denarii.

    Args:
        houses_tax_data: Liste de floor(pop × taxe_par_habitant) par maison,
            pré-calculée par le module population.

    Returns:
        Total des taxes collectées.
    """
    total = sum(houses_tax_data)
    state.denarii += total
    return total


def apply_wheat_consumption(
    state: ResourceState,
    houses_pop: list[int],
) -> list[bool]:
    """Consomme le blé pour chaque maison selon l'ordre FIFO (tout-ou-rien).

    Chaque maison consomme ceil(pop / 10) blé/tour. Si le blé est insuffisant
    pour une maison, elle est marquée famine mais son blé N'est PAS déduit :
    le blé restant est disponible pour les maisons suivantes.

    Args:
        state: État des ressources à modifier.
        houses_pop: Population actuelle par maison, dans l'ordre de placement (FIFO).

    Returns:
        Liste de bool de même longueur que houses_pop. True = famine.
    """
    famine_flags: list[bool] = []
    for pop in houses_pop:
        need = math.ceil(pop / 10) if pop > 0 else 0
        if need == 0:
            famine_flags.append(False)
        elif state.wheat >= need:
            state.wheat -= need
            famine_flags.append(False)
        else:
            famine_flags.append(True)
    return famine_flags


def can_afford(state: ResourceState, cost: dict[str, int]) -> bool:
    """Vérifie si les stocks couvrent le coût d'un bâtiment.

    Args:
        cost: Dict {resource_key: montant_requis}.
    """
    return all(get_stock(state, resource) >= amount for resource, amount in cost.items())


def pay_cost(state: ResourceState, cost: dict[str, int]) -> None:
    """Déduit le coût d'un bâtiment des stocks. Suppose can_afford() == True."""
    for resource, amount in cost.items():
        set_stock(state, resource, get_stock(state, resource) - amount)
