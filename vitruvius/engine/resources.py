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
    farm_modifier: float = 0.0,
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
        farm_modifier: Modificateur multiplicatif sur la production de blé.
            Appliqué uniquement au blé : max(0, floor(amount * (1 + farm_modifier))).
            Valeur 0.0 = aucun effet.

    Returns:
        Dict {resource_key: montant_effectivement_ajouté}.
    """
    # Précalcul unique des capacités de stockage : évite O(N×M) → O(N+M)
    # storage_caps[r] = capacité totale pour la ressource r (absent = 0 = pas de stockage)
    storage_caps: dict[str, int] = {}
    for pb in placed_buildings.values():
        cfg = building_configs[pb.building_id]
        if cfg.storage is not None:
            r = cfg.storage.resource
            storage_caps[r] = storage_caps.get(r, 0) + cfg.storage.capacity

    produced: dict[str, int] = {}

    for pb in placed_buildings.values():
        cfg = building_configs[pb.building_id]
        if cfg.production is None:
            continue

        resource = cfg.production.resource
        amount = cfg.production.amount

        # Modificateur sécheresse/bonne récolte sur le blé uniquement
        if resource == "wheat" and farm_modifier != 0.0:
            amount = max(0, math.floor(amount * (1 + farm_modifier)))

        # Vérification : stockage requis pour les ressources non-denarii (O(1))
        if resources_config.resources[resource].storage_building is not None:
            if storage_caps.get(resource, 0) == 0:
                produced.setdefault(resource, 0)
                continue

        # cap = None pour denarii (absent de storage_caps) → illimité
        cap = storage_caps.get(resource)
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


def refund_cost(state: ResourceState, cost: dict[str, int], ratio: float = 0.5) -> None:
    """Rembourse une fraction (floor) du coût d'un bâtiment démoli.

    Args:
        state: État des ressources à modifier.
        cost: Coût original du bâtiment (dict {resource_key: montant}).
        ratio: Fraction remboursée. Défaut 0.5 (50%).
    """
    for resource, amount in cost.items():
        refund = math.floor(amount * ratio)
        set_stock(state, resource, get_stock(state, resource) + refund)


def clamp_stocks_to_capacity(
    state: ResourceState,
    placed_buildings: dict[tuple[int, int], PlacedBuilding],
    building_configs: dict[str, BuildingConfig],
) -> dict[str, int]:
    """Réduit les stocks aux capacités de stockage actuelles (après démolition d'un entrepôt).

    Denarii n'est jamais limité. Les ressources physiques (blé, bois, marbre) sont
    capées par la capacité totale des bâtiments de stockage encore présents.

    Args:
        state: État des ressources à modifier.
        placed_buildings: Bâtiments actuellement sur la grille (après démolition).
        building_configs: Configs des bâtiments.

    Returns:
        Dict {resource_key: montant_perdu} pour les ressources qui ont été réduites.
    """
    lost: dict[str, int] = {}
    for key in _RESOURCE_KEYS - {"denarii"}:
        cap = compute_storage_cap(key, placed_buildings, building_configs)
        if cap is not None:
            current = get_stock(state, key)
            if current > cap:
                lost[key] = int(current - cap)
                set_stock(state, key, cap)
    return lost
