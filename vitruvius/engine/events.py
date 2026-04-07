"""Événements aléatoires : Incendie, Sécheresse, Bonne récolte, Immigration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel, model_validator

from vitruvius.engine.resources import ResourceState, clamp_stocks_to_capacity
from vitruvius.engine.services import compute_coverage_grid

if TYPE_CHECKING:
    from vitruvius.engine.buildings import BuildingConfig
    from vitruvius.engine.grid import Grid
    from vitruvius.engine.population import HouseLevelConfig, HouseState


class EventEffect(BaseModel):
    """Effet d'un événement."""

    type: str
    count: int | None = None
    immune_unique: bool = False
    modifier: float | None = None
    min_amount: int | None = None
    max_amount: int | None = None


class EventPrevention(BaseModel):
    """Mécanisme de prévention d'un événement."""

    building: str
    risk_divisor: int


class EventConfig(BaseModel):
    """Configuration d'un événement."""

    display_name: str
    probability: float
    duration: int
    effect: EventEffect
    prevention: EventPrevention | None = None


class EventsConfig(BaseModel):
    """Configuration complète des événements chargée depuis events.yaml."""

    events: dict[str, EventConfig]

    @model_validator(mode="after")
    def validate_probabilities(self) -> "EventsConfig":
        total = sum(e.probability for e in self.events.values())
        if total >= 1.0:
            raise ValueError(
                f"La somme des probabilités d'événements ({total:.3f}) doit être < 1.0."
            )
        return self


# ---------------------------------------------------------------------------
# Runtime state
# ---------------------------------------------------------------------------


@dataclass
class ActiveEvent:
    """Événement actif avec timer décrémenté chaque tour."""

    event_type: str                              # clé depuis events.yaml
    turns_remaining: int                         # supprimé quand atteint 0
    data: dict[str, object] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Tirage
# ---------------------------------------------------------------------------


def draw_event(
    events_config: dict[str, EventConfig],
    rng: np.random.Generator,
) -> str | None:
    """Tire un événement selon les plages cumulatives de probabilité.

    Un seul nombre aléatoire r = rng.random() est tiré par tour.
    Les événements sont évalués dans l'ordre d'insertion du dict (YAML).
    Si r dépasse la somme de toutes les probabilités → aucun événement.

    Args:
        events_config: Dict des configs d'événements, dans l'ordre du YAML.
        rng: Générateur aléatoire seedé.

    Returns:
        Clé de l'événement tiré (ex. "fire"), ou None si hors plage.
    """
    r = rng.random()
    cumulative = 0.0
    for event_type, event_cfg in events_config.items():
        cumulative += event_cfg.probability
        if r < cumulative:
            return event_type
    return None


# ---------------------------------------------------------------------------
# Effets des événements
# ---------------------------------------------------------------------------


def _apply_fire(
    event_config: EventConfig,
    rng: np.random.Generator,
    grid: Grid,
    resource_state: ResourceState,
    building_configs: dict[str, BuildingConfig],
) -> ActiveEvent | None:
    """Détruit un bâtiment non-unique au hasard, sans remboursement.

    Sélectionne une cible parmi les bâtiments non-uniques. Si une préfecture
    couvre la cible, un jet supplémentaire peut faire fizzler l'incendie.
    La destruction suit le même chemin que remove_building + clamp, mais
    SANS refund_cost (contrairement à try_demolish).

    Args:
        event_config: Config de l'événement incendie.
        rng: Générateur aléatoire seedé.
        grid: Grille de jeu (modifiée en place si destruction).
        resource_state: État des ressources (pour clamp stockage).
        building_configs: Configs des bâtiments.

    Returns:
        ActiveEvent si un bâtiment est détruit, None si fizzle ou grille vide.
    """
    # Batiments eligibles : exclure les uniques (forum, obelisque)
    eligible = [
        (origin, pb)
        for origin, pb in grid.placed_buildings.items()
        if not building_configs[pb.building_id].unique
    ]
    if not eligible:
        return None

    idx = int(rng.integers(0, len(eligible)))
    (ox, oy), pb = eligible[idx]
    bid = pb.building_id

    # Check prevention (prefecture couvre-t-elle la cible ?)
    if event_config.prevention is not None:
        coverage = compute_coverage_grid(grid, building_configs, resource_state)
        security_tiles = coverage.get("security", set())
        w, h = pb.size
        target_tiles = {(ox + dx, oy + dy) for dy in range(h) for dx in range(w)}
        if target_tiles & security_tiles:
            if rng.random() < 1.0 - 1.0 / event_config.prevention.risk_divisor:
                return None  # fizzle

    # Destruction sans remboursement
    destroyed_cfg = building_configs[bid]
    grid.remove_building(ox, oy)
    # Clamp uniquement si le bâtiment détruit avait du stockage — sinon, cap==0
    # zérerait des ressources qui n'étaient pas stockées dans ce bâtiment.
    if destroyed_cfg.storage is not None:
        clamp_stocks_to_capacity(resource_state, grid.placed_buildings, building_configs)

    return ActiveEvent(
        event_type="fire",
        turns_remaining=event_config.duration,
        data={"destroyed_building": bid, "destroyed_at": (ox, oy)},
    )


def _apply_drought(
    event_config: EventConfig,
    active_events: list[ActiveEvent],
) -> ActiveEvent:
    """Crée une sécheresse ou remet son timer à 3 si déjà active.

    Pas d'empilement : un seul ActiveEvent drought à la fois, modifier non doublé.

    Args:
        event_config: Config de l'événement sécheresse.
        active_events: Liste des événements actifs (modifiée en place).

    Returns:
        L'ActiveEvent créé ou réinitialisé.
    """
    for event in active_events:
        if event.event_type == "drought":
            event.turns_remaining = event_config.duration
            return event

    new_event = ActiveEvent(
        event_type="drought",
        turns_remaining=event_config.duration,
        data={"modifier": event_config.effect.modifier},
    )
    active_events.append(new_event)
    return new_event


def apply_event(
    event_type: str,
    event_config: EventConfig,
    active_events: list[ActiveEvent],
    rng: np.random.Generator,
    grid: Grid,
    resource_state: ResourceState,
    building_configs: dict[str, BuildingConfig],
    houses: dict[tuple[int, int], HouseState] | None = None,
    house_levels: list[HouseLevelConfig] | None = None,
) -> ActiveEvent | None:
    """Applique l'effet d'un événement tiré et crée l'ActiveEvent correspondant.

    Args:
        event_type: Clé de l'événement (ex. "fire").
        event_config: Config de l'événement.
        active_events: Liste des événements actifs (modifiée en place).
        rng: Générateur aléatoire seedé.
        grid: Grille de jeu.
        resource_state: État des ressources.
        building_configs: Configs des bâtiments.
        houses: États des maisons (requis pour immigration).
        house_levels: Configs des niveaux de maison (requis pour immigration).

    Returns:
        L'ActiveEvent créé, ou None si l'événement fizzle (ex. incendie sans cible).
    """
    if event_type == "fire":
        new_event = _apply_fire(event_config, rng, grid, resource_state, building_configs)
        if new_event is not None:
            active_events.append(new_event)
        return new_event

    if event_type == "drought":
        return _apply_drought(event_config, active_events)

    if event_type == "good_harvest":
        new_event = ActiveEvent(
            event_type="good_harvest",
            turns_remaining=event_config.duration,
            data={"modifier": event_config.effect.modifier},
        )
        active_events.append(new_event)
        return new_event

    if event_type == "immigration":
        from vitruvius.engine.population import apply_immigration

        amount = int(rng.integers(
            event_config.effect.min_amount,
            event_config.effect.max_amount + 1,
        ))
        settled = 0
        if houses is not None and house_levels is not None:
            settled = apply_immigration(houses, amount, house_levels)
        new_event = ActiveEvent(
            event_type="immigration",
            turns_remaining=event_config.duration,
            data={"amount": amount, "settled": settled},
        )
        active_events.append(new_event)
        return new_event

    return None


# ---------------------------------------------------------------------------
# Timer
# ---------------------------------------------------------------------------


def tick_events(active_events: list[ActiveEvent]) -> None:
    """Décrémente turns_remaining et supprime les événements expirés (in-place).

    Args:
        active_events: Liste des événements actifs, modifiée en place.
    """
    for event in active_events:
        event.turns_remaining -= 1
    active_events[:] = [e for e in active_events if e.turns_remaining > 0]


# ---------------------------------------------------------------------------
# Agrégation
# ---------------------------------------------------------------------------


def get_farm_modifier(active_events: list[ActiveEvent]) -> float:
    """Somme les modifiers de production actifs (sécheresse, bonne récolte).

    Args:
        active_events: Liste des événements actifs.

    Returns:
        Somme des modifiers. 0.0 si aucun modificateur actif.
    """
    return sum(
        float(e.data["modifier"])
        for e in active_events
        if "modifier" in e.data
    )


# ---------------------------------------------------------------------------
# Point d'entrée pour turn.py
# ---------------------------------------------------------------------------


def process_events(
    events_config: dict[str, EventConfig],
    active_events: list[ActiveEvent],
    rng: np.random.Generator,
    grid: Grid,
    resource_state: ResourceState,
    building_configs: dict[str, BuildingConfig],
    houses: dict[tuple[int, int], HouseState] | None = None,
    house_levels: list[HouseLevelConfig] | None = None,
) -> ActiveEvent | None:
    """Orchestre l'étape 11 du tour : tick → tirage → application.

    L'ordre tick-puis-draw garantit qu'un événement de duration=N affecte
    exactement N tours de production (step 3 du tour suivant).

    Args:
        events_config: Dict des configs d'événements.
        active_events: Liste des événements actifs (modifiée en place).
        rng: Générateur aléatoire seedé.
        grid: Grille de jeu.
        resource_state: État des ressources.
        building_configs: Configs des bâtiments.
        houses: États des maisons (pour immigration).
        house_levels: Configs des niveaux de maison (pour immigration).

    Returns:
        Le nouvel ActiveEvent créé ce tour, ou None si aucun événement.
    """
    tick_events(active_events)
    event_type = draw_event(events_config, rng)
    if event_type is None:
        return None
    return apply_event(
        event_type,
        events_config[event_type],
        active_events,
        rng,
        grid,
        resource_state,
        building_configs,
        houses,
        house_levels,
    )
