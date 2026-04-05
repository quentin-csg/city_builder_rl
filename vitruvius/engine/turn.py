"""Boucle de tour : production → taxes → consommation → satisfaction → évolution maisons → événements → victoire."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from vitruvius.engine.buildings import try_demolish, try_place_building
from vitruvius.engine.events import ActiveEvent, get_farm_modifier, process_events
from vitruvius.engine.game_state import GameState
from vitruvius.engine.population import (
    HouseState,
    apply_exodus,
    apply_famine_loss,
    apply_growth,
    compute_global_satisfaction,
    compute_house_taxes,
    evolve_houses,
)
from vitruvius.engine.resources import (
    apply_maintenance,
    apply_passive_income,
    apply_production,
    apply_taxes,
    apply_wheat_consumption,
)
from vitruvius.engine.services import compute_coverage
from vitruvius.engine.victory import check_defeat, compute_city_level

if TYPE_CHECKING:
    from vitruvius.config import GameConfig


# ---------------------------------------------------------------------------
# Structures d'action et de résultat
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Action:
    """Action choisie par le joueur ou l'agent pour un tour.

    Attributes:
        type: Type d'action parmi "place", "demolish", "do_nothing".
        building_id: Identifiant du bâtiment (obligatoire pour "place").
        x: Colonne cible (0-indexed).
        y: Ligne cible (0-indexed).
    """

    type: Literal["place", "demolish", "do_nothing"]
    building_id: str | None = None
    x: int = 0
    y: int = 0


@dataclass
class TurnResult:
    """Résultat d'un tour, utilisé par reward.py pour le calcul du reward RL.

    Tous les champs sont des scalaires ou des dicts simples — pas de références
    au GameState pour éviter les effets de bord.
    """

    production: dict[str, int]
    taxes_collected: float
    maintenance_paid: float
    passive_income: float
    famine_count: int         # maisons marquées famine à l'étape 5
    famine_pop_lost: int      # pop perdue à l'étape 6
    evolved: int              # maisons qui ont monté de niveau
    regressed: int            # maisons qui ont descendu de niveau
    growth: int               # habitants ajoutés (croissance)
    exodus: int               # habitants perdus (exode)
    new_event: ActiveEvent | None
    global_satisfaction: float
    total_population: int
    city_level: int
    done: bool
    victory: bool
    defeat: bool
    bankrupt: bool            # denarii < -500 après étape 4


# ---------------------------------------------------------------------------
# Boucle de tour principale
# ---------------------------------------------------------------------------


def step(game_state: GameState, config: GameConfig, action: Action) -> TurnResult:
    """Exécute un tour complet en 13 étapes. Modifie game_state en place.

    Args:
        game_state: État courant du jeu (modifié en place).
        config: Configuration unifiée du jeu.
        action: Action choisie pour ce tour.

    Returns:
        TurnResult contenant tous les signaux nécessaires au reward RL.
    """
    gs = game_state
    bldg = config.buildings.buildings
    house_levels = config.needs.house_levels

    # ------------------------------------------------------------------
    # Étape 1-2 : Action
    # ------------------------------------------------------------------
    if action.type == "place" and action.building_id is not None:
        success = try_place_building(
            gs.grid, gs.resource_state, action.building_id, action.x, action.y, bldg
        )
        if success:
            cfg = bldg[action.building_id]
            if cfg.special_effect is not None and cfg.special_effect.is_housing:
                origin = (action.x, action.y)
                gs.houses[origin] = HouseState(
                    origin=origin, level=0, population=0
                )

    elif action.type == "demolish":
        demolished = try_demolish(
            gs.grid, gs.resource_state, action.x, action.y, bldg
        )
        if demolished is not None:
            cfg = bldg[demolished.building_id]
            if cfg.special_effect is not None and cfg.special_effect.is_housing:
                gs.houses.pop((demolished.x, demolished.y), None)

    # do_nothing : aucune opération

    # ------------------------------------------------------------------
    # Étape 3 : Production
    # ------------------------------------------------------------------
    farm_mod = get_farm_modifier(gs.active_events)
    production = apply_production(
        gs.resource_state, gs.grid.placed_buildings, bldg, config.resources,
        farm_modifier=farm_mod,
    )

    # ------------------------------------------------------------------
    # Étape 4 : Taxes + Entretien
    # ------------------------------------------------------------------
    passive = apply_passive_income(gs.resource_state, config.resources.passive_income)

    house_tax_list = compute_house_taxes(gs.houses, house_levels)
    base_taxes = apply_taxes(gs.resource_state, house_tax_list)

    # Bonus fiscaux ADDITIFS (design_decisions.md) :
    # forum special_effect.tax_bonus (+0.15) + satisfaction > 75% (+0.10)
    tax_bonus = 0.0
    for pb in gs.grid.placed_buildings.values():
        se = bldg[pb.building_id].special_effect
        if se is not None and se.tax_bonus > 0.0:
            tax_bonus += se.tax_bonus
    if gs.global_satisfaction > 0.75:
        tax_bonus += 0.10

    bonus_amount = base_taxes * tax_bonus
    gs.resource_state.denarii += bonus_amount
    taxes_collected = base_taxes + bonus_amount

    maintenance = apply_maintenance(gs.resource_state, gs.grid.placed_buildings, bldg)

    # Suivi banqueroute : denarii < -500 après prise en compte des taxes/maintenance
    bankrupt = gs.resource_state.denarii < -500
    if bankrupt:
        gs.consecutive_bankrupt_turns += 1
    else:
        gs.consecutive_bankrupt_turns = 0

    # ------------------------------------------------------------------
    # Étape 5 : Consommation de blé (FIFO, tout-ou-rien par maison)
    # ------------------------------------------------------------------
    houses_ordered = list(gs.houses.values())  # ordre insertion = FIFO
    houses_pop = [h.population for h in houses_ordered]
    famine_flags = apply_wheat_consumption(gs.resource_state, houses_pop)
    famine_count = 0
    for h, famine in zip(houses_ordered, famine_flags):
        h.famine = famine
        if famine:
            famine_count += 1

    # ------------------------------------------------------------------
    # Étape 6 : Pertes de population dues à la famine
    # ------------------------------------------------------------------
    famine_pop_lost = apply_famine_loss(gs.houses)

    # ------------------------------------------------------------------
    # Étape 7 : Services (recalcul des rayons d'influence)
    # ------------------------------------------------------------------
    coverage = compute_coverage(gs.grid, bldg, gs.resource_state)

    # ------------------------------------------------------------------
    # Étape 8 : Satisfaction
    # ------------------------------------------------------------------
    global_sat = compute_global_satisfaction(gs.houses, coverage, house_levels, gs.grid)
    gs.global_satisfaction = global_sat

    # ------------------------------------------------------------------
    # Étape 9 : Évolution / Régression des maisons
    # ------------------------------------------------------------------
    evolved, regressed = evolve_houses(gs.houses, coverage, house_levels)

    # ------------------------------------------------------------------
    # Étape 10 : Croissance naturelle et exode
    # ------------------------------------------------------------------
    growth = apply_growth(gs.houses, global_sat, house_levels)
    exodus = apply_exodus(gs.houses, global_sat)

    # ------------------------------------------------------------------
    # Étape 11 : Événements (tick → tirage → application)
    # ------------------------------------------------------------------
    new_event = process_events(
        config.events.events,
        gs.active_events,
        gs.rng,
        gs.grid,
        gs.resource_state,
        bldg,
        houses=gs.houses,
        house_levels=house_levels,
    )

    # Sync : un incendie peut avoir détruit un housing sans mettre à jour houses
    orphans = [o for o in gs.houses if o not in gs.grid.placed_buildings]
    for o in orphans:
        del gs.houses[o]

    # ------------------------------------------------------------------
    # Étape 12 : Victoire / Défaite
    # ------------------------------------------------------------------
    total_pop = sum(h.population for h in gs.houses.values())
    city_level = compute_city_level(
        total_pop, global_sat, gs.grid._placed_ids, config.city_levels.city_levels
    )
    gs.city_level = city_level

    defeat = check_defeat(total_pop, gs.consecutive_bankrupt_turns, has_housing=len(gs.houses) > 0)
    victory_flag = city_level >= 5

    gs.done = defeat or victory_flag
    gs.victory = victory_flag

    # ------------------------------------------------------------------
    # Étape 13 : Finalisation
    # ------------------------------------------------------------------
    gs.resource_state.denarii = round(gs.resource_state.denarii, 2)
    gs.turn += 1

    return TurnResult(
        production=production,
        taxes_collected=taxes_collected,
        maintenance_paid=maintenance,
        passive_income=passive,
        famine_count=famine_count,
        famine_pop_lost=famine_pop_lost,
        evolved=evolved,
        regressed=regressed,
        growth=growth,
        exodus=exodus,
        new_event=new_event,
        global_satisfaction=global_sat,
        total_population=total_pop,
        city_level=city_level,
        done=gs.done,
        victory=victory_flag,
        defeat=defeat,
        bankrupt=bankrupt,
    )
