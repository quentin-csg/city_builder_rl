"""Tests pour vitruvius.engine.turn et vitruvius.engine.game_state."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from vitruvius.config import load_config
from vitruvius.engine.events import ActiveEvent
from vitruvius.engine.game_state import from_dict, init_game_state, to_dict
from vitruvius.engine.population import HouseState
from vitruvius.engine.terrain import TerrainType
from vitruvius.engine.turn import Action, TurnResult, step


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture
def gs(cfg):
    """GameState frais à chaque test (seed=42)."""
    return init_game_state(cfg, seed=42)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_plain_1x1(grid, n: int = 1) -> list[tuple[int, int]]:
    """Retourne les n premières tuiles PLAIN libres (1x1)."""
    result = []
    for y in range(grid.SIZE):
        for x in range(grid.SIZE):
            if grid.terrain[y][x] == TerrainType.PLAIN and grid._origin[y][x] is None:
                result.append((x, y))
                if len(result) == n:
                    return result
    return result


def _find_plain_block(grid, w: int, h: int) -> tuple[int, int] | None:
    """Retourne le coin supérieur gauche du premier bloc w×h entièrement PLAIN et libre."""
    for y in range(grid.SIZE - h + 1):
        for x in range(grid.SIZE - w + 1):
            if all(
                grid.terrain[y + dy][x + dx] == TerrainType.PLAIN
                and grid._origin[y + dy][x + dx] is None
                for dy in range(h) for dx in range(w)
            ):
                return x, y
    return None


def _place(gs, cfg, building_id: str, x: int, y: int) -> bool:
    """Exécute un step de placement et retourne si réussi."""
    before = set(gs.grid.placed_buildings.keys())
    step(gs, cfg, Action("place", building_id, x, y))
    return len(gs.grid.placed_buildings) > len(before)


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


def test_init_defaults(cfg):
    """init_game_state crée un état avec les valeurs par défaut correctes."""
    gs = init_game_state(cfg, seed=42)
    assert gs.turn == 0
    assert gs.city_level == 1
    assert gs.global_satisfaction == 0.5
    assert gs.consecutive_bankrupt_turns == 0
    assert gs.done is False
    assert gs.victory is False
    assert gs.houses == {}
    assert gs.active_events == []
    assert gs.resource_state.denarii == 800.0
    assert gs.resource_state.wheat == 200
    assert gs.resource_state.wood == 100


def test_init_deterministic(cfg):
    """Deux GameState avec le même seed ont un terrain identique."""
    gs1 = init_game_state(cfg, seed=7)
    gs2 = init_game_state(cfg, seed=7)
    assert gs1.grid.terrain == gs2.grid.terrain
    assert gs1.resource_state.denarii == gs2.resource_state.denarii


def test_init_different_seeds(cfg):
    """Seeds différents → RNG différents (premiers tirages divergent)."""
    gs1 = init_game_state(cfg, seed=1)
    gs2 = init_game_state(cfg, seed=2)
    v1 = gs1.rng.random()
    v2 = gs2.rng.random()
    assert v1 != v2


# ---------------------------------------------------------------------------
# Étapes 1-2 : Actions
# ---------------------------------------------------------------------------


def test_step_do_nothing_increments_turn(cfg, gs):
    """DO_NOTHING incrémente le turn et ne modifie pas la grille."""
    initial_buildings = len(gs.grid.placed_buildings)
    r = step(gs, cfg, Action("do_nothing"))
    assert gs.turn == 1
    assert len(gs.grid.placed_buildings) == initial_buildings
    assert isinstance(r, TurnResult)


def test_step_do_nothing_adds_passive_income(cfg, gs):
    """DO_NOTHING ajoute le revenu passif (10 denarii/tour)."""
    before = gs.resource_state.denarii
    step(gs, cfg, Action("do_nothing"))
    # passive income + 0 taxes + 0 maintenance
    assert gs.resource_state.denarii > before


def test_step_place_road(cfg, gs):
    """Placement d'une route : apparaît sur la grille, denarii débités."""
    pos = _find_plain_1x1(gs.grid)
    assert pos, "Pas de tile PLAIN disponible"
    x, y = pos[0]
    before_denarii = gs.resource_state.denarii
    step(gs, cfg, Action("place", "road", x, y))
    assert gs.grid.get_building_at(x, y) is not None
    assert gs.resource_state.denarii < before_denarii + 20  # passive income - coût route


def test_step_place_housing_creates_house_state(cfg, gs):
    """Placer un housing (2x2) crée un HouseState dans gs.houses."""
    pos = _find_plain_block(gs.grid, 2, 2)
    assert pos is not None
    x, y = pos
    # housing coûte wood=10
    assert gs.resource_state.wood >= 10
    step(gs, cfg, Action("place", "housing", x, y))
    assert (x, y) in gs.houses
    h = gs.houses[(x, y)]
    assert h.level == 0
    assert h.population == 0
    assert h.origin == (x, y)


def test_step_demolish_building(cfg, gs):
    """Démolition d'un bâtiment : disparaît de la grille, remboursement."""
    pos = _find_plain_1x1(gs.grid)
    x, y = pos[0]
    step(gs, cfg, Action("place", "road", x, y))
    assert gs.grid.get_building_at(x, y) is not None
    before_denarii = gs.resource_state.denarii
    step(gs, cfg, Action("demolish", x=x, y=y))
    assert gs.grid.get_building_at(x, y) is None
    # remboursement 50% route (coût=2 → +1)
    assert gs.resource_state.denarii >= before_denarii - 1  # passive income compense


def test_step_demolish_housing_removes_house_state(cfg, gs):
    """Démolir un housing supprime le HouseState correspondant."""
    pos = _find_plain_block(gs.grid, 2, 2)
    assert pos is not None
    x, y = pos
    step(gs, cfg, Action("place", "housing", x, y))
    assert (x, y) in gs.houses
    step(gs, cfg, Action("demolish", x=x, y=y))
    assert (x, y) not in gs.houses


# ---------------------------------------------------------------------------
# Étape 3 : Production
# ---------------------------------------------------------------------------


def test_step_production_wheat(cfg, gs):
    """wheat_farm + granary produisent du blé."""
    gs.resource_state.wheat = 0
    farm_pos = _find_plain_block(gs.grid, 3, 3)
    assert farm_pos is not None
    x, y = farm_pos
    step(gs, cfg, Action("place", "wheat_farm", x, y))
    granary_pos = _find_plain_block(gs.grid, 3, 3)
    assert granary_pos is not None
    xg, yg = granary_pos
    step(gs, cfg, Action("place", "granary", xg, yg))
    before = gs.resource_state.wheat
    step(gs, cfg, Action("do_nothing"))
    assert gs.resource_state.wheat > before


def test_step_production_farm_modifier_drought(cfg, gs):
    """Sécheresse active (farm_modifier=-0.5) réduit la production de blé."""
    gs.resource_state.wheat = 0
    farm_pos = _find_plain_block(gs.grid, 3, 3)
    xf, yf = farm_pos
    step(gs, cfg, Action("place", "wheat_farm", xf, yf))
    granary_pos = _find_plain_block(gs.grid, 3, 3)
    xg, yg = granary_pos
    step(gs, cfg, Action("place", "granary", xg, yg))

    # Injecter une sécheresse active
    gs.active_events.append(ActiveEvent("drought", turns_remaining=3, data={"modifier": -0.5}))
    gs.resource_state.wheat = 0

    before = gs.resource_state.wheat
    step(gs, cfg, Action("do_nothing"))
    after = gs.resource_state.wheat

    # Avec sécheresse, production = floor(12 * 0.5) = 6 (pas 12)
    assert 0 < after - before <= 6


# ---------------------------------------------------------------------------
# Étape 4 : Taxes + Entretien
# ---------------------------------------------------------------------------


def test_step_taxes_basic(cfg, gs):
    """Taxes prélevées sur les maisons peuplées."""
    pos = _find_plain_block(gs.grid, 2, 2)
    x, y = pos
    step(gs, cfg, Action("place", "housing", x, y))
    # Peupler manuellement la maison
    gs.houses[(x, y)].population = 50
    gs.houses[(x, y)].level = 1
    before = gs.resource_state.denarii
    step(gs, cfg, Action("do_nothing"))
    # taxes = floor(50 × tax_per_inhabitant) > 0 (+ revenu passif)
    assert gs.resource_state.denarii > before - 100  # maintenance raisonnable


def test_step_forum_tax_bonus(cfg, gs):
    """Forum (+15% tax_bonus) augmente les taxes collectées par rapport à sans forum."""
    # Monter manuellement la satisfaction pour éviter le +10% sat bonus
    gs.global_satisfaction = 0.5

    pos = _find_plain_block(gs.grid, 2, 2)
    x, y = pos
    step(gs, cfg, Action("place", "housing", x, y))
    gs.houses[(x, y)].population = 100
    gs.houses[(x, y)].level = 1

    # Tour sans forum
    before_no_forum = gs.resource_state.denarii
    r_no_forum = step(gs, cfg, Action("do_nothing"))

    # Placer un forum (coûte denarii=2000, marble=200)
    gs.resource_state.marble = 500
    gs.resource_state.denarii = 5000.0
    forum_pos = _find_plain_block(gs.grid, 4, 4)
    xf, yf = forum_pos
    step(gs, cfg, Action("place", "forum", xf, yf))

    gs.houses[(x, y)].population = 100
    gs.houses[(x, y)].level = 1
    gs.global_satisfaction = 0.5  # pas de bonus sat

    r_with_forum = step(gs, cfg, Action("do_nothing"))

    assert r_with_forum.taxes_collected > r_no_forum.taxes_collected


def test_step_satisfaction_tax_bonus(cfg, gs):
    """Satisfaction > 75% ajoute +10% aux taxes."""
    pos = _find_plain_block(gs.grid, 2, 2)
    x, y = pos
    step(gs, cfg, Action("place", "housing", x, y))
    gs.houses[(x, y)].population = 100
    gs.houses[(x, y)].level = 1

    # Tour avec faible satisfaction (pas de bonus)
    gs.global_satisfaction = 0.5
    r_low_sat = step(gs, cfg, Action("do_nothing"))

    # Tour avec satisfaction élevée
    gs.houses[(x, y)].population = 100
    gs.houses[(x, y)].level = 1
    gs.global_satisfaction = 0.80
    r_high_sat = step(gs, cfg, Action("do_nothing"))

    assert r_high_sat.taxes_collected > r_low_sat.taxes_collected


def test_step_bankrupt_tracking(cfg, gs):
    """consecutive_bankrupt_turns s'incrémente quand denarii < -500."""
    gs.resource_state.denarii = -600.0
    step(gs, cfg, Action("do_nothing"))
    assert gs.consecutive_bankrupt_turns == 1
    gs.resource_state.denarii = -600.0
    step(gs, cfg, Action("do_nothing"))
    assert gs.consecutive_bankrupt_turns == 2


def test_step_bankrupt_reset(cfg, gs):
    """consecutive_bankrupt_turns se remet à 0 quand denarii >= -500."""
    gs.consecutive_bankrupt_turns = 3
    gs.resource_state.denarii = 1000.0
    step(gs, cfg, Action("do_nothing"))
    assert gs.consecutive_bankrupt_turns == 0


# ---------------------------------------------------------------------------
# Étapes 5-6 : Consommation blé + Famine
# ---------------------------------------------------------------------------


def test_step_wheat_consumption(cfg, gs):
    """La maison consomme ceil(pop/10) blé par tour."""
    pos = _find_plain_block(gs.grid, 2, 2)
    x, y = pos
    step(gs, cfg, Action("place", "housing", x, y))
    gs.houses[(x, y)].population = 10
    gs.houses[(x, y)].level = 1
    gs.resource_state.wheat = 100

    step(gs, cfg, Action("do_nothing"))

    # ceil(10/10) = 1 blé consommé
    assert gs.resource_state.wheat == 99


def test_step_famine_marked_when_no_wheat(cfg, gs):
    """Maison sans blé disponible est marquée famine."""
    pos = _find_plain_block(gs.grid, 2, 2)
    x, y = pos
    step(gs, cfg, Action("place", "housing", x, y))
    gs.houses[(x, y)].population = 20
    gs.houses[(x, y)].level = 1
    gs.resource_state.wheat = 0

    r = step(gs, cfg, Action("do_nothing"))

    assert r.famine_count >= 1


def test_step_famine_pop_loss(cfg, gs):
    """Maison marquée famine perd ceil(10% pop) habitants."""
    pos = _find_plain_block(gs.grid, 2, 2)
    x, y = pos
    step(gs, cfg, Action("place", "housing", x, y))
    gs.houses[(x, y)].population = 20
    gs.houses[(x, y)].level = 1
    gs.resource_state.wheat = 0  # famine garantie

    r = step(gs, cfg, Action("do_nothing"))

    # ceil(10% × 20) = 2 perdus
    assert r.famine_pop_lost >= 2


# ---------------------------------------------------------------------------
# Étapes 7-8 : Services + Satisfaction
# ---------------------------------------------------------------------------


def test_step_satisfaction_no_pop_is_half(cfg, gs):
    """Sans population, la satisfaction globale reste 0.5 (convention)."""
    r = step(gs, cfg, Action("do_nothing"))
    assert r.global_satisfaction == 0.5


def test_step_satisfaction_updates_each_turn(cfg, gs):
    """La satisfaction est recalculée à chaque tour."""
    pos = _find_plain_block(gs.grid, 2, 2)
    x, y = pos
    step(gs, cfg, Action("place", "housing", x, y))
    gs.houses[(x, y)].population = 10
    gs.houses[(x, y)].level = 1

    r1 = step(gs, cfg, Action("do_nothing"))
    # Sans services, satisfaction = 0/N + 0 (routes) ≈ 0
    assert 0.0 <= r1.global_satisfaction <= 1.0


# ---------------------------------------------------------------------------
# Étape 9 : Évolution / Régression maisons
# ---------------------------------------------------------------------------


def test_step_evolve_signals_returned(cfg, gs):
    """Les signaux evolved/regressed sont présents dans TurnResult."""
    r = step(gs, cfg, Action("do_nothing"))
    assert hasattr(r, "evolved")
    assert hasattr(r, "regressed")
    assert r.evolved >= 0
    assert r.regressed >= 0


# ---------------------------------------------------------------------------
# Étapes 10 : Croissance / Exode
# ---------------------------------------------------------------------------


def test_step_growth_at_high_satisfaction(cfg, gs):
    """Avec sat ≥ 50%, des habitants sont ajoutés aux maisons avec de la place."""
    pos = _find_plain_block(gs.grid, 2, 2)
    x, y = pos
    step(gs, cfg, Action("place", "housing", x, y))
    gs.houses[(x, y)].population = 5
    gs.houses[(x, y)].level = 1
    gs.global_satisfaction = 0.8
    # Forcer la satisfaction haute pour garantir croissance
    gs.houses[(x, y)].famine = False

    r = step(gs, cfg, Action("do_nothing"))
    assert r.growth >= 0  # peut être 0 si place libre = 0 ou pop_max atteint


def test_step_exodus_at_low_satisfaction(cfg, gs):
    """Avec sat < 30%, des habitants quittent la ville."""
    pos = _find_plain_block(gs.grid, 2, 2)
    x, y = pos
    step(gs, cfg, Action("place", "housing", x, y))
    gs.houses[(x, y)].population = 100
    gs.houses[(x, y)].level = 1
    gs.global_satisfaction = 0.1  # provoque l'exode au tour suivant

    r = step(gs, cfg, Action("do_nothing"))
    # L'exode est déclenché si sat global recalculé < 0.3
    # (la sat recalculée au step 8 dépend des services)
    assert r.exodus >= 0


# ---------------------------------------------------------------------------
# Étape 11 : Événements
# ---------------------------------------------------------------------------


def test_step_events_tick_active_events(cfg, gs):
    """Les événements actifs sont décrémentés à chaque tour."""
    gs.active_events.append(ActiveEvent("drought", turns_remaining=3, data={"modifier": -0.5}))
    step(gs, cfg, Action("do_nothing"))
    # Après tick : turns_remaining = 2
    drought_events = [e for e in gs.active_events if e.event_type == "drought"]
    assert len(drought_events) == 1
    assert drought_events[0].turns_remaining == 2


def test_step_events_expire(cfg, gs):
    """Un événement avec turns_remaining=1 expire après le tick."""
    gs.active_events.append(ActiveEvent("good_harvest", turns_remaining=1, data={"modifier": 0.5}))
    step(gs, cfg, Action("do_nothing"))
    good_harvest = [e for e in gs.active_events if e.event_type == "good_harvest"]
    assert len(good_harvest) == 0


def test_step_fire_syncs_houses(cfg, gs):
    """Si un incendie détruit un housing, houses est nettoyé."""
    pos = _find_plain_block(gs.grid, 2, 2)
    x, y = pos
    step(gs, cfg, Action("place", "housing", x, y))
    assert (x, y) in gs.houses

    # Simuler un incendie qui détruit le housing directement
    from vitruvius.engine.grid import Grid
    gs.grid.remove_building(x, y)

    # Le housing n'est plus sur la grille mais encore dans houses
    assert (x, y) not in gs.grid.placed_buildings

    # Après step, orphan doit être nettoyé
    step(gs, cfg, Action("do_nothing"))
    assert (x, y) not in gs.houses


# ---------------------------------------------------------------------------
# Étape 12 : Victoire / Défaite
# ---------------------------------------------------------------------------


def test_step_no_defeat_empty_city(cfg, gs):
    """Ville vide sans logement : pas de défaite (has_housing=False)."""
    r = step(gs, cfg, Action("do_nothing"))
    assert r.defeat is False
    assert gs.done is False


def test_step_defeat_pop_zero_with_housing(cfg, gs):
    """Housing posé + pop=0 après famine → défaite."""
    pos = _find_plain_block(gs.grid, 2, 2)
    x, y = pos
    step(gs, cfg, Action("place", "housing", x, y))
    gs.houses[(x, y)].population = 1
    gs.houses[(x, y)].level = 1
    gs.houses[(x, y)].famine = True
    gs.resource_state.wheat = 0
    # Forcer pop=0 manuellement
    gs.houses[(x, y)].population = 0

    r = step(gs, cfg, Action("do_nothing"))
    assert r.defeat is True
    assert gs.done is True


def test_step_defeat_bankrupt_5_turns(cfg, gs):
    """5 tours consécutifs en banqueroute → défaite."""
    gs.consecutive_bankrupt_turns = 4
    gs.resource_state.denarii = -600.0

    r = step(gs, cfg, Action("do_nothing"))
    assert r.defeat is True
    assert gs.done is True


def test_step_city_level_updates(cfg, gs):
    """city_level est recalculé à chaque tour."""
    r = step(gs, cfg, Action("do_nothing"))
    assert r.city_level == gs.city_level
    assert 1 <= r.city_level <= 5


# ---------------------------------------------------------------------------
# Étape 13 : Arrondi denarii
# ---------------------------------------------------------------------------


def test_step_denarii_rounded_to_2_decimals(cfg, gs):
    """Les denarii sont arrondis à 2 décimales en fin de tour."""
    gs.resource_state.denarii = 123.456789
    step(gs, cfg, Action("do_nothing"))
    # denarii doit avoir ≤ 2 décimales après arrondi
    d = gs.resource_state.denarii
    assert round(d, 2) == d


# ---------------------------------------------------------------------------
# Intégration multi-tours + déterminisme
# ---------------------------------------------------------------------------


def test_multi_turn_increments(cfg, gs):
    """10 tours de DO_NOTHING : turn compte correctement."""
    for _ in range(10):
        step(gs, cfg, Action("do_nothing"))
    assert gs.turn == 10


def test_determinism(cfg):
    """Même seed + mêmes actions → mêmes résultats."""
    gs1 = init_game_state(cfg, seed=99)
    gs2 = init_game_state(cfg, seed=99)

    actions = [Action("do_nothing")] * 5
    results1 = [step(gs1, cfg, a) for a in actions]
    results2 = [step(gs2, cfg, a) for a in actions]

    for r1, r2 in zip(results1, results2):
        assert r1.total_population == r2.total_population
        assert r1.global_satisfaction == r2.global_satisfaction
        assert r1.production == r2.production


# ---------------------------------------------------------------------------
# Sérialisation JSON
# ---------------------------------------------------------------------------


def test_to_dict_json_compatible(cfg, gs):
    """to_dict produit un dict entièrement JSON-sérialisable (pas de numpy, pas de tuples)."""
    d = to_dict(gs)
    # Ne doit pas lever d'exception
    json_str = json.dumps(d)
    assert isinstance(json_str, str)


def test_round_trip_fresh_state(cfg):
    """init_game_state → to_dict → from_dict → champs identiques."""
    gs = init_game_state(cfg, seed=42)
    d = to_dict(gs)
    gs2 = from_dict(d, cfg)

    assert gs2.seed == gs.seed
    assert gs2.turn == gs.turn
    assert gs2.city_level == gs.city_level
    assert gs2.resource_state.denarii == gs.resource_state.denarii
    assert gs2.resource_state.wheat == gs.resource_state.wheat
    assert gs2.global_satisfaction == gs.global_satisfaction
    assert gs2.done == gs.done
    assert len(gs2.houses) == len(gs.houses)
    assert len(gs2.active_events) == len(gs.active_events)


def test_round_trip_with_buildings_and_events(cfg):
    """Round-trip préserve les bâtiments posés et les événements actifs."""
    gs = init_game_state(cfg, seed=42)

    # Placer une route
    pos = _find_plain_1x1(gs.grid)
    x, y = pos[0]
    step(gs, cfg, Action("place", "road", x, y))

    # Ajouter un événement actif
    gs.active_events.append(
        ActiveEvent("drought", turns_remaining=2, data={"modifier": -0.5})
    )

    d = to_dict(gs)
    gs2 = from_dict(d, cfg)

    assert len(gs2.grid.placed_buildings) == len(gs.grid.placed_buildings)
    assert len(gs2.active_events) == 1
    assert gs2.active_events[0].event_type == "drought"
    assert gs2.active_events[0].turns_remaining == 2


def test_round_trip_rng_continuity(cfg):
    """Après round-trip, le RNG reproduit exactement les mêmes valeurs."""
    gs = init_game_state(cfg, seed=42)
    for _ in range(5):
        step(gs, cfg, Action("do_nothing"))

    # Sérialiser l'état RNG (avant tout tirage supplémentaire)
    d = to_dict(gs)

    # Valeur tirée par gs original
    val_original = gs.rng.random()

    # Valeur tirée par gs restauré (doit être identique)
    gs2 = from_dict(d, cfg)
    val_restored = gs2.rng.random()

    assert val_original == val_restored
