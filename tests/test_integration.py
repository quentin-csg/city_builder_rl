"""Tests d'intégration : parties complètes, invariants multi-tours, cas limites.

Ces tests ne se substituent pas aux tests unitaires : ils couvrent les interactions
entre modules sur de longues séquences d'actions que les tests unitaires ne peuvent
pas détecter (bugs émergents, effets de bord cross-tour, cohérence de l'état global).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from vitruvius.config import load_config
from vitruvius.engine.actions import (
    DO_NOTHING,
    compute_action_mask,
    get_building_order,
)
from vitruvius.engine.game_state import init_game_state
from vitruvius.engine.resources import compute_storage_cap
from vitruvius.engine.turn import Action, TurnResult, step


# ---------------------------------------------------------------------------
# Setup partagé
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def building_list(cfg):
    bl, _ = get_building_order(cfg)
    return bl


@pytest.fixture(scope="module")
def house_levels(cfg):
    return {hl.level: hl for hl in cfg.needs.house_levels}


# ---------------------------------------------------------------------------
# Vérificateur d'invariants — appelé après chaque step
# ---------------------------------------------------------------------------

def check_invariants(gs, result: TurnResult, cfg, label: str = "") -> list[str]:
    """Retourne la liste des violations d'invariants détectées."""
    violations = []
    rs = gs.resource_state
    bldg = cfg.buildings.buildings
    house_levels_map = {hl.level: hl for hl in cfg.needs.house_levels}

    def fail(msg):
        violations.append(f"[Tour {gs.turn}]{' ' + label if label else ''}: {msg}")

    # 1. Satisfaction bornée [0, 1]
    if not (0.0 <= gs.global_satisfaction <= 1.001):
        fail(f"satisfaction={gs.global_satisfaction:.4f} hors [0,1]")

    # 2. Ressources physiques non négatives
    if rs.wheat < 0:
        fail(f"wheat={rs.wheat} < 0")
    if rs.wood < 0:
        fail(f"wood={rs.wood} < 0")
    if rs.marble < 0:
        fail(f"marble={rs.marble} < 0")

    # 3. City level dans [1, 5]
    if not (1 <= gs.city_level <= 5):
        fail(f"city_level={gs.city_level} hors [1,5]")

    # 4. consecutive_bankrupt_turns non négatif
    if gs.consecutive_bankrupt_turns < 0:
        fail(f"consecutive_bankrupt_turns={gs.consecutive_bankrupt_turns} < 0")

    # 5. Populations des maisons non négatives et dans les bornes
    for origin, h in gs.houses.items():
        if h.population < 0:
            fail(f"house {origin}: population={h.population} < 0")
        if h.level < 0 or h.level > 6:
            fail(f"house {origin}: level={h.level} hors [0,6]")
        if h.level >= 1:
            max_pop = house_levels_map[h.level].max_population
            if h.population > max_pop:
                fail(f"house {origin} level={h.level}: pop={h.population} > max={max_pop}")

    # 6. TurnResult.total_population == somme réelle des maisons
    real_pop = sum(h.population for h in gs.houses.values())
    if result.total_population != real_pop:
        fail(f"result.total_population={result.total_population} != somme réelle={real_pop}")

    # 7. TurnResult.city_level == gs.city_level
    if result.city_level != gs.city_level:
        fail(f"result.city_level={result.city_level} != gs.city_level={gs.city_level}")

    # 8. TurnResult.global_satisfaction == gs.global_satisfaction
    if abs(result.global_satisfaction - gs.global_satisfaction) > 1e-9:
        fail(f"result.global_satisfaction={result.global_satisfaction:.4f} != gs={gs.global_satisfaction:.4f}")

    # 9. Stockage non dépassé (uniquement si bâtiment de stockage présent)
    for resource in ("wheat", "wood", "marble"):
        cap = compute_storage_cap(resource, gs.grid.placed_buildings, bldg)
        stock = getattr(rs, resource)
        if cap > 0 and stock > cap:
            fail(f"{resource}={stock} dépasse cap={cap}")

    # 10. Maisons orphelines (dans gs.houses mais pas dans la grille)
    for origin in gs.houses:
        if origin not in gs.grid.placed_buildings:
            fail(f"maison orpheline en {origin} (pas dans la grille)")

    # 11. Consistency done/victory/defeat
    if result.victory and not result.done:
        fail("victory=True mais done=False")
    if result.defeat and not result.done:
        fail("defeat=True mais done=False")

    return violations


def run_game(cfg, actions: list[Action], seed: int = 42) -> list[str]:
    """Joue une séquence d'actions et retourne toutes les violations d'invariants."""
    gs = init_game_state(cfg, seed=seed)
    all_violations = []
    for action in actions:
        result = step(gs, cfg, action)
        violations = check_invariants(gs, result, cfg)
        all_violations.extend(violations)
        if result.done:
            break
    return all_violations


# ---------------------------------------------------------------------------
# Tests scénarisés ciblés
# ---------------------------------------------------------------------------

def test_invariants_minimal_well_housing(cfg):
    """Well + housing, 80 tours sans ferme → famine inévitable."""
    actions = (
        [Action("place", "well", 4, 4), Action("place", "housing", 5, 4)]
        + [Action("do_nothing")] * 80
    )
    violations = run_game(cfg, actions, seed=42)
    assert violations == [], "\n".join(violations)


def test_invariants_full_early_economy(cfg):
    """Ferme + granary + well + market + 2 logements → cycle économique complet."""
    actions = [
        Action("place", "wheat_farm", 1, 1),
        Action("place", "granary", 5, 1),
        Action("place", "well", 9, 4),
        Action("place", "market", 10, 6),
        Action("place", "housing", 12, 4),
        Action("place", "housing", 15, 4),
    ] + [Action("do_nothing")] * 150
    violations = run_game(cfg, actions, seed=42)
    assert violations == [], "\n".join(violations)


def test_invariants_spam_illegal_placements(cfg):
    """Placements illégaux répétés (ressources insuffisantes) → état cohérent."""
    # Forum coûte 2000 denarii + 200 marbre : impossible au départ
    actions = [Action("place", "forum", 5, 5)] * 50
    violations = run_game(cfg, actions, seed=42)
    assert violations == [], "\n".join(violations)


def test_invariants_demolish_rebuild(cfg):
    """Démolir puis reconstruire le même bâtiment → pas d'état corrompu."""
    actions = (
        [
            Action("place", "well", 4, 4),
            Action("place", "housing", 6, 4),
            Action("do_nothing"),
            Action("demolish", None, 4, 4),   # détruire le puits
            Action("place", "well", 4, 4),    # reconstruire
        ]
        + [Action("do_nothing")] * 50
    )
    violations = run_game(cfg, actions, seed=42)
    assert violations == [], "\n".join(violations)


def test_fire_does_not_zero_resources_without_storage(cfg):
    """Régression : incendie sur bâtiment non-stockage ne doit pas zeriser blé/bois."""
    gs = init_game_state(cfg, seed=42)
    actions = (
        [Action("place", "well", 4, 4), Action("place", "housing", 5, 4)]
        + [Action("do_nothing")] * 80
    )
    for action in actions:
        result = step(gs, cfg, action)
        rs = gs.resource_state
        # Après un incendie sur non-storage, les ressources ne doivent pas chuter à 0
        if result.new_event and result.new_event.event_type == "fire":
            destroyed_bid = result.new_event.data.get("destroyed_building", "")
            storage_cfg = cfg.buildings.buildings[destroyed_bid].storage
            if storage_cfg is None:
                # Bâtiment sans stockage brûlé → ressources inchangées par le fire
                assert rs.wheat >= 0, f"wheat < 0 après fire sur {destroyed_bid}"
                assert rs.wood >= 0, f"wood < 0 après fire sur {destroyed_bid}"
                # La chute brutale à 0 était le bug : vérifier que le blé initial n'a pas disparu
                # s'il n'y a aucun granary (pas de raison que le feu touche au blé)
                granary_present = any(
                    pb.building_id == "granary"
                    for pb in gs.grid.placed_buildings.values()
                )
                if not granary_present and rs.wheat == 0 and gs.turn < 30:
                    pytest.fail(
                        f"Fire sur {destroyed_bid} (non-storage) a zéré le blé au tour {gs.turn}"
                    )
        if result.done:
            break


def test_demolish_non_storage_preserves_resources(cfg):
    """Démolir un puits ne doit pas zeriser blé/bois (même bug que fire, côté demolish)."""
    gs = init_game_state(cfg, seed=42)
    # Placer well + farm, accumuler blé, démolir le well
    for action in [
        Action("place", "wheat_farm", 1, 1),
        Action("place", "well", 8, 8),
    ]:
        step(gs, cfg, action)

    for _ in range(5):
        step(gs, cfg, Action("do_nothing"))

    wheat_before = gs.resource_state.wheat
    result = step(gs, cfg, Action("demolish", None, 8, 8))  # démolir le puits
    wheat_after = gs.resource_state.wheat

    assert result.action_succeeded, "La démolition du puits devrait réussir"
    assert wheat_after == wheat_before, (
        f"Démolir un puits a changé le blé : {wheat_before} → {wheat_after}"
    )


def test_demolish_granary_clamps_wheat(cfg):
    """Démolir le seul granary doit perdre le blé excédentaire (comportement voulu)."""
    gs = init_game_state(cfg, seed=42)
    for action in [
        Action("place", "wheat_farm", 1, 1),
        Action("place", "granary", 5, 1),
    ]:
        step(gs, cfg, action)

    # Accumuler du blé
    for _ in range(15):
        step(gs, cfg, Action("do_nothing"))

    wheat_before = gs.resource_state.wheat
    assert wheat_before > 0, "Il devrait y avoir du blé après 15 tours de production"

    # Démolir le granary → le blé doit être clampé à 0 (cap = 0 sans granary)
    step(gs, cfg, Action("demolish", None, 5, 1))
    wheat_after = gs.resource_state.wheat

    cap_after = compute_storage_cap("wheat", gs.grid.placed_buildings, cfg.buildings.buildings)
    assert cap_after == 0, "Après démolition du seul granary, cap doit être 0"
    assert wheat_after == 0, (
        f"Après démolition du granary, blé devrait être clampé à 0, got {wheat_after}"
    )


def test_action_succeeded_false_on_illegal_placement(cfg):
    """Placer un bâtiment sans ressources → action_succeeded=False."""
    gs = init_game_state(cfg, seed=42)
    # Forum coûte 2000 denarii + 200 marble, on a 800 denarii et 0 marble
    result = step(gs, cfg, Action("place", "forum", 5, 5))
    assert not result.action_succeeded, "Forum sans ressources doit échouer"
    # La grille ne doit pas contenir de forum
    bldg_ids = {pb.building_id for pb in gs.grid.placed_buildings.values()}
    assert "forum" not in bldg_ids


def test_action_succeeded_true_on_legal_placement(cfg):
    """Placer un bâtiment légal → action_succeeded=True et bâtiment présent."""
    gs = init_game_state(cfg, seed=42)
    result = step(gs, cfg, Action("place", "road", 5, 5))
    assert result.action_succeeded, "Placement d'une route doit réussir"
    bldg_ids = {pb.building_id for pb in gs.grid.placed_buildings.values()}
    assert "road" in bldg_ids


def test_do_nothing_always_legal(cfg, building_list):
    """DO_NOTHING doit toujours être dans le masque, quel que soit l'état."""
    gs = init_game_state(cfg, seed=42)
    for action in [
        Action("place", "well", 4, 4),
        Action("place", "housing", 6, 4),
    ] + [Action("do_nothing")] * 30:
        step(gs, cfg, action)
        mask = compute_action_mask(gs, cfg, building_list)
        assert mask[DO_NOTHING], f"DO_NOTHING illégal au tour {gs.turn}"
        if gs.done:
            break


def test_unique_building_cannot_be_placed_twice(cfg):
    """Forum (unique) ne peut être placé qu'une seule fois."""
    # On triche sur les ressources pour pouvoir placer le forum
    gs = init_game_state(cfg, seed=42)
    gs.resource_state.denarii = 10_000.0
    gs.resource_state.marble = 1_000

    result1 = step(gs, cfg, Action("place", "forum", 5, 5))
    assert result1.action_succeeded, "Premier placement forum doit réussir"

    result2 = step(gs, cfg, Action("place", "forum", 15, 15))
    assert not result2.action_succeeded, "Deuxième placement forum doit échouer (unique)"

    forums = [pb for pb in gs.grid.placed_buildings.values() if pb.building_id == "forum"]
    assert len(forums) == 1, f"Un seul forum doit exister, got {len(forums)}"


def test_place_on_water_terrain_fails(cfg):
    """Placer un bâtiment sur eau doit échouer (terrain invalide)."""
    gs = init_game_state(cfg, seed=42)
    # Tile water confirmée sur seed 42 : (13, 19)
    result = step(gs, cfg, Action("place", "road", 13, 19))
    assert not result.action_succeeded, "Placement sur eau doit échouer"


def test_place_on_occupied_tile_fails(cfg):
    """Placer sur une case déjà occupée doit échouer."""
    gs = init_game_state(cfg, seed=42)
    step(gs, cfg, Action("place", "well", 4, 4))
    result = step(gs, cfg, Action("place", "road", 4, 4))
    assert not result.action_succeeded, "Placement sur case occupée doit échouer"


def test_immigration_lost_when_houses_full(cfg):
    """Avec une seule maison pleine (5/5) et aucune autre, la pop ne dépasse jamais 5."""
    gs = init_game_state(cfg, seed=42)
    step(gs, cfg, Action("place", "well", 4, 4))
    step(gs, cfg, Action("place", "housing", 5, 4))

    # Forcer la maison à pleine capacité
    origin = (5, 4)
    if origin in gs.houses:
        gs.houses[origin].level = 1
        gs.houses[origin].population = 5

    for _ in range(50):
        result = step(gs, cfg, Action("do_nothing"))
        total = sum(h.population for h in gs.houses.values())
        assert total <= 5, f"Pop={total} dépasse le max de la seule maison (5)"
        if result.done:
            break


def test_wheat_farm_on_plain_only(cfg):
    """wheat_farm sur tuile hill → doit échouer (contrainte terrain)."""
    gs = init_game_state(cfg, seed=42)
    # Hill tile connue seed 42 : (4, 0)
    result = step(gs, cfg, Action("place", "wheat_farm", 4, 0))
    # wheat_farm requiert all_tiles=plain, (4,0) est hill → échec
    assert not result.action_succeeded, "wheat_farm sur hill doit échouer"


def test_marble_quarry_on_hill_only(cfg):
    """marble_quarry sur plain → doit échouer."""
    gs = init_game_state(cfg, seed=42)
    # Plain tile connue seed 42 : (10, 10)
    result = step(gs, cfg, Action("place", "marble_quarry", 10, 10))
    assert not result.action_succeeded, "marble_quarry sur plain doit échouer"


def test_bankrupt_defeat_after_5_turns(cfg):
    """5 tours consécutifs sous -500 denarii → défaite."""
    gs = init_game_state(cfg, seed=42)
    # Forcer une situation de banqueroute permanente
    gs.resource_state.denarii = -600.0

    done = False
    for turn in range(20):
        result = step(gs, cfg, Action("do_nothing"))
        if result.defeat:
            done = True
            break

    assert done, "Défaite par banqueroute non déclenchée après 5 tours sous -500"


def test_famine_reduces_population(cfg):
    """Sans blé ni ferme, la famine doit déclencher des pertes de pop (famine_pop_lost > 0).

    Note : avec une maison level 1 (seul besoin = water), la croissance naturelle
    compense immédiatement la perte si la satisfaction reste haute. On vérifie donc
    famine_pop_lost plutôt que la pop finale (qui peut rester stable si growth compense).
    """
    gs = init_game_state(cfg, seed=42)

    step(gs, cfg, Action("place", "well", 4, 4))
    step(gs, cfg, Action("place", "housing", 5, 4))

    # Forcer une maison level 1, pop=5, et blé=0 après placement
    origin = (5, 4)
    if origin in gs.houses:
        gs.houses[origin].level = 1
        gs.houses[origin].population = 5
    gs.resource_state.wheat = 0

    total_famine_lost = 0
    for _ in range(15):
        result = step(gs, cfg, Action("do_nothing"))
        total_famine_lost += result.famine_pop_lost
        if result.done:
            break

    assert total_famine_lost > 0, (
        "La famine n'a produit aucune perte de pop sur 15 tours malgré blé=0"
    )


def test_rng_determinism(cfg, building_list):
    """Même seed → séquences d'événements identiques."""
    def run(seed):
        gs = init_game_state(cfg, seed=seed)
        events = []
        for _ in range(50):
            result = step(gs, cfg, Action("do_nothing"))
            if result.new_event:
                events.append((gs.turn, result.new_event.event_type))
            if result.done:
                break
        return events

    assert run(42) == run(42), "Même seed → mêmes événements"
    assert run(7) == run(7), "Même seed → mêmes événements"


def test_turnresult_consistency_over_full_game(cfg):
    """TurnResult doit être cohérent avec GameState à chaque tour sur 100 tours."""
    actions = (
        [
            Action("place", "wheat_farm", 1, 1),
            Action("place", "granary", 5, 1),
            Action("place", "well", 9, 4),
            Action("place", "market", 11, 6),
            Action("place", "housing", 13, 4),
        ]
        + [Action("do_nothing")] * 100
    )
    violations = run_game(cfg, actions, seed=42)
    assert violations == [], "\n".join(violations)


# ---------------------------------------------------------------------------
# Tests de fuzzing : actions légales aléatoires sur beaucoup de seeds
# ---------------------------------------------------------------------------

FUZZ_SEEDS = list(range(30))          # 30 seeds différentes
FUZZ_MAX_TURNS = 300                  # 300 tours par partie


@pytest.mark.parametrize("seed", FUZZ_SEEDS)
def test_fuzz_random_legal_actions(cfg, building_list, seed):
    """Joue 300 tours d'actions légales aléatoires et vérifie tous les invariants."""
    rng = np.random.default_rng(seed=seed + 1000)  # RNG indépendant du jeu
    gs = init_game_state(cfg, seed=seed)
    violations = []

    for _ in range(FUZZ_MAX_TURNS):
        mask = compute_action_mask(gs, cfg, building_list)
        legal = np.where(mask)[0]
        action_int = int(rng.choice(legal))

        # Décoder l'action
        if action_int == DO_NOTHING:
            action = Action("do_nothing")
        elif action_int >= len(building_list) * 1024:
            # Demolish range
            idx = action_int - len(building_list) * 1024
            x, y = idx % 32, idx // 32
            action = Action("demolish", None, x, y)
        else:
            bldg_idx = action_int // 1024
            cell = action_int % 1024
            x, y = cell % 32, cell // 32
            action = Action("place", building_list[bldg_idx], x, y)

        result = step(gs, cfg, action)
        v = check_invariants(gs, result, cfg, label=f"seed={seed}")
        violations.extend(v)

        if result.done:
            break

    assert violations == [], "\n".join(violations[:20])  # limiter l'output si nombreux


# ---------------------------------------------------------------------------
# Tests de fuzzing : actions DO_NOTHING pures sur seeds variées
# ---------------------------------------------------------------------------

IDLE_SEEDS = list(range(20, 50))  # seeds 20-49


@pytest.mark.parametrize("seed", IDLE_SEEDS)
def test_fuzz_idle_game(cfg, seed):
    """DO_NOTHING pendant 200 tours sur 30 seeds : invariants toujours tenus."""
    gs = init_game_state(cfg, seed=seed)
    violations = []
    for _ in range(200):
        result = step(gs, cfg, Action("do_nothing"))
        v = check_invariants(gs, result, cfg, label=f"idle seed={seed}")
        violations.extend(v)
        if result.done:
            break
    assert violations == [], "\n".join(violations)


# ---------------------------------------------------------------------------
# Test de progression jusqu'au niveau 2
# ---------------------------------------------------------------------------

def test_compute_city_level_2_conditions(cfg):
    """compute_city_level retourne 2 quand pop >= 200, sat >= 40%, forum posé."""
    from collections import Counter
    from vitruvius.engine.victory import compute_city_level

    city_levels = cfg.city_levels.city_levels
    placed_forum = Counter({"forum": 1})
    placed_empty = Counter()

    assert compute_city_level(200, 0.40, placed_forum, city_levels) == 2
    assert compute_city_level(199, 0.40, placed_forum, city_levels) == 1   # pop insuffisante
    assert compute_city_level(200, 0.39, placed_forum, city_levels) == 1   # sat insuffisante
    assert compute_city_level(200, 0.40, placed_empty, city_levels) == 1   # pas de forum
    assert compute_city_level(500, 1.00, placed_forum, city_levels) == 2   # niveau 3 pas atteint (pas de temple)


def test_house_reaches_level_2_with_food_and_water(cfg):
    """Une maison avec eau ET nourriture doit évoluer de level 1 à level 2."""
    gs = init_game_state(cfg, seed=42)
    # Infrastructure minimale pour level 2 : well (eau) + market (nourriture) + ferme
    # Tout placer en zone plain bien dégagée (seed 42, zone centrale)
    gs.resource_state.denarii = 10_000.0

    for action in [
        Action("place", "wheat_farm", 1, 1),
        Action("place", "granary", 5, 1),
        Action("place", "well", 9, 5),
        Action("place", "market", 9, 8),
        Action("place", "housing", 11, 5),
    ]:
        result = step(gs, cfg, action)

    # Forcer un habitant dans la maison (level=1) pour éviter d'attendre l'immigration
    origin = (11, 5)
    if origin in gs.houses:
        gs.houses[origin].level = 1
        gs.houses[origin].population = 3

    evolved = False
    for _ in range(30):
        result = step(gs, cfg, Action("do_nothing"))
        if result.evolved > 0:
            evolved = True
        h = gs.houses.get(origin)
        if h and h.level >= 2:
            break
        if result.done:
            break

    h = gs.houses.get(origin)
    level = h.level if h else 0
    assert level >= 2 or evolved, (
        f"La maison n'a pas évolué au niveau 2 après 30 tours "
        f"(niveau actuel: {level}). Vérifier coverage water/food."
    )
