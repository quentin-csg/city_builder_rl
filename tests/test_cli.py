"""Tests pour vitruvius.cli (parser + fonctions de rendu)."""

from __future__ import annotations

import pytest

from vitruvius.cli import (
    format_building_info,
    format_buildings_list,
    format_inspect,
    format_turn_result,
    parse_command,
    render_grid,
    render_state,
)
from vitruvius.config import load_config
from vitruvius.engine.events import ActiveEvent
from vitruvius.engine.game_state import init_game_state
from vitruvius.engine.turn import Action, TurnResult, step


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def building_list(cfg):
    return list(cfg.buildings.buildings.keys())


@pytest.fixture
def gs(cfg):
    return init_game_state(cfg, seed=42)


@pytest.fixture
def dummy_result() -> TurnResult:
    return TurnResult(
        production={"wheat": 12, "wood": 5},
        taxes_collected=80.0,
        maintenance_paid=20.0,
        passive_income=10.0,
        famine_count=0,
        famine_pop_lost=0,
        evolved=1,
        regressed=0,
        growth=6,
        exodus=0,
        new_event=None,
        global_satisfaction=0.65,
        total_population=120,
        city_level=1,
        done=False,
        victory=False,
        defeat=False,
        bankrupt=False,
    )


# ---------------------------------------------------------------------------
# parse_command — actions de jeu
# ---------------------------------------------------------------------------


def test_parse_place(building_list):
    result = parse_command("place road 5 7", building_list)
    assert result == Action("place", "road", 4, 6)


def test_parse_place_alias(building_list):
    result = parse_command("p housing 1 1", building_list)
    assert result == Action("place", "housing", 0, 0)


def test_parse_demolish(building_list):
    result = parse_command("demolish 3 4", building_list)
    assert result == Action("demolish", x=2, y=3)


def test_parse_demolish_alias(building_list):
    result = parse_command("d 10 20", building_list)
    assert result == Action("demolish", x=9, y=19)


def test_parse_wait(building_list):
    assert parse_command("wait", building_list) == Action("do_nothing")


def test_parse_wait_alias(building_list):
    assert parse_command("w", building_list) == Action("do_nothing")


def test_parse_empty_line(building_list):
    assert parse_command("", building_list) == Action("do_nothing")


# ---------------------------------------------------------------------------
# parse_command — commandes spéciales
# ---------------------------------------------------------------------------


def test_parse_quit(building_list):
    assert parse_command("quit", building_list) == "quit"
    assert parse_command("q", building_list) == "quit"


def test_parse_help(building_list):
    assert parse_command("help", building_list) == "help"
    assert parse_command("h", building_list) == "help"
    assert parse_command("?", building_list) == "help"


def test_parse_list(building_list):
    assert parse_command("list", building_list) == "list"
    assert parse_command("ls", building_list) == "list"


def test_parse_info(building_list):
    assert parse_command("info road", building_list) == "info:road"
    assert parse_command("i housing", building_list) == "info:housing"


def test_parse_save(building_list):
    assert parse_command("save game.json", building_list) == "save:game.json"


def test_parse_load(building_list):
    assert parse_command("load game.json", building_list) == "load:game.json"


# ---------------------------------------------------------------------------
# parse_command — erreurs
# ---------------------------------------------------------------------------


def test_parse_invalid_building(building_list):
    with pytest.raises(ValueError, match="inconnu"):
        parse_command("place nonexistent 0 0", building_list)


def test_parse_out_of_bounds(building_list):
    with pytest.raises(ValueError, match="hors grille"):
        parse_command("place road 50 50", building_list)


def test_parse_place_missing_args(building_list):
    with pytest.raises(ValueError):
        parse_command("place road 5", building_list)


def test_parse_demolish_missing_args(building_list):
    with pytest.raises(ValueError):
        parse_command("demolish 5", building_list)


def test_parse_unknown_command(building_list):
    with pytest.raises(ValueError, match="inconnue"):
        parse_command("fly 0 0", building_list)


def test_parse_non_int_coords(building_list):
    with pytest.raises(ValueError):
        parse_command("place road abc def", building_list)


# ---------------------------------------------------------------------------
# render_grid
# ---------------------------------------------------------------------------


def test_render_grid_dimensions(gs):
    output = render_grid(gs.grid)
    data_lines = [l for l in output.split("\n") if l and l[0].isdigit() or (len(l) > 1 and l[1].isdigit())]
    # On compte les lignes numérotées (y de 0 à 31)
    numbered = [l for l in output.split("\n") if l[:2].strip().isdigit()]
    assert len(numbered) == 32


def test_render_grid_contains_terrain_chars(gs):
    output = render_grid(gs.grid)
    # Au moins un tile plaine '.' doit être présent (seed=42 a de la plaine)
    assert "." in output


def test_render_grid_32_cols(gs):
    output = render_grid(gs.grid)
    numbered_lines = [l for l in output.split("\n") if l[:2].strip().isdigit()]
    for line in numbered_lines:
        # Format : "YY XXXXXXXX...32 chars"
        content = line[3:]  # skip "YY "
        assert len(content) == 32, f"Ligne incorrecte: {line!r}"


# ---------------------------------------------------------------------------
# render_state
# ---------------------------------------------------------------------------


def test_render_state_contains_fields(gs):
    output = render_state(gs)
    assert "Tour" in output
    assert "Niveau" in output
    assert "Pop" in output
    assert "Denarii" in output
    assert "Maisons" in output


def test_render_state_satisfaction_percent(gs):
    gs.global_satisfaction = 0.75
    output = render_state(gs)
    assert "75%" in output


# ---------------------------------------------------------------------------
# format_turn_result
# ---------------------------------------------------------------------------


def test_format_turn_result_production(dummy_result):
    output = format_turn_result(dummy_result)
    assert "wheat" in output or "bois" in output or "Production" in output


def test_format_turn_result_taxes(dummy_result):
    output = format_turn_result(dummy_result)
    assert "Taxes" in output
    assert "80" in output


def test_format_turn_result_growth(dummy_result):
    output = format_turn_result(dummy_result)
    assert "Croissance" in output
    assert "6" in output


def test_format_turn_result_famine():
    r = TurnResult(
        production={}, taxes_collected=0.0, maintenance_paid=0.0,
        passive_income=0.0, famine_count=3, famine_pop_lost=8,
        evolved=0, regressed=0, growth=0, exodus=0, new_event=None,
        global_satisfaction=0.1, total_population=80, city_level=1,
        done=False, victory=False, defeat=False, bankrupt=False,
    )
    output = format_turn_result(r)
    assert "Famine" in output
    assert "3" in output


def test_format_turn_result_bankrupt():
    r = TurnResult(
        production={}, taxes_collected=0.0, maintenance_paid=0.0,
        passive_income=0.0, famine_count=0, famine_pop_lost=0,
        evolved=0, regressed=0, growth=0, exodus=0, new_event=None,
        global_satisfaction=0.5, total_population=100, city_level=1,
        done=False, victory=False, defeat=False, bankrupt=True,
    )
    output = format_turn_result(r)
    assert "Banqueroute" in output


def test_format_turn_result_event():
    r = TurnResult(
        production={}, taxes_collected=0.0, maintenance_paid=0.0,
        passive_income=0.0, famine_count=0, famine_pop_lost=0,
        evolved=0, regressed=0, growth=0, exodus=0,
        new_event=ActiveEvent("drought", turns_remaining=3, data={}),
        global_satisfaction=0.5, total_population=100, city_level=1,
        done=False, victory=False, defeat=False, bankrupt=False,
    )
    output = format_turn_result(r)
    assert "drought" in output


# ---------------------------------------------------------------------------
# format_buildings_list / format_building_info
# ---------------------------------------------------------------------------


def test_format_buildings_list_count(cfg):
    output = format_buildings_list(cfg)
    # 20 batiments + header + separator = 22 lignes minimum
    lines = [l for l in output.split("\n") if l.strip()]
    building_lines = [l for l in lines if not l.startswith("-") and "ID" not in l]
    assert len(building_lines) == 20


def test_format_building_info_road(cfg):
    output = format_building_info("road", cfg)
    assert "road" in output
    assert "Taille" in output
    assert "Coût" in output


def test_format_building_info_unknown(cfg):
    with pytest.raises(KeyError):
        format_building_info("nonexistent", cfg)


# ---------------------------------------------------------------------------
# parse_command — inspect
# ---------------------------------------------------------------------------


def test_parse_inspect(building_list):
    assert parse_command("inspect 5 7", building_list) == "inspect:4,6"
    assert parse_command("x 1 1", building_list) == "inspect:0,0"


def test_parse_inspect_out_of_bounds(building_list):
    with pytest.raises(ValueError, match="hors grille"):
        parse_command("inspect 50 0", building_list)


def test_parse_inspect_missing_args(building_list):
    with pytest.raises(ValueError):
        parse_command("inspect 5", building_list)


# ---------------------------------------------------------------------------
# format_inspect
# ---------------------------------------------------------------------------


def test_format_inspect_empty_tile(cfg, gs):
    """Case vide : affiche terrain, pas de bâtiment."""
    from vitruvius.engine.terrain import TerrainType
    # Trouver une case PLAIN libre
    x, y = None, None
    for iy in range(gs.grid.SIZE):
        for ix in range(gs.grid.SIZE):
            if gs.grid.terrain[iy][ix] == TerrainType.PLAIN and gs.grid._origin[iy][ix] is None:
                x, y = ix, iy
                break
        if x is not None:
            break
    assert x is not None
    output = format_inspect(x, y, gs, cfg)
    assert "plain" in output
    assert "aucun" in output


def test_format_inspect_building(cfg, gs):
    """Case avec bâtiment : affiche le building_id."""
    from vitruvius.engine.terrain import TerrainType
    from vitruvius.engine.turn import step
    x, y = None, None
    for iy in range(gs.grid.SIZE):
        for ix in range(gs.grid.SIZE):
            if gs.grid.terrain[iy][ix] == TerrainType.PLAIN and gs.grid._origin[iy][ix] is None:
                x, y = ix, iy
                break
        if x is not None:
            break
    step(gs, cfg, Action("place", "road", x, y))
    output = format_inspect(x, y, gs, cfg)
    assert "road" in output


def test_format_inspect_house_coverage(cfg):
    """Maison inspectée : affiche les 6 besoins avec statut couverture."""
    gs = init_game_state(cfg, seed=42)
    from vitruvius.engine.terrain import TerrainType
    # Trouver un bloc 2x2 libre
    pos = None
    for iy in range(gs.grid.SIZE - 1):
        for ix in range(gs.grid.SIZE - 1):
            if all(
                gs.grid.terrain[iy + dy][ix + dx] == TerrainType.PLAIN
                and gs.grid._origin[iy + dy][ix + dx] is None
                for dy in range(2) for dx in range(2)
            ):
                pos = (ix, iy)
                break
        if pos is not None:
            break
    assert pos is not None
    x, y = pos
    step(gs, cfg, Action("place", "housing", x, y))
    output = format_inspect(x, y, gs, cfg)
    assert "water" in output
    assert "food" in output
    assert "Niveau" in output
    assert "Population" in output
    assert "Famine" in output
