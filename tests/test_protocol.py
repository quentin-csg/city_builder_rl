"""Tests du module protocol.py — purs, synchrones, pas d'asyncio."""

from __future__ import annotations

import json

import pytest

from vitruvius.bridge.protocol import (
    AUTO_STEP_MAX,
    ActionMsg,
    AutoStepMsg,
    LoadModelMsg,
    ProtocolError,
    ResetMsg,
    build_ack_message,
    build_error_message,
    build_init_message,
    build_state_message,
    buildings_catalog_to_json,
    parse_client_message,
    terrain_to_json,
)
from vitruvius.config import load_config
from vitruvius.engine.game_state import init_game_state
from vitruvius.engine.turn import Action


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def config():
    return load_config()


@pytest.fixture(scope="module")
def gs(config):
    return init_game_state(config, seed=0)


# ---------------------------------------------------------------------------
# terrain_to_json
# ---------------------------------------------------------------------------


def test_terrain_to_json_uses_enum_values(gs):
    """Le terrain est sérialisé comme strings TerrainType.value."""
    terrain = terrain_to_json(gs.grid)
    assert len(terrain) == 32
    assert all(len(row) == 32 for row in terrain)
    valid_values = {"plain", "forest", "hill", "water", "marsh"}
    for row in terrain:
        for cell in row:
            assert cell in valid_values


# ---------------------------------------------------------------------------
# buildings_catalog_to_json
# ---------------------------------------------------------------------------


def test_buildings_catalog_contains_all_buildings(config):
    catalog = buildings_catalog_to_json(config)
    expected_ids = set(config.buildings.buildings.keys())
    assert set(catalog.keys()) == expected_ids


def test_buildings_catalog_has_required_fields(config):
    catalog = buildings_catalog_to_json(config)
    for bid, entry in catalog.items():
        assert "display_name" in entry, f"Manque display_name pour {bid}"
        assert "size" in entry, f"Manque size pour {bid}"
        assert "cost" in entry, f"Manque cost pour {bid}"
        assert "unique" in entry, f"Manque unique pour {bid}"
        assert isinstance(entry["size"], list)
        assert len(entry["size"]) == 2


# ---------------------------------------------------------------------------
# build_init_message
# ---------------------------------------------------------------------------


def test_build_init_includes_terrain_and_catalog(gs, config):
    msg = build_init_message(gs, config)
    assert msg["type"] == "init"
    assert msg["seed"] == gs.seed
    assert msg["size"] == 32
    assert len(msg["terrain"]) == 32
    assert len(msg["terrain"][0]) == 32
    assert isinstance(msg["buildings_catalog"], dict)
    assert len(msg["buildings_catalog"]) == len(config.buildings.buildings)
    assert "state" in msg
    assert isinstance(msg["model_loaded"], bool)


def test_build_init_json_serializable(gs, config):
    """Le message init doit être entièrement sérialisable en JSON."""
    msg = build_init_message(gs, config)
    dumped = json.dumps(msg)
    reloaded = json.loads(dumped)
    assert reloaded["type"] == "init"


# ---------------------------------------------------------------------------
# build_state_message
# ---------------------------------------------------------------------------


def test_build_state_message_structure(gs, config):
    """build_state retourne type=state avec turn_result, done, victory."""
    from vitruvius.engine.turn import step

    import copy
    gs_copy = copy.deepcopy(gs)
    result = step(gs_copy, config, Action("do_nothing"))
    msg = build_state_message(gs_copy, result)
    assert msg["type"] == "state"
    assert "state" in msg
    assert "turn_result" in msg
    assert "done" in msg
    assert "victory" in msg
    tr = msg["turn_result"]
    assert "taxes_collected" in tr
    assert "action_succeeded" in tr


def test_build_state_json_serializable(gs, config):
    from vitruvius.engine.turn import step
    import copy

    gs_copy = copy.deepcopy(gs)
    result = step(gs_copy, config, Action("do_nothing"))
    msg = build_state_message(gs_copy, result)
    dumped = json.dumps(msg)
    assert json.loads(dumped)["type"] == "state"


# ---------------------------------------------------------------------------
# build_error_message / build_ack_message
# ---------------------------------------------------------------------------


def test_build_error_message():
    msg = build_error_message("test erreur")
    assert msg == {"type": "error", "message": "test erreur"}


def test_build_ack_message_basic():
    msg = build_ack_message("load_model")
    assert msg["type"] == "ack"
    assert msg["action"] == "load_model"
    assert msg["ok"] is True


def test_build_ack_message_with_payload():
    msg = build_ack_message("load_model", {"path": "models/test.zip"})
    assert msg["path"] == "models/test.zip"


# ---------------------------------------------------------------------------
# parse_client_message — do_nothing
# ---------------------------------------------------------------------------


def test_parse_action_do_nothing(config):
    raw = json.dumps({"type": "action", "action": {"type": "do_nothing"}})
    msg = parse_client_message(raw, config)
    assert isinstance(msg, ActionMsg)
    assert msg.action == Action("do_nothing")


# ---------------------------------------------------------------------------
# parse_client_message — place
# ---------------------------------------------------------------------------


def test_parse_action_place_valid(config):
    raw = json.dumps({"type": "action", "action": {"type": "place", "building_id": "housing", "x": 5, "y": 10}})
    msg = parse_client_message(raw, config)
    assert isinstance(msg, ActionMsg)
    assert msg.action == Action("place", building_id="housing", x=5, y=10)


def test_parse_action_invalid_coordinates_negative(config):
    raw = json.dumps({"type": "action", "action": {"type": "place", "building_id": "housing", "x": -1, "y": 5}})
    with pytest.raises(ProtocolError, match="hors grille"):
        parse_client_message(raw, config)


def test_parse_action_invalid_coordinates_too_large(config):
    raw = json.dumps({"type": "action", "action": {"type": "place", "building_id": "housing", "x": 32, "y": 5}})
    with pytest.raises(ProtocolError, match="hors grille"):
        parse_client_message(raw, config)


def test_parse_action_unknown_building(config):
    raw = json.dumps({"type": "action", "action": {"type": "place", "building_id": "forteresse", "x": 5, "y": 5}})
    with pytest.raises(ProtocolError, match="inconnu"):
        parse_client_message(raw, config)


def test_parse_action_place_missing_building_id(config):
    raw = json.dumps({"type": "action", "action": {"type": "place", "x": 5, "y": 5}})
    with pytest.raises(ProtocolError):
        parse_client_message(raw, config)


# ---------------------------------------------------------------------------
# parse_client_message — demolish
# ---------------------------------------------------------------------------


def test_parse_action_demolish_valid(config):
    raw = json.dumps({"type": "action", "action": {"type": "demolish", "x": 10, "y": 20}})
    msg = parse_client_message(raw, config)
    assert isinstance(msg, ActionMsg)
    assert msg.action == Action("demolish", x=10, y=20)


# ---------------------------------------------------------------------------
# parse_client_message — reset
# ---------------------------------------------------------------------------


def test_parse_reset_with_seed(config):
    raw = json.dumps({"type": "reset", "seed": 99})
    msg = parse_client_message(raw, config)
    assert isinstance(msg, ResetMsg)
    assert msg.seed == 99


def test_parse_reset_without_seed(config):
    raw = json.dumps({"type": "reset"})
    msg = parse_client_message(raw, config)
    assert isinstance(msg, ResetMsg)
    assert msg.seed is None


def test_parse_reset_invalid_seed_type(config):
    raw = json.dumps({"type": "reset", "seed": "quarante-deux"})
    with pytest.raises(ProtocolError):
        parse_client_message(raw, config)


# ---------------------------------------------------------------------------
# parse_client_message — load_model
# ---------------------------------------------------------------------------


def test_parse_load_model_valid(config):
    raw = json.dumps({"type": "load_model", "path": "models/test.zip"})
    msg = parse_client_message(raw, config)
    assert isinstance(msg, LoadModelMsg)
    assert msg.path == "models/test.zip"


def test_parse_load_model_missing_path(config):
    raw = json.dumps({"type": "load_model"})
    with pytest.raises(ProtocolError, match="path"):
        parse_client_message(raw, config)


def test_parse_load_model_empty_path(config):
    raw = json.dumps({"type": "load_model", "path": ""})
    with pytest.raises(ProtocolError, match="vide"):
        parse_client_message(raw, config)


# ---------------------------------------------------------------------------
# parse_client_message — auto_step
# ---------------------------------------------------------------------------


def test_parse_auto_step_default(config):
    raw = json.dumps({"type": "auto_step"})
    msg = parse_client_message(raw, config)
    assert isinstance(msg, AutoStepMsg)
    assert msg.n == 1


def test_parse_auto_step_custom_n(config):
    raw = json.dumps({"type": "auto_step", "n": 50})
    msg = parse_client_message(raw, config)
    assert msg.n == 50


def test_parse_auto_step_bounds(config):
    raw = json.dumps({"type": "auto_step", "n": AUTO_STEP_MAX + 1})
    with pytest.raises(ProtocolError, match="maximum"):
        parse_client_message(raw, config)


def test_parse_auto_step_zero_rejected(config):
    raw = json.dumps({"type": "auto_step", "n": 0})
    with pytest.raises(ProtocolError):
        parse_client_message(raw, config)


# ---------------------------------------------------------------------------
# parse_client_message — erreurs génériques
# ---------------------------------------------------------------------------


def test_parse_unknown_type(config):
    raw = json.dumps({"type": "teleport"})
    with pytest.raises(ProtocolError, match="inconnu"):
        parse_client_message(raw, config)


def test_parse_malformed_json(config):
    with pytest.raises(ProtocolError, match="JSON"):
        parse_client_message("ce n'est pas du json {{{", config)


def test_parse_missing_type(config):
    raw = json.dumps({"action": {"type": "do_nothing"}})
    with pytest.raises(ProtocolError, match="type"):
        parse_client_message(raw, config)


def test_parse_not_an_object(config):
    raw = json.dumps([1, 2, 3])
    with pytest.raises(ProtocolError, match="objet"):
        parse_client_message(raw, config)
