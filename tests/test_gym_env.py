"""Tests pour vitruvius/rl/gym_env.py."""

from __future__ import annotations

import numpy as np
import pytest

from vitruvius.config import load_config
from vitruvius.engine.actions import DO_NOTHING, TOTAL_ACTIONS
from vitruvius.rl.gym_env import VitruviusEnv


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture
def env(cfg):
    e = VitruviusEnv(config=cfg, seed=42)
    e.reset()
    return e


# ---------------------------------------------------------------------------
# Espaces
# ---------------------------------------------------------------------------


def test_env_spaces(cfg):
    e = VitruviusEnv(config=cfg, seed=42)
    from gymnasium import spaces
    assert isinstance(e.observation_space, spaces.Dict)
    assert e.observation_space["grid"].shape == (32, 32, 31)
    assert e.observation_space["global_features"].shape == (18,)
    assert e.action_space.n == TOTAL_ACTIONS


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------


def test_env_reset_returns_obs_info(cfg):
    e = VitruviusEnv(config=cfg, seed=42)
    obs, info = e.reset()
    assert set(obs.keys()) == {"grid", "global_features"}
    assert obs["grid"].shape == (32, 32, 31)
    assert obs["global_features"].shape == (18,)
    assert isinstance(info, dict)


def test_env_reset_seed_determinism(cfg):
    e1 = VitruviusEnv(config=cfg)
    e2 = VitruviusEnv(config=cfg)
    obs1, _ = e1.reset(seed=42)
    obs2, _ = e2.reset(seed=42)
    np.testing.assert_array_equal(obs1["grid"], obs2["grid"])
    np.testing.assert_array_equal(obs1["global_features"], obs2["global_features"])


# ---------------------------------------------------------------------------
# Step
# ---------------------------------------------------------------------------


def test_env_step_returns_5_tuple(env):
    result = env.step(DO_NOTHING)
    assert len(result) == 5
    obs, reward, terminated, truncated, info = result
    assert obs["grid"].shape == (32, 32, 31)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


def test_env_step_do_nothing_advances_turn(env):
    turn_before = env.gs.turn
    env.step(DO_NOTHING)
    assert env.gs.turn == turn_before + 1


def test_env_step_reward_survival(env):
    """DO_NOTHING sur grille vide → pas de delta, mais W_POSITIVE_INCOME + W_SURVIVAL."""
    _, reward, terminated, truncated, _ = env.step(DO_NOTHING)
    if not terminated:
        # passif 25 > maintenance 0 → W_POSITIVE_INCOME=0.05 + W_SURVIVAL=0.01 = 0.06
        assert reward == pytest.approx(0.06, abs=1e-6)


def test_env_reward_determinism(cfg):
    """Même seed + même séquence d'actions → même séquence de rewards."""
    def run(seed: int) -> list[float]:
        e = VitruviusEnv(config=cfg, seed=seed, max_turns=10)
        e.reset(seed=seed)
        rewards = []
        for _ in range(5):
            _, r, term, trunc, _ = e.step(DO_NOTHING)
            rewards.append(r)
            if term or trunc:
                break
        return rewards

    r1 = run(42)
    r2 = run(42)
    assert r1 == pytest.approx(r2, abs=1e-6)


# ---------------------------------------------------------------------------
# Masque d'actions
# ---------------------------------------------------------------------------


def test_env_action_masks_shape(env):
    mask = env.action_masks()
    assert mask.shape == (TOTAL_ACTIONS,)
    assert mask.dtype == np.bool_


def test_env_action_masks_do_nothing_true(env):
    mask = env.action_masks()
    assert mask[DO_NOTHING] is np.bool_(True)


def test_env_info_mask_is_copy(env):
    """info['action_mask'] ne doit pas partager la mémoire avec le masque interne."""
    _, _, _, _, info = env.step(DO_NOTHING)
    mask_copy = info["action_mask"]
    mask_copy[DO_NOTHING] = False
    assert env.action_masks()[DO_NOTHING] is np.bool_(True)


# ---------------------------------------------------------------------------
# Truncation
# ---------------------------------------------------------------------------


def test_env_truncated_after_max_turns(cfg):
    e = VitruviusEnv(config=cfg, seed=42, max_turns=3)
    e.reset()
    truncated = False
    for _ in range(5):
        _, _, terminated, truncated, _ = e.step(DO_NOTHING)
        if truncated or terminated:
            break
    assert truncated is True


# ---------------------------------------------------------------------------
# Dynamics inter-tour
# ---------------------------------------------------------------------------


def test_dynamics_net_income_excludes_passive_income(cfg):
    """net_income = (taxes - maint) / 1000 — passive_income exclu.

    Bug précédent (server.py) : (taxes + passive - maint) / 1000 → delta constant +0.01/+0.02.
    Sur grille vide : taxes=0, maintenance=0 → net_income doit être exactement 0.0.
    """
    e = VitruviusEnv(config=cfg, seed=42)
    e.reset()
    e.step(DO_NOTHING)
    assert e._last_dynamics["net_income"] == pytest.approx(0.0), (
        f"net_income={e._last_dynamics['net_income']} — passive_income probablement inclus"
    )


def test_dynamics_growth_rate_uses_stored_prev_pop(cfg):
    """growth_rate = (new_pop - _prev_pop) / max(1, _prev_pop), clampé [-1, 1].

    Bug précédent (server.py) : result.growth / total_pop ignorait _prev_pop.
    Ici on force _prev_pop=200 avec une grille sans maisons (new_pop=0) :
    growth_rate attendu = (0 - 200) / max(1, 200) = -1.0 après clamp.
    """
    e = VitruviusEnv(config=cfg, seed=42)
    e.reset()
    e._prev_pop = 200  # pas de maisons → new_pop reste 0 après le step
    e.step(DO_NOTHING)
    assert e._last_dynamics["growth_rate"] == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# check_env (Gymnasium)
# ---------------------------------------------------------------------------


def test_env_check_env(cfg):
    from gymnasium.utils.env_checker import check_env
    e = VitruviusEnv(config=cfg, seed=42, max_turns=10)
    check_env(e, warn=True)


# ---------------------------------------------------------------------------
# Snapshot reward state — regression tests pour les nouveaux champs
# ---------------------------------------------------------------------------


def test_snapshot_total_houses_zero_on_empty_grid(cfg):
    """total_houses=0 sur grille vide après reset."""
    e = VitruviusEnv(config=cfg, seed=42)
    e.reset()
    snap = e._snapshot_reward_state()
    assert snap.total_houses == 0


def test_snapshot_marble_milestone_triggered(cfg):
    """reached_marble_50 True quand resource_state.marble >= 50."""
    e = VitruviusEnv(config=cfg, seed=42)
    e.reset()
    e.gs.resource_state.marble = 50
    snap = e._snapshot_reward_state()
    assert snap.reached_marble_50 is True
    assert snap.reached_marble_100 is False
    assert snap.reached_marble_200 is False


def test_snapshot_marble_milestones_100_and_200(cfg):
    """Paliers 100 et 200 activés correctement selon la valeur marble."""
    e = VitruviusEnv(config=cfg, seed=42)
    e.reset()
    e.gs.resource_state.marble = 200
    snap = e._snapshot_reward_state()
    assert snap.reached_marble_50 is True
    assert snap.reached_marble_100 is True
    assert snap.reached_marble_200 is True
    assert snap.reached_marble_500 is False


def test_snapshot_milestone_irreversible_after_marble_drops(cfg):
    """reached_marble_50 reste True même quand marble redescend sous 50."""
    e = VitruviusEnv(config=cfg, seed=42)
    e.reset()
    # Premier snapshot avec marble >= 50
    e.gs.resource_state.marble = 50
    snap1 = e._snapshot_reward_state()
    assert snap1.reached_marble_50 is True
    # Simule un paiement qui vide le stock (impossible en gameplay, mais test du contrat)
    e._prev_reward_state = snap1
    e.gs.resource_state.marble = 0
    snap2 = e._snapshot_reward_state()
    assert snap2.reached_marble_50 is True  # irréversible grâce à _keep()


def test_snapshot_house_level_milestone(cfg):
    """first_house_level_2 True quand une maison de niveau >= 2 existe."""
    from vitruvius.engine.population import HouseState
    e = VitruviusEnv(config=cfg, seed=42)
    e.reset()
    # Injection directe d'une maison niveau 2
    e.gs.houses[(5, 5)] = HouseState(origin=(5, 5), level=2, population=10)
    snap = e._snapshot_reward_state()
    assert snap.first_house_level_2 is True
    assert snap.first_house_level_3 is False
    assert snap.total_houses == 1


def test_snapshot_pop_milestone_triggered(cfg):
    """reached_pop_100 True quand sum(population) >= 100."""
    from vitruvius.engine.population import HouseState
    e = VitruviusEnv(config=cfg, seed=42)
    e.reset()
    e.gs.houses[(3, 3)] = HouseState(origin=(3, 3), level=2, population=100)
    snap = e._snapshot_reward_state()
    assert snap.reached_pop_100 is True
    assert snap.reached_pop_250 is False
    assert snap.first_population is True
