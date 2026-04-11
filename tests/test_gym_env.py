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
    """DO_NOTHING sur grille vide → pas de delta, mais W_POSITIVE_INCOME car passif > maintenance."""
    _, reward, terminated, truncated, _ = env.step(DO_NOTHING)
    if not terminated:
        # passif 20 > maintenance 0 → bonus W_POSITIVE_INCOME = 0.05
        assert reward == pytest.approx(0.05, abs=1e-6)


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
# check_env (Gymnasium)
# ---------------------------------------------------------------------------


def test_env_check_env(cfg):
    from gymnasium.utils.env_checker import check_env
    e = VitruviusEnv(config=cfg, seed=42, max_turns=10)
    check_env(e, warn=True)
