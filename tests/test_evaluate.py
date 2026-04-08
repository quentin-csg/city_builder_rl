"""Tests pour vitruvius/rl/evaluate.py : run_episode, aggregate_stats, argparser, smoke."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest
from sb3_contrib import MaskablePPO

from vitruvius.config import load_config
from vitruvius.rl.evaluate import (
    DEFAULT_MAX_TURNS,
    DEFAULT_N_EPISODES,
    DEFAULT_SEED,
    aggregate_stats,
    build_argparser,
    evaluate,
    run_episode,
)
from vitruvius.rl.gym_env import VitruviusEnv
from vitruvius.rl.train import build_vec_env, train


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EPISODE_KEYS = {
    "total_reward",
    "episode_length",
    "victory",
    "defeat",
    "bankrupt",
    "final_city_level",
    "final_population",
    "final_satisfaction",
    "final_denarii",
    "final_turn",
}

_AGG_KEYS = {
    "n_episodes",
    "victory_rate",
    "defeat_rate",
    "bankrupt_rate",
    "reward_mean",
    "reward_median",
    "reward_std",
    "reward_min",
    "reward_max",
    "length_mean",
    "length_median",
    "city_level_mean",
    "city_level_max",
    "population_mean",
    "population_max",
    "satisfaction_mean",
}


def _make_untrained_model(tmp_path: Path) -> Path:
    """Entraine un modele minimal (256 steps) et retourne le chemin du .zip."""
    args = argparse.Namespace(
        total_timesteps=256,
        n_envs=1,
        max_turns=50,
        seed=0,
        run_name="eval_smoke",
        run_dir=str(tmp_path / "runs"),
        model_dir=str(tmp_path / "models"),
        checkpoint_freq=1_000_000,
        subproc=False,
        resume=None,
        learning_rate=3e-4,
        n_steps=64,
        batch_size=64,
        n_epochs=1,
        gamma=0.99,
        ent_coef=0.01,
    )
    return train(args)


def _eval_args(tmp_path: Path, model_path: Path) -> argparse.Namespace:
    """Namespace minimal pour un smoke test evaluate tres court."""
    return argparse.Namespace(
        model_path=str(model_path),
        n_episodes=2,
        max_turns=20,
        seed=0,
        stochastic=False,
    )


# ---------------------------------------------------------------------------
# Tests run_episode
# ---------------------------------------------------------------------------


class TestRunEpisode:
    def _make_model_and_env(self, tmp_path: Path):
        model_path = _make_untrained_model(tmp_path)
        model = MaskablePPO.load(str(model_path))
        config = load_config()
        env = VitruviusEnv(config=config, seed=0, max_turns=20)
        return model, env

    def test_returns_expected_keys(self, tmp_path: Path) -> None:
        model, env = self._make_model_and_env(tmp_path)
        stats = run_episode(model, env, seed=0)
        env.close()
        assert _EPISODE_KEYS == set(stats.keys())

    def test_episode_length_bounded_by_max_turns(self, tmp_path: Path) -> None:
        model, env = self._make_model_and_env(tmp_path)
        stats = run_episode(model, env, seed=0)
        env.close()
        assert stats["episode_length"] <= 20

    def test_victory_defeat_are_bool(self, tmp_path: Path) -> None:
        model, env = self._make_model_and_env(tmp_path)
        stats = run_episode(model, env, seed=0)
        env.close()
        assert isinstance(stats["victory"], bool)
        assert isinstance(stats["defeat"], bool)
        assert isinstance(stats["bankrupt"], bool)

    def test_city_level_in_valid_range(self, tmp_path: Path) -> None:
        model, env = self._make_model_and_env(tmp_path)
        stats = run_episode(model, env, seed=0)
        env.close()
        assert 1 <= stats["final_city_level"] <= 5

    def test_stochastic_flag(self, tmp_path: Path) -> None:
        model, env = self._make_model_and_env(tmp_path)
        stats = run_episode(model, env, seed=0, deterministic=False)
        env.close()
        assert _EPISODE_KEYS == set(stats.keys())


# ---------------------------------------------------------------------------
# Tests aggregate_stats
# ---------------------------------------------------------------------------


class TestAggregateStats:
    def _fake_episode(self, reward: float, length: int, victory: bool = False) -> dict:
        return {
            "total_reward": reward,
            "episode_length": length,
            "victory": victory,
            "defeat": not victory,
            "bankrupt": False,
            "final_city_level": 1,
            "final_population": 50,
            "final_satisfaction": 0.5,
            "final_denarii": 1000.0,
            "final_turn": length,
        }

    def test_single_episode_no_crash(self) -> None:
        ep = self._fake_episode(reward=5.0, length=10)
        agg = aggregate_stats([ep])
        assert agg["n_episodes"] == 1
        assert agg["reward_std"] == 0.0
        assert _AGG_KEYS == set(agg.keys())

    def test_multiple_episodes_means(self) -> None:
        episodes = [
            self._fake_episode(reward=10.0, length=20, victory=True),
            self._fake_episode(reward=0.0, length=10, victory=False),
            self._fake_episode(reward=5.0, length=15, victory=True),
        ]
        agg = aggregate_stats(episodes)
        assert agg["n_episodes"] == 3
        assert abs(agg["reward_mean"] - 5.0) < 1e-9
        assert abs(agg["victory_rate"] - 2 / 3) < 1e-9
        assert agg["reward_min"] == 0.0
        assert agg["reward_max"] == 10.0
        assert agg["length_median"] == 15.0

    def test_rates_sum_not_enforced(self) -> None:
        """victory_rate + defeat_rate peuvent ne pas sommer a 1 (truncation possible)."""
        episodes = [self._fake_episode(reward=0.0, length=5, victory=False)]
        agg = aggregate_stats(episodes)
        assert 0.0 <= agg["victory_rate"] <= 1.0
        assert 0.0 <= agg["defeat_rate"] <= 1.0


# ---------------------------------------------------------------------------
# Tests argparser
# ---------------------------------------------------------------------------


class TestArgparser:
    def test_defaults(self) -> None:
        p = build_argparser()
        args = p.parse_args(["dummy_model.zip"])
        assert args.model_path == "dummy_model.zip"
        assert args.n_episodes == DEFAULT_N_EPISODES
        assert args.max_turns == DEFAULT_MAX_TURNS
        assert args.seed == DEFAULT_SEED
        assert args.stochastic is False

    def test_stochastic_flag(self) -> None:
        p = build_argparser()
        args = p.parse_args(["model.zip", "--stochastic"])
        assert args.stochastic is True

    def test_custom_values(self) -> None:
        p = build_argparser()
        args = p.parse_args(["model.zip", "--n-episodes", "5", "--seed", "100"])
        assert args.n_episodes == 5
        assert args.seed == 100


# ---------------------------------------------------------------------------
# Smoke test end-to-end
# ---------------------------------------------------------------------------


class TestSmokeEvaluate:
    def test_smoke_evaluate_end_to_end(self, tmp_path: Path) -> None:
        """evaluate() tourne sans exception et retourne les bonnes cles."""
        model_path = _make_untrained_model(tmp_path)
        args = _eval_args(tmp_path, model_path)
        agg = evaluate(args)
        assert _AGG_KEYS == set(agg.keys())
        assert agg["n_episodes"] == 2
        assert 0.0 <= agg["victory_rate"] <= 1.0
        assert agg["length_mean"] > 0

    def test_smoke_model_not_found_raises(self, tmp_path: Path) -> None:
        args = _eval_args(tmp_path, tmp_path / "missing.zip")
        with pytest.raises(FileNotFoundError):
            evaluate(args)
