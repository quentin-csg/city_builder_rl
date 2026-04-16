"""Tests pour vitruvius/rl/train.py : factory, vec_env, modele, smoke test."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pytest
from sb3_contrib import MaskablePPO

from vitruvius.rl.gym_env import VitruviusEnv
from vitruvius.rl.train import (
    DEFAULT_CHECKPOINT_FREQ,
    DEFAULT_MAX_TURNS,
    DEFAULT_N_ENVS,
    DEFAULT_SEED,
    DEFAULT_TOTAL_TIMESTEPS,
    build_argparser,
    build_model,
    build_vec_env,
    make_env_fn,
    train,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _smoke_args(tmp_path: Path) -> argparse.Namespace:
    """Retourne un Namespace minimal pour un smoke test tres court."""
    return argparse.Namespace(
        total_timesteps=256,
        n_envs=1,
        max_turns=50,
        seed=0,
        run_name="smoke",
        run_dir=str(tmp_path / "runs"),
        model_dir=str(tmp_path / "models"),
        checkpoint_freq=1_000_000,  # pas de checkpoint intermediaire
        subproc=False,
        resume=None,
        learning_rate=3e-4,
        lr_end=1e-4,
        n_steps=64,
        batch_size=64,
        n_epochs=1,
        gamma=0.995,
        ent_coef=0.01,
    )


# ---------------------------------------------------------------------------
# Tests factory
# ---------------------------------------------------------------------------


class TestMakeEnvFn:
    def test_returns_callable(self) -> None:
        fn = make_env_fn(seed=0, max_turns=100)
        assert callable(fn)

    def test_creates_vitruvius_env(self) -> None:
        fn = make_env_fn(seed=0, max_turns=100)
        env = fn()
        assert isinstance(env, VitruviusEnv)
        env.close()

    def test_max_turns_propagated(self) -> None:
        fn = make_env_fn(seed=0, max_turns=42)
        env = fn()
        assert env.max_turns == 42
        env.close()

    def test_different_seeds_give_independent_envs(self) -> None:
        # Seeds suffisamment eloignes pour eviter les collisions lors des retries terrain
        env_a = make_env_fn(seed=0, max_turns=50)()
        env_b = make_env_fn(seed=1000, max_turns=50)()
        obs_a, _ = env_a.reset(seed=0)
        obs_b, _ = env_b.reset(seed=1000)
        grids_differ = not np.array_equal(obs_a["grid"], obs_b["grid"])
        assert grids_differ
        env_a.close()
        env_b.close()


# ---------------------------------------------------------------------------
# Tests build_vec_env
# ---------------------------------------------------------------------------


class TestBuildVecEnv:
    def test_creates_dummy_vec_env(self) -> None:
        from stable_baselines3.common.vec_env import DummyVecEnv

        vec = build_vec_env(n_envs=2, seed=0, max_turns=50, subproc=False)
        assert isinstance(vec, DummyVecEnv)
        assert vec.num_envs == 2
        vec.close()

    def test_reset_returns_batch(self) -> None:
        vec = build_vec_env(n_envs=2, seed=0, max_turns=50, subproc=False)
        obs = vec.reset()
        # DummyVecEnv avec Dict obs retourne un dict de tableaux (batch)
        assert "grid" in obs
        assert obs["grid"].shape[0] == 2
        vec.close()

    def test_action_masks_accessible(self) -> None:
        vec = build_vec_env(n_envs=2, seed=0, max_turns=50, subproc=False)
        vec.reset()
        masks = vec.env_method("action_masks")
        assert len(masks) == 2
        for mask in masks:
            assert mask.dtype == np.bool_
            assert mask.any()  # au moins une action valide (DO_NOTHING)
        vec.close()

    def test_single_env(self) -> None:
        vec = build_vec_env(n_envs=1, seed=42, max_turns=100, subproc=False)
        assert vec.num_envs == 1
        vec.close()


# ---------------------------------------------------------------------------
# Tests build_model
# ---------------------------------------------------------------------------


class TestBuildModel:
    def test_creates_maskable_ppo(self, tmp_path: Path) -> None:
        vec = build_vec_env(n_envs=1, seed=0, max_turns=50, subproc=False)
        args = _smoke_args(tmp_path)
        model = build_model(vec, tmp_path / "runs", args)
        assert isinstance(model, MaskablePPO)
        vec.close()

    def test_policy_is_multiinput(self, tmp_path: Path) -> None:
        vec = build_vec_env(n_envs=1, seed=0, max_turns=50, subproc=False)
        args = _smoke_args(tmp_path)
        model = build_model(vec, tmp_path / "runs", args)
        # Policy doit supporter les observations Dict
        assert model.policy is not None
        vec.close()

    def test_hyperparams_applied(self, tmp_path: Path) -> None:
        vec = build_vec_env(n_envs=1, seed=0, max_turns=50, subproc=False)
        args = _smoke_args(tmp_path)
        args.learning_rate = 1e-3
        args.gamma = 0.95
        model = build_model(vec, tmp_path / "runs", args)
        assert abs(model.gamma - 0.95) < 1e-9
        vec.close()


# ---------------------------------------------------------------------------
# Tests argparser
# ---------------------------------------------------------------------------


class TestArgparser:
    def test_defaults(self) -> None:
        p = build_argparser()
        args = p.parse_args([])
        assert args.total_timesteps == DEFAULT_TOTAL_TIMESTEPS
        assert args.n_envs == DEFAULT_N_ENVS
        assert args.max_turns == DEFAULT_MAX_TURNS
        assert args.seed == DEFAULT_SEED
        assert args.checkpoint_freq == DEFAULT_CHECKPOINT_FREQ
        assert args.run_name == "nova_roma"
        assert args.run_dir == "runs"
        assert args.model_dir == "models"
        assert args.subproc is False
        assert args.resume is None
        assert abs(args.learning_rate - 3e-4) < 1e-10
        assert args.n_steps == 2048
        assert args.batch_size == 256
        assert args.n_epochs == 10
        assert abs(args.gamma - 0.995) < 1e-10
        assert abs(args.ent_coef - 0.01) < 1e-10

    def test_override_timesteps(self) -> None:
        p = build_argparser()
        args = p.parse_args(["--total-timesteps", "500000"])
        assert args.total_timesteps == 500_000

    def test_subproc_flag(self) -> None:
        p = build_argparser()
        args = p.parse_args(["--subproc"])
        assert args.subproc is True

    def test_resume_flag(self) -> None:
        p = build_argparser()
        args = p.parse_args(["--resume", "models/vitruvius_100000_steps.zip"])
        assert args.resume == "models/vitruvius_100000_steps.zip"


# ---------------------------------------------------------------------------
# Smoke test d'entrainement
# ---------------------------------------------------------------------------


class TestSmokeTrain:
    def test_smoke_train_short(self, tmp_path: Path) -> None:
        """train() tourne sans exception sur 256 timesteps et produit le fichier final."""
        args = _smoke_args(tmp_path)
        final_path = train(args)
        assert final_path.exists(), f"Fichier final absent : {final_path}"
        assert final_path.suffix == ".zip"

    def test_smoke_train_creates_run_dir(self, tmp_path: Path) -> None:
        args = _smoke_args(tmp_path)
        train(args)
        run_dir = tmp_path / "runs"
        assert run_dir.exists()

    def test_smoke_train_creates_model_dir(self, tmp_path: Path) -> None:
        args = _smoke_args(tmp_path)
        train(args)
        model_dir = tmp_path / "models" / "smoke"
        assert model_dir.exists()

    def test_smoke_train_model_loadable(self, tmp_path: Path) -> None:
        """Le modele final peut etre recharge avec MaskablePPO.load."""
        args = _smoke_args(tmp_path)
        final_path = train(args)
        vec = build_vec_env(n_envs=1, seed=0, max_turns=50, subproc=False)
        loaded = MaskablePPO.load(str(final_path), env=vec)
        assert loaded is not None
        vec.close()
