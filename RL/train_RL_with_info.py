#!/usr/bin/env python3
"""Train a PPO controller with curriculum, evaluation, and secure export."""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecEnv,
    VecNormalize,
)

from swarm.constants import HORIZON_SEC, SAFE_META_FILENAME, SIM_DT
from swarm.training import (
    CurriculumConfig,
    ObservationNoiseWrapper,
    PotentialBasedRewardWrapper,
    TaskResamplingEnv,
    make_sequential_task_sampler,
)


def information_save(model: PPO, save_stem: str) -> None:
    """Append the minimal safe metadata required by the validator loader."""

    zip_path = Path(save_stem)
    if zip_path.suffix != ".zip":
        zip_path = zip_path.with_suffix(".zip")

    act_attr = getattr(model.policy, "activation_fn", th.nn.ReLU)
    act_name = act_attr.__name__ if isinstance(act_attr, type) else act_attr.__class__.__name__

    def _infer_net_arch() -> Any:
        extractor = getattr(model.policy, "mlp_extractor", None)
        if extractor is not None and hasattr(extractor, "net_arch"):
            return extractor.net_arch

        def _layers(seq) -> List[int]:
            out: List[int] = []
            for module in getattr(seq, "_modules", {}).values():
                if isinstance(module, th.nn.Linear):
                    out.append(int(module.out_features))
            return out

        shared = _layers(getattr(extractor, "shared_net", th.nn.Sequential()))
        pi = _layers(getattr(extractor, "policy_net", th.nn.Sequential()))
        vf = _layers(getattr(extractor, "value_net", th.nn.Sequential()))
        return (shared + [dict(pi=pi, vf=vf)]) if shared else dict(pi=pi, vf=vf)

    meta: Dict[str, Any] = {
        "format": "sb3-safe-meta@1",
        "algo": "PPO",
        "activation_fn": act_name,
        "net_arch": _infer_net_arch(),
        "use_sde": bool(getattr(model, "use_sde", False)),
    }

    with zipfile.ZipFile(zip_path, mode="a", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(SAFE_META_FILENAME, json.dumps(meta, indent=2))


class SwarmEvalCallback(BaseCallback):
    """Periodic evaluation with success-rate tracking and checkpointing."""

    def __init__(
        self,
        eval_env: VecEnv,
        *,
        eval_freq: int,
        n_eval_episodes: int,
        step_time: float,
        best_model_dir: Optional[Path] = None,
        deterministic: bool = True,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        self.eval_env = eval_env
        self.eval_freq = int(eval_freq)
        self.n_eval_episodes = int(n_eval_episodes)
        self.deterministic = deterministic
        self.best_model_dir = best_model_dir
        self.best_model_file: Optional[str] = None
        self.best_success = -np.inf
        self.last_result: Optional[Dict[str, float]] = None
        self._step_time = float(step_time)

    # SB3 hooks -----------------------------------------------------------------
    def _init_callback(self) -> None:
        if self.best_model_dir is not None:
            self.best_model_dir.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        if self.eval_freq <= 0 or self.n_eval_episodes <= 0:
            return True
        if self.num_timesteps % self.eval_freq != 0:
            return True

        result = self.evaluate()
        for key, value in result.items():
            self.logger.record(f"eval/{key}", value)

        if self.verbose > 0:
            print(
                f"\n[eval] step={self.num_timesteps:,} "
                f"success={result['success_rate']:.2%} reward={result['mean_reward']:.3f}"
            )

        if result["success_rate"] >= self.best_success:
            self.best_success = result["success_rate"]
            if self.best_model_dir is not None:
                best_path = self.best_model_dir / "best_model"
                self.model.save(str(best_path))
                self.best_model_file = str(best_path.with_suffix(".zip"))
        return True

    # Helper utilities -----------------------------------------------------------
    def _reset_eval_sampler(self) -> None:
        base = self.eval_env
        while isinstance(base, VecNormalize):
            base = base.venv
        # base is now a VecEnv (Dummy/Subproc)
        envs = getattr(base, "envs", [])
        for env in envs:
            inner = env
            while hasattr(inner, "env"):
                inner = inner.env
            if hasattr(inner, "_episode_idx"):
                inner._episode_idx = 0

    def evaluate(self) -> Dict[str, float]:
        if isinstance(self.eval_env, VecNormalize):
            self.eval_env.training = False
            if isinstance(self.training_env, VecNormalize):
                self.eval_env.obs_rms = self.training_env.obs_rms
                self.eval_env.clip_obs = self.training_env.clip_obs
        self._reset_eval_sampler()

        rewards: List[float] = []
        lengths: List[int] = []
        successes: List[float] = []
        times: List[float] = []

        for _ in range(self.n_eval_episodes):
            obs = self.eval_env.reset()
            done = False
            total_reward = 0.0
            length = 0
            last_info: Dict[str, Any] = {}

            while not done:
                action, _ = self.model.predict(obs, deterministic=self.deterministic)
                obs, reward, terminated, truncated, infos = self.eval_env.step(action)
                done = bool(terminated[0] or truncated[0])
                total_reward += float(reward[0])
                length += 1
                last_info = infos[0]

            rewards.append(total_reward)
            lengths.append(length)
            successes.append(1.0 if last_info.get("success", False) else 0.0)
            t_to_goal = last_info.get("t_to_goal")
            if t_to_goal is None:
                t_to_goal = length * self._step_time
            times.append(float(t_to_goal))

        result = {
            "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
            "reward_std": float(np.std(rewards)) if len(rewards) > 1 else 0.0,
            "success_rate": float(np.mean(successes)) if successes else 0.0,
            "avg_time": float(np.mean(times)) if times else 0.0,
            "mean_length": float(np.mean(lengths)) if lengths else 0.0,
        }
        self.last_result = result
        return result


def build_vec_env(fns: List, num_envs: int) -> VecEnv:
    if num_envs <= 1:
        return DummyVecEnv(fns)
    return SubprocVecEnv(fns, start_method="spawn")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Advanced PPO trainer for the Swarm drone environment.")
    parser.add_argument("--total-timesteps", type=int, default=2_000_000, help="Total number of training timesteps.")
    parser.add_argument("--num-envs", type=int, default=8, help="Number of parallel environments for vectorised PPO.")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="PPO learning rate.")
    parser.add_argument("--gamma", type=float, default=0.995, help="Discount factor used by PPO.")
    parser.add_argument("--n-steps", type=int, default=1024, help="Rollout length per environment before an optimisation step.")
    parser.add_argument("--batch-size", type=int, default=512, help="Mini-batch size during PPO updates.")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="Generalised advantage estimation lambda.")
    parser.add_argument("--clip-range", type=float, default=0.2, help="PPO clipping range.")
    parser.add_argument("--ent-coef", type=float, default=0.01, help="Entropy coefficient to encourage exploration.")
    parser.add_argument("--max-grad-norm", type=float, default=0.5, help="Gradient clipping value.")
    parser.add_argument("--n-epochs", type=int, default=10, help="Number of optimisation epochs per PPO update.")
    parser.add_argument("--obs-noise", type=float, default=0.02, help="Standard deviation of Gaussian observation noise during training.")
    parser.add_argument("--distance-scale", type=float, default=0.25, help="Scaling applied to the potential-based shaping reward.")
    parser.add_argument("--action-penalty", type=float, default=0.01, help="Weight for penalising large velocity commands.")
    parser.add_argument("--crash-penalty", type=float, default=2.0, help="Extra penalty when a collision terminates the episode.")
    parser.add_argument("--curriculum-episodes", type=int, default=600, help="Number of episodes to anneal from easy to full difficulty.")
    parser.add_argument("--eval-freq", type=int, default=100_000, help="Frequency (in timesteps) for evaluation runs.")
    parser.add_argument("--eval-episodes", type=int, default=5, help="How many episodes to average per evaluation.")
    parser.add_argument("--eval-seeds", type=int, nargs="*", default=[101, 202, 303, 404, 505], help="Deterministic seeds used for evaluation tasks.")
    parser.add_argument("--save-dir", default="model", help="Directory to store the exported policy and artefacts.")
    parser.add_argument("--checkpoint-dir", default="model/checkpoints", help="Directory where best-model checkpoints are stored.")
    parser.add_argument("--tensorboard-log", default="runs/ppo_drone", help="TensorBoard logging directory.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")
    parser.add_argument("--device", default="auto", help="Torch device (cpu, cuda, cuda:0, …).")
    parser.add_argument("--no-sde", action="store_true", help="Disable state-dependent exploration noise.")
    parser.add_argument("--gui", action="store_true", help="Render the first training environment for debugging.")
    parser.add_argument("--eval-gui", action="store_true", help="Render the evaluation environment when running callbacks.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_random_seed(args.seed)

    if args.num_envs < 1:
        raise ValueError("--num-envs must be at least 1")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    curriculum = None if args.curriculum_episodes <= 0 else CurriculumConfig(warmup_episodes=args.curriculum_episodes)

    def make_env(rank: int):
        env_seed = args.seed + rank * 9973

        def _init():
            env = TaskResamplingEnv(
                sim_dt=SIM_DT,
                horizon=HORIZON_SEC,
                gui=args.gui and rank == 0,
                curriculum=curriculum,
                seed=env_seed,
            )
            env = PotentialBasedRewardWrapper(
                env,
                gamma=args.gamma,
                distance_scale=args.distance_scale,
                action_penalty=args.action_penalty,
                crash_penalty=args.crash_penalty,
            )
            if args.obs_noise > 0.0:
                env = ObservationNoiseWrapper(env, noise_std=args.obs_noise, seed=env_seed)
            return Monitor(env)

        return _init

    train_env = VecNormalize(
        build_vec_env([make_env(i) for i in range(args.num_envs)], args.num_envs),
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
    )

    eval_sampler = make_sequential_task_sampler(args.eval_seeds, sim_dt=SIM_DT, horizon=HORIZON_SEC)

    def make_eval_env():
        env = TaskResamplingEnv(
            sim_dt=SIM_DT,
            horizon=HORIZON_SEC,
            gui=args.eval_gui,
            task_sampler=eval_sampler,
        )
        env = PotentialBasedRewardWrapper(env, gamma=args.gamma, distance_scale=0.0, crash_penalty=args.crash_penalty)
        return Monitor(env)

    eval_env = VecNormalize(
        DummyVecEnv([make_eval_env]),
        training=False,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
    )

    policy_kwargs = dict(
        activation_fn=th.nn.SiLU,
        net_arch=dict(pi=[256, 256, 128], vf=[256, 256, 128]),
        ortho_init=False,
        log_std_init=-1.0,
    )

    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=0.7,
        max_grad_norm=args.max_grad_norm,
        n_epochs=args.n_epochs,
        use_sde=not args.no_sde,
        sde_sample_freq=4,
        tensorboard_log=args.tensorboard_log,
        verbose=1,
        device=args.device,
        policy_kwargs=policy_kwargs,
    )

    eval_callback = SwarmEvalCallback(
        eval_env,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.eval_episodes,
        step_time=SIM_DT,
        best_model_dir=checkpoint_dir,
        deterministic=True,
        verbose=1,
    )

    model.learn(total_timesteps=args.total_timesteps, callback=eval_callback)

    # Optionally reload the best checkpoint for export
    if eval_callback.best_model_file and Path(eval_callback.best_model_file).exists():
        model = PPO.load(eval_callback.best_model_file, env=train_env, device=args.device)

    # Final evaluation with the best weights
    final_metrics = eval_callback.evaluate()
    print("Final evaluation:", final_metrics)

    # Persist VecNormalize statistics for future rollouts
    vecnorm_path = save_dir / "vecnormalize.pkl"
    train_env.save(str(vecnorm_path))

    # Export model and metadata
    save_stem = save_dir / "ppo_policy"
    model.save(str(save_stem))
    information_save(model, str(save_stem))

    # Record training configuration for reproducibility
    with (save_dir / "training_config.json").open("w", encoding="utf-8") as fh:
        json.dump(vars(args), fh, indent=2)

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
