"""Utility environments and wrappers for richer drone RL training."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np

from swarm.constants import (
    H_MAX,
    H_MIN,
    MAX_RAY_DISTANCE,
    R_MAX,
    R_MIN,
)
from swarm.protocol import MapTask
from swarm.utils.env_factory import make_env
from swarm.validator.task_gen import random_task

TaskSampler = Callable[[np.random.Generator, int], MapTask]


@dataclass
class CurriculumConfig:
    """Simple curriculum for gradually expanding task difficulty."""

    warmup_episodes: int = 500
    initial_r_bounds: Tuple[float, float] = (6.0, 18.0)
    final_r_bounds: Tuple[float, float] = (float(R_MIN), float(R_MAX))
    initial_h_bounds: Tuple[float, float] = (float(H_MIN), 6.0)
    final_h_bounds: Tuple[float, float] = (float(H_MIN), float(H_MAX))

    def _progress(self, episode_idx: int) -> float:
        if self.warmup_episodes <= 0:
            return 1.0
        return float(np.clip(episode_idx / max(1, self.warmup_episodes), 0.0, 1.0))

    def radius_bounds(self, episode_idx: int) -> Tuple[float, float]:
        progress = self._progress(episode_idx)
        start_min, start_max = self.initial_r_bounds
        final_min, final_max = self.final_r_bounds
        r_min = (1.0 - progress) * start_min + progress * final_min
        r_max = (1.0 - progress) * start_max + progress * final_max
        lower, upper = sorted((float(r_min), float(r_max)))
        return lower, upper

    def height_bounds(self, episode_idx: int) -> Tuple[float, float]:
        progress = self._progress(episode_idx)
        start_min, start_max = self.initial_h_bounds
        final_min, final_max = self.final_h_bounds
        h_min = (1.0 - progress) * start_min + progress * final_min
        h_max = (1.0 - progress) * start_max + progress * final_max
        lower, upper = sorted((float(h_min), float(h_max)))
        return lower, upper


class TaskResamplingEnv(gym.Env):
    """Gymnasium environment that rebuilds the PyBullet world every reset."""

    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(
        self,
        *,
        sim_dt: float,
        horizon: float,
        gui: bool = False,
        task_sampler: Optional[TaskSampler] = None,
        curriculum: Optional[CurriculumConfig] = None,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        self._sim_dt = float(sim_dt)
        self._horizon = float(horizon)
        self._gui = bool(gui)
        self._task_sampler = task_sampler
        self._curriculum = curriculum
        self._rng = np.random.default_rng(seed)
        self._env = None
        self._current_task: Optional[MapTask] = None
        self._episode_idx = 0

        # Create an initial environment to expose action/observation spaces.
        self._build_new_env()
        assert self._env is not None  # for mypy
        self.action_space = self._env.action_space
        self.observation_space = self._env.observation_space

    # ------------------------------------------------------------------
    # Environment lifecycle helpers
    # ------------------------------------------------------------------
    def _sample_task(self) -> MapTask:
        if self._task_sampler is not None:
            task = self._task_sampler(self._rng, self._episode_idx)
            if not isinstance(task, MapTask):
                raise TypeError("Task sampler must return a MapTask instance")
            return task

        # Default sampler that honours the curriculum
        seed = int(self._rng.integers(0, 2**32 - 1))
        if self._curriculum is not None:
            r_min, r_max = self._curriculum.radius_bounds(self._episode_idx)
            h_min, h_max = self._curriculum.height_bounds(self._episode_idx)
        else:
            r_min, r_max = float(R_MIN), float(R_MAX)
            h_min, h_max = float(H_MIN), float(H_MAX)

        angle = float(self._rng.uniform(0.0, 2 * np.pi))
        radius = float(self._rng.uniform(r_min, r_max))
        goal_x = radius * float(np.cos(angle))
        goal_y = radius * float(np.sin(angle))
        goal_z = float(self._rng.uniform(h_min, h_max))

        return MapTask(
            map_seed=seed,
            start=(0.0, 0.0, 1.5),
            goal=(goal_x, goal_y, goal_z),
            sim_dt=self._sim_dt,
            horizon=self._horizon,
            version="1",
        )

    def _build_new_env(self) -> None:
        task = self._sample_task()
        if self._env is not None:
            self._env.close()
        env = make_env(task, gui=self._gui)
        self._env = env
        self._current_task = task

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):  # type: ignore[override]
        self._build_new_env()
        assert self._env is not None
        obs, info = self._env.reset(seed=seed, options=options)
        info = dict(info)
        if self._current_task is not None:
            info.setdefault("task_seed", self._current_task.map_seed)
            info.setdefault("goal", self._current_task.goal)
            if self._curriculum is not None:
                info.setdefault(
                    "curriculum_progress",
                    self._curriculum._progress(self._episode_idx),
                )
        self._episode_idx += 1
        return obs, info

    def step(self, action):  # type: ignore[override]
        assert self._env is not None
        obs, reward, terminated, truncated, info = self._env.step(action)
        info = dict(info)
        if self._current_task is not None:
            info.setdefault("task_seed", self._current_task.map_seed)
            info.setdefault("goal", self._current_task.goal)
        return obs, reward, terminated, truncated, info

    def render(self):  # type: ignore[override]
        if self._env is not None:
            return self._env.render()
        return None

    def close(self):  # type: ignore[override]
        if self._env is not None:
            self._env.close()
            self._env = None
        super().close()

    # Convenience accessor -------------------------------------------------
    @property
    def current_task(self) -> Optional[MapTask]:
        return self._current_task


class ObservationNoiseWrapper(gym.ObservationWrapper):
    """Inject Gaussian noise into observations during training."""

    def __init__(self, env: gym.Env, noise_std: float = 0.01, seed: Optional[int] = None) -> None:
        super().__init__(env)
        self._noise_std = float(max(0.0, noise_std))
        self._rng = np.random.default_rng(seed)

    def observation(self, observation):  # type: ignore[override]
        obs = np.asarray(observation, dtype=np.float32)
        if self._noise_std <= 0.0:
            return obs
        noise = self._rng.normal(loc=0.0, scale=self._noise_std, size=obs.shape).astype(np.float32)
        return obs + noise


class PotentialBasedRewardWrapper(gym.Wrapper):
    """Dense shaping based on potential (distance to goal)."""

    def __init__(
        self,
        env: gym.Env,
        *,
        gamma: float = 0.99,
        distance_scale: float = 1.0,
        action_penalty: float = 0.0,
        crash_penalty: float = 0.0,
    ) -> None:
        super().__init__(env)
        self._gamma = float(gamma)
        self._distance_scale = float(distance_scale)
        self._action_penalty = float(max(0.0, action_penalty))
        self._crash_penalty = float(max(0.0, crash_penalty))
        self._prev_potential = 0.0

    def reset(self, **kwargs):  # type: ignore[override]
        obs, info = self.env.reset(**kwargs)
        self._prev_potential = self._potential(obs)
        return obs, info

    def step(self, action):  # type: ignore[override]
        obs, reward, terminated, truncated, info = self.env.step(action)
        potential = self._potential(obs)
        shaping = self._gamma * potential - self._prev_potential
        shaped_reward = float(reward) + self._distance_scale * shaping

        if self._action_penalty > 0.0:
            shaped_reward -= self._action_penalty * float(np.linalg.norm(action))

        if self._crash_penalty > 0.0 and info.get("collision", False):
            shaped_reward -= self._crash_penalty

        self._prev_potential = potential
        return obs, shaped_reward, terminated, truncated, info

    @staticmethod
    def _potential(obs) -> float:
        arr = np.asarray(obs, dtype=np.float32).reshape(-1)
        # Last three elements are the relative goal vector scaled by MAX_RAY_DISTANCE
        rel = arr[-3:] * float(MAX_RAY_DISTANCE)
        distance = float(np.linalg.norm(rel))
        return -distance


def make_sequential_task_sampler(
    seeds: Sequence[int],
    *,
    sim_dt: float,
    horizon: float,
) -> TaskSampler:
    """Create a sampler that cycles through a fixed list of seeds."""

    seed_cycle = tuple(int(s) for s in seeds)

    def _sampler(rng: np.random.Generator, episode_idx: int) -> MapTask:
        if not seed_cycle:
            # Fallback to default random task if no seeds supplied
            return random_task(sim_dt=sim_dt, horizon=horizon, seed=int(rng.integers(0, 2**32 - 1)))
        idx = episode_idx % len(seed_cycle)
        return random_task(sim_dt=sim_dt, horizon=horizon, seed=seed_cycle[idx])

    return _sampler

