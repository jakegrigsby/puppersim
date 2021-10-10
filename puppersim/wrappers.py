import os
from collections import deque
import numpy as np

import gym
import puppersim
import gin
from pybullet_envs.minitaur.envs_v2 import env_loader
import puppersim.data as pd

import deep_control as dc
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor


def create_pupper_env(render=False, persistence=True):
    CONFIG_DIR = puppersim.getPupperSimPath() + "/"
    _CONFIG_FILE = os.path.join(CONFIG_DIR, "pupper_pmtg_slow.gin")
    gin.bind_parameter("scene_base.SceneBase.data_root", pd.getDataPath() + "/")
    gin.parse_config_file(_CONFIG_FILE)
    if render:
        gin.bind_parameter("SimulationParameters.enable_rendering", True)
    env = env_loader.load()
    env = dc.envs.ScaleReward(env, scale=100.0)
    env = dc.envs.NormalizeContinuousActionSpace(env)
    if persistence:
        env = dc.envs.PersistenceAwareWrapper(env)
    return env

class StateStack(gym.Wrapper):
    def __init__(self, env: gym.Env, num_stack: int, skip : int = 1):
        gym.Wrapper.__init__(self, env)
        self._k = num_stack
        self._frames = deque([], maxlen=num_stack * skip)
        shp = env.observation_space.shape[0]
        low = np.array([env.observation_space.low for _ in range(num_stack)]).flatten()
        high = np.array(
            [env.observation_space.high for _ in range(num_stack)]
        ).flatten()
        self.observation_space = gym.spaces.Box(
            low=low,
            high=high,
            shape=(shp * num_stack,),
            dtype=env.observation_space.dtype,
        )
        self._skip = skip

    def reset(self):
        obs = self.env.reset()
        for _ in range(self._k * self._skip):
            self._frames.append(obs)
        return self._get_obs()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._frames.append(obs)
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        assert len(self._frames) == self._k * self._skip
        obs = np.concatenate(list(self._frames)[::-self._skip], axis=0)
        return obs


class SqueezeRew(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)


    def reward(self, r):
        return r[0]


def make_sb3_env(rank, log_dir, seed=0):
    def _init():
        env = Monitor(
            SqueezeRew(gym.wrappers.TimeLimit(create_pupper_env(), 10_000)),
            log_dir,
        )
        env.seed(seed + rank)
        return env

    set_random_seed(seed)
    return _init


