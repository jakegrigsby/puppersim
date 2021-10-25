import argparse
import os
import gym
import puppersim
import functools
import gin
from pybullet_envs.minitaur.envs_v2 import env_loader
import puppersim.data as pd
import cv2

from super_sac.nets import distributions, weight_init
from super_sac.augmentations import AugmentationSequence, Drqv2Aug

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import gym

import puppersim
import pybullet
import pybullet_envs

import super_sac
from super_sac.wrappers import (
    SimpleGymWrapper,
    NormActionSpace,
    ParallelActors,
    ScaleReward,
    FrameStack,
    FrameSkip,
    StateStack,
    Uint8Wrapper,
)


class PupperFromVision(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._img = None
        self.env = env
        self.reset()

    def reset(self):
        self.env.reset()
        self._img = self.env.render("rgb_array").transpose(2, 0, 1)
        return self._img

    RENDER_DELAY = 1

    def render(self, *args, **kwargs):
        cv2.imshow("Puppersim", self._img.transpose(1, 2, 0))
        cv2.waitKey(self.RENDER_DELAY)

    def step(self, action):
        next_state, rew, done, info = self.env.step(action)
        self._img = self.env.render("rgb_array").transpose(2, 0, 1)
        return self._img, rew, done, info


class NoRender(gym.Wrapper):
    def render(self, *args, **kwargs):
        pass

@gin.configurable
def create_pupper_env(render=False, from_pixels=True, skip=0, stack=1):
    # build env from pybullet config
    CONFIG_DIR = puppersim.getPupperSimPath() + "/"
    _CONFIG_FILE = os.path.join(CONFIG_DIR, "pupper_manual_rl.gin")
    gin.bind_parameter("scene_base.SceneBase.data_root", pd.getDataPath() + "/")
    gin.parse_config_file(_CONFIG_FILE)
    if render:
        gin.bind_parameter("SimulationParameters.enable_rendering", True)
    env = env_loader.load()

    if from_pixels:
        env = PupperFromVision(env)
        env = FrameStack(env, num_stack=stack)
    else:
        env = StateStack(env, num_stack=stack)
        env = NoRender(env)
    env = FrameSkip(env, skip=skip)

    env = ScaleReward(env, scale=100.0)
    env = NormActionSpace(env)

    if from_pixels:
        env = Uint8Wrapper(env)
    else:
        env = SimpleGymWrapper(env)
    return env


class CCEncoder(super_sac.nets.Encoder):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self._dim = out_dim

    @property
    def embedding_dim(self):
        return self._dim

    def forward(self, obs_dict):
        return obs_dict["obs"]


class VisionEncoder(super_sac.nets.Encoder):
    def __init__(self, inp_shape, emb_dim=50):
        super().__init__()
        self._dim = emb_dim
        self.conv_block = super_sac.nets.cnns.BigPixelEncoder(inp_shape, emb_dim)

    def forward(self, obs_dict):
        emb = self.conv_block(obs_dict["obs"])
        return emb

    @property
    def embedding_dim(self):
        return self._dim


@gin.configurable
def train_pupper(
    name,
    from_pixels=False,
    batch_size=1024,
):
    train_env = create_pupper_env()
    test_env = create_pupper_env()
    """
    state = train_env.reset()["obs"]
    import tqdm
    for _ in tqdm.tqdm(range(1000)):
        *_, done, _ = train_env.step(train_env.action_space.sample())
        if done: train_env.reset()
    """

    if from_pixels:
        encoder = VisionEncoder(test_env.reset()["obs"].shape, 50)
        augmenter=AugmentationSequence([Drqv2Aug(batch_size)])
    else:
        encoder = CCEncoder(None, train_env.observation_space.shape[0])
        augmenter = None

    # create agent
    agent = super_sac.Agent(
        act_space_size=train_env.action_space.shape[0],
        encoder=encoder,
        actor_network_cls=super_sac.nets.mlps.ContinuousStochasticActor,
        critic_network_cls=super_sac.nets.mlps.ContinuousCritic,
    )

    buffer = super_sac.replay.PrioritizedReplayBuffer(size=1_000_000)

    # run training
    super_sac.super_sac(
        agent=agent,
        train_env=train_env,
        test_env=test_env,
        buffer=buffer,
        name=name,
        logging_method="wandb",
        augmenter=augmenter,
    )

def main(args):
    gin.parse_config_file(args.config)
    train_pupper(name=args.name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args)
