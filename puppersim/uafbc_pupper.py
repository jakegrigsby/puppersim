import argparse
import os
import gym
import puppersim
import functools
import gin
from pybullet_envs.minitaur.envs_v2 import env_loader
import puppersim.data as pd
import cv2

from uafbc.nets import distributions, weight_init
from uafbc.augmentations import AugmentationSequence, DrqAug

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import gym

import puppersim
import pybullet
import pybullet_envs

import uafbc
from uafbc.wrappers import (
    SimpleGymWrapper,
    NormActionSpace,
    ParallelActors,
    ScaleReward,
    FrameStack,
    FrameSkip,
    StateStack,
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


def create_pupper_env(render=False, from_pixels=True, skip=0, stack=1):
    # build env from pybullet config
    CONFIG_DIR = puppersim.getPupperSimPath() + "/"
    _CONFIG_FILE = os.path.join(CONFIG_DIR, "pupper_pmtg_slow.gin")
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
    env = SimpleGymWrapper(env)
    return env


class CCEncoder(uafbc.nets.Encoder):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self._dim = out_dim

    @property
    def embedding_dim(self):
        return self._dim

    def forward(self, obs_dict):
        return obs_dict["obs"]


class VisionEncoder(uafbc.nets.Encoder):
    def __init__(self, inp_shape, emb_dim=50):
        super().__init__()
        self._dim = emb_dim
        self.conv_block = uafbc.nets.cnns.BigPixelEncoder(inp_shape, emb_dim)

    def forward(self, obs_dict):
        return self.conv_block(obs_dict["obs"])
        # bs = obs_dict["obs"].shape[0]
        # return torch.randn((bs, 50)).float()

    @property
    def embedding_dim(self):
        return self._dim


def train_cont_gym_online(args):

    make_env = functools.partial(
        create_pupper_env,
        render=args.render,
        from_pixels=args.from_pixels,
        skip=args.skip,
        stack=args.stack,
    )
    train_env = ParallelActors(make_env, args.parallel_envs)
    test_env = make_env()
    """
    state = train_env.reset()["obs"]
    breakpoint()
    for _ in range(1000):
        _, rew, done, _ = train_env.step(train_env.action_space.sample())
        train_env.render()
        if done:
            train_env.reset()
        print(rew)
    """

    if args.from_pixels:
        encoder = VisionEncoder(train_env.reset()["obs"].shape, 50)
        augmenter = AugmentationSequence([DrqAug(args.batch_size)])
    else:
        encoder = CCEncoder(None, train_env.observation_space.shape[0])
        augmenter = None

    # create agent
    agent = uafbc.Agent(
        act_space_size=train_env.action_space.shape[0],
        encoder=encoder,
        actor_network_cls=uafbc.nets.mlps.ContinuousStochasticActor,
        critic_network_cls=uafbc.nets.mlps.ContinuousCritic,
        ensemble_size=args.ensemble,
        num_critics=args.critics,
        ucb_bonus=args.ucb_bonus,
        hidden_size=args.hidden_size,
        discrete=False,
        auto_rescale_targets=False,
        beta_dist=args.beta_dist,
    )

    size = 1_000_000 if not args.from_pixels else 100_000
    buffer = uafbc.replay.PrioritizedReplayBuffer(size=size)

    # run training
    uafbc.uafbc(
        agent=agent,
        train_env=train_env,
        test_env=test_env,
        buffer=buffer,
        verbosity=1,
        name=args.name,
        use_pg_update_online=True,
        actor_lr=1e-4,
        critic_lr=1e-4,
        actor_clip=10.0,
        critic_clip=10.0,
        encoder_clip=10.0,
        batch_size=args.batch_size,
        critic_updates_per_step=1,
        gamma=args.gamma,
        weighted_bellman_temp=10.0,
        weight_type="sunrise",
        use_bc_update_online=False,
        bc_warmup_steps=0,
        num_steps_offline=0,
        num_steps_online=args.steps,
        random_warmup_steps=100,
        max_episode_steps=args.max_steps,
        eval_episodes=1,
        eval_interval=args.eval_interval,
        pop=False,
        init_alpha=0.1,
        use_exploration_process=False,
        target_entropy_mul=1.0,
        alpha_lr=1e-4,
        render=args.render,
        augmenter=augmenter,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="uafbc_pupper")
    parser.add_argument("--parallel_envs", type=int, default=6)
    parser.add_argument("--ensemble", type=int, default=1)
    parser.add_argument("--critics", type=int, default=2)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--eval_interval", type=int, default=20_000)
    parser.add_argument("--max_steps", type=int, default=10_000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--steps", type=int, default=10_000_000)
    parser.add_argument("--ucb_bonus", type=float, default=0.0)
    parser.add_argument("--beta_dist", action="store_true")
    parser.add_argument("--from_pixels", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--stack", type=int, default=3)
    parser.add_argument("--skip", type=int, default=2)
    args = parser.parse_args()
    train_cont_gym_online(args)
