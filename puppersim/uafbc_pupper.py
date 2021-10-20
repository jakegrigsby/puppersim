import argparse
import os
import gym
import puppersim
import functools
import gin
from pybullet_envs.minitaur.envs_v2 import env_loader
import puppersim.data as pd

from uafbc.nets import distributions, weight_init

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
)
from deep_control.envs import PersistenceAwareWrapper
from wrappers import StateStack


def create_pupper_env(render=False, k=1):
    CONFIG_DIR = puppersim.getPupperSimPath() + "/"
    _CONFIG_FILE = os.path.join(CONFIG_DIR, "pupper_pmtg_slow.gin")
    gin.bind_parameter("scene_base.SceneBase.data_root", pd.getDataPath() + "/")
    gin.parse_config_file(_CONFIG_FILE)
    if render:
        gin.bind_parameter("SimulationParameters.enable_rendering", True)
    env = env_loader.load()
    env = ScaleReward(env, scale=100.0)
    env = PersistenceAwareWrapper(env, k=k, return_history=False)
    env = NormActionSpace(env)
    return env


class ScaleAction(gym.ActionWrapper):
    def action(self, act):
        return 1. * act


class Encoder(uafbc.nets.Encoder):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self._dim = out_dim
        #self.shared_fc1 = nn.Linear(in_dim, 256)
        #self.shared_fc2 = nn.Linear(256, out_dim)

    @property
    def embedding_dim(self):
        return self._dim

    def forward(self, obs_dict):
        return obs_dict["obs"]
        #rep = F.relu(self.shared_fc1(obs_dict["obs"]))
        #rep = F.relu(self.shared_fc2(rep))
        #return rep


class RandomActor(torch.nn.Module):
    def __init__(
        self,
        state_size,
        action_size,
        log_std_low=-10.0,
        log_std_high=2.0,
        hidden_size=256,
        dist_impl="pyd",
    ):
        super().__init__()
        assert dist_impl in ["pyd", "beta"]
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 2 * action_size)
        self.log_std_low = log_std_low
        self.log_std_high = log_std_high
        self.apply(weight_init)
        self.dist_impl = dist_impl

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        out = self.fc3(x)
        mu, log_std = out.chunk(2, dim=-1)
        if self.dist_impl == "pyd":
            log_std = torch.tanh(log_std)
            log_std = self.log_std_low + 0.5 * (
                self.log_std_high - self.log_std_low
            ) * (log_std + 1)
            std = log_std.exp()

            # CHANGED
            mu = torch.randn_like(mu)
            std = torch.randn_like(std).clamp(1e-3, 0.1)

            dist = distributions.SquashedNormal(mu, std)
        elif self.dist_impl == "beta":
            out = 1.0 + F.softplus(out)
            alpha, beta = out.chunk(2, dim=-1)
            dist = distributions.BetaDist(alpha, beta)
        return dist


def train_cont_gym_online(args):
    def make_env(render=False, k=1):
        env = ScaleAction(create_pupper_env(render=render, k=k))
        return StateStack(env, num_stack=4, skip=args.skip)
        return env

    make_full_env = functools.partial(make_env, render=False, k=args.k)
    train_env = SimpleGymWrapper(ParallelActors(make_full_env, args.parallel_envs))
    test_env = SimpleGymWrapper(make_full_env())

    # create agent
    agent = uafbc.Agent(
        act_space_size=train_env.action_space.shape[0],
        encoder=Encoder(None, train_env.observation_space.shape[0]),
        actor_network_cls=uafbc.nets.mlps.ContinuousStochasticActor,
        # actor_network_cls=RandomActor,
        critic_network_cls=uafbc.nets.mlps.ContinuousCritic,
        ensemble_size=args.ensemble,
        num_critics=args.critics,
        ucb_bonus=args.ucb_bonus,
        hidden_size=args.hidden_size,
        discrete=False,
        auto_rescale_targets=False,
        beta_dist=args.beta_dist,
    )

    buffer = uafbc.replay.PrioritizedReplayBuffer(size=1_000_000)

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
        actor_clip=10.,
        critic_clip=10.,
        batch_size=args.batch_size,
        critic_updates_per_step=1,
        gamma=args.gamma,
        weighted_bellman_temp=10.0,
        weight_type="sunrise",
        use_bc_update_online=False,
        bc_warmup_steps=0,
        num_steps_offline=0,
        num_steps_online=args.steps,
        random_warmup_steps=1_000,
        max_episode_steps=args.max_steps,
        eval_episodes=1,
        eval_interval=args.eval_interval,
        pop=False,
        init_alpha=0.1,
        use_exploration_process=False,
        target_entropy_mul=1.0,
        alpha_lr=1e-4,
        # render=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="uafbc_pupper")
    parser.add_argument("--parallel_envs", type=int, default=6)
    parser.add_argument("--skip", type=int, default=6)
    parser.add_argument("--ensemble", type=int, default=1)
    parser.add_argument("--critics", type=int, default=2)
    parser.add_argument("--gamma", type=float, default=.995)
    parser.add_argument("--hidden_size", type=int, default=1024)
    parser.add_argument("--eval_interval", type=int, default=10_000)
    parser.add_argument("--max_steps", type=int, default=10_000)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=10_000_000)
    parser.add_argument("--ucb_bonus", type=float, default=0.0)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--beta_dist", action="store_true")
    args = parser.parse_args()
    train_cont_gym_online(args)
