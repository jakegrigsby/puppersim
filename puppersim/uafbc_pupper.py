import argparse

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import gym

import pybullet
import pybullet_envs

import uafbc
from uafbc.wrappers import SimpleGymWrapper, NormActionSpace, ParallelActors
from wrappers import create_pupper_env, SqueezeRew, StateStack

class Encoder(uafbc.nets.Encoder):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self._dim = out_dim
        self.shared_fc1 = nn.Linear(in_dim, 256)
        self.shared_fc2 = nn.Linear(256, out_dim)

    @property
    def embedding_dim(self):
        return self._dim

    def forward(self, obs_dict):
        rep = F.relu(self.shared_fc1(obs_dict["obs"]))
        rep = F.relu(self.shared_fc2(rep))
        return rep


def train_cont_gym_online(args):

    def make_env(render=True):
        env = create_pupper_env(render=render, persistence=False)
        return StateStack(env, num_stack=4, skip=args.skip)

    train_env = SimpleGymWrapper(ParallelActors(make_env, args.parallel_envs))
    test_env = SimpleGymWrapper(make_env(render=False))

    # create agent
    agent = uafbc.Agent(
        act_space_size=train_env.action_space.shape[0],
        encoder=Encoder(train_env.observation_space.shape[0], 128),
        actor_network_cls=uafbc.nets.mlps.ContinuousStochasticActor,
        critic_network_cls=uafbc.nets.mlps.ContinuousCritic,
        critic_ensemble_size=6,
        actor_ensemble_size=6,
        ucb_bonus=25.,
        hidden_size=128,
        discrete=False,
        auto_rescale_targets=True,
        beta_dist=False,
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
        actor_clip=1.,
        critic_clip=1.,
        critic_lr=1e-4,
        encoder_lr=1e-4,
        batch_size=512,
        critic_updates_per_step=2,
        gamma=.99,
        weighted_bellman_temp=20.,
        weight_type="softmax",
        use_bc_update_online=False,
        bc_warmup_steps=0,
        num_steps_offline=0,
        num_steps_online=1_000_000,
        random_warmup_steps=10_000,
        max_episode_steps=10_000,
        eval_episodes=1,
        pop=True,
        init_alpha=0.1,
        use_exploration_process=False,
        target_entropy_mul=1.,
        alpha_lr=1e-4,
        render=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="uafbc_pupper")
    parser.add_argument("--parallel_envs", type=int, default=12)
    parser.add_argument("--skip", type=int, default=6)
    args = parser.parse_args()
    train_cont_gym_online(args)
