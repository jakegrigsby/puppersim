import argparse
import os

from torch import nn
import torch.nn.functional as F
import gym
import gin

import super_sac
from super_sac.wrappers import SimpleGymWrapper, ParallelActors, NormActionSpace

from reacher_env import ReacherEnv


class IdentityEncoder(super_sac.nets.Encoder):
    def __init__(self, dim):
        super().__init__()
        self._dim = dim

    @property
    def embedding_dim(self):
        return self._dim

    def forward(self, obs_dict):
        return obs_dict["obs"]


def make_env():
    env = ReacherEnv()
    env = NormActionSpace(env)
    return env


def train_reacher(args):
    gin.parse_config_file(args.config)
    train_env = SimpleGymWrapper(ParallelActors(make_env, 1))
    test_env = SimpleGymWrapper(ParallelActors(make_env, 1))
    state_space = train_env.observation_space
    encoder = IdentityEncoder(state_space.shape[0])

    # create agent
    agent = super_sac.Agent(
        act_space_size=train_env.action_space.shape[0],
        encoder=encoder,
    )

    buffer = super_sac.replay.ReplayBuffer(size=2_000_000)

    # run training
    super_sac.super_sac(
        agent=agent,
        train_env=train_env,
        test_env=test_env,
        buffer=buffer,
        name=args.name,
        logging_method=args.logging,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--name", type=str, default="reacher_rl")
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument(
        "--logging", type=str, choices=["tensorboard", "wandb"], default="tensorboard"
    )
    args = parser.parse_args()
    for _ in range(args.trials):
        train_reacher(args)
