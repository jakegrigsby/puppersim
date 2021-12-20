import argparse
import tqdm
import random
import pickle

import numpy as np
import gym
import torch
import gin

import super_sac
from super_sac_pupper import create_pupper_env, CCEncoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expert_policy_file", type=str, required=True)
    parser.add_argument("--max_steps", type=int, default=5_000)
    parser.add_argument("--save_experience", type=str, default=None)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--config", type=str, default=None, required=True)
    parser.add_argument("--num_rollouts", type=int, default=10)
    args = parser.parse_args()
    gin.parse_config_file(args.config)

    env = create_pupper_env()
    encoder = CCEncoder(None, env.observation_space.shape[0])
    agent = super_sac.Agent(
        act_space_size=env.action_space.shape[0],
        encoder=encoder,
    )
    agent.load(args.expert_policy_file)
    agent.to(super_sac.device)
    agent.eval()

    returns = []
    reward_histories = []
    actions, rewards, dones = [], [], []
    state_keys = env.reset().keys()
    states = {k:[] for k in state_keys}
    next_states = {k:[] for k in state_keys}
    ep_lengths = []
    for i in tqdm.tqdm(range(args.num_rollouts)):
        obs = env.reset()
        done = False
        totalr = 0.0
        steps = 0
        while not done and steps < args.max_steps:
            with torch.no_grad():
                    action = agent.forward(obs)
            next_obs, rew, done, _ = env.step(action)

            for key, val in obs.items():
                states[key].append(val)
            for key, val in next_obs.items():
                next_states[key].append(val)
            actions.append(action)
            rewards.append(rew)
            dones.append(done)

            obs = next_obs
            if args.render:
                env.render()
            totalr += rew
            steps += 1

        ep_lengths.append(steps)
        returns.append(totalr)

    print("returns", returns)
    print("ep_lengths", ep_lengths)
    print("mean return", np.mean(returns))
    print("std of return", np.std(returns))

    if args.save_experience is not None:
        with open(args.save_experience, "wb") as f:
            pickle.dump(
                dict(
                    states=states,
                    actions=actions,
                    rewards=rewards,
                    next_states=next_states,
                    dones=dones,
                ),
                f,
            )


if __name__ == "__main__":
    main()
