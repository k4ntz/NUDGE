import random
import numpy as np
import torch


def make_deterministic(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    print(f"Set all environment deterministic to seed {seed}")


def initialize_game(env, args):
    """initialize game"""
    if args.m == 'threefish' or args.m == 'loot':
        reward, state, done = env.observe()
    elif args.m == 'getout':
        # return the whole getout information
        state = env.reset()
    elif args.m == "atari":
        state = env.reset()
        state = env.objects
    return state


def env_step(action, env, args):
    """take step of each game"""
    if args.m == 'getout':
        state, reward, done, _, info = env.step(action)
        # perhaps need some reward shaping
        if args.rules == 'ppo_simple_policy':
            # simpler policy
            if action in [3]:
                reward += -0.2
    elif args.m == 'threefish' or args.m == 'loot':
        env.act(action)
        reward, state, done = env.observe()
        reward = reward[0]
    elif args.m == 'atari':
        pix_state, reward, done, _, _ = env.step(action)
        state = env.objects
        # reward = reward[0]
    return reward, state, done
