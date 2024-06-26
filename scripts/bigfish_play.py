import gym3
import argparse
import numpy as np
import sys
sys.path.insert(0,'../')
from nudge.utils import make_deterministic
from nudge.environments.procgen.procgen import ProcgenGym3Env
from nsfr.utils.common import get_predictions
from nsfr.common import get_nsfr_model
from envs.threefish.state_extraction import extract_logic_state as extract_logic_state_threefish


def explaining_to_action(explaining):
    """map explaining to action"""
    if 'up' in explaining:
        return np.array([5])
    elif 'down' in explaining:
        return np.array([3])
    elif 'left' in explaining:
        return np.array([1])
    elif 'right' in explaining:
        return np.array([7])
    elif 'idle' in explaining:
        return np.array([4])


def run():
    device = "cuda:0"

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--seed", help="Seed for pytorch + env", default=0,
                        required=False, action="store", dest="seed", type=int)
    parser.add_argument("-m", "--mode", help="the game mode you want to play with",
                        required=False, action="store", dest="m", default='threefish',
                        choices=['threefish'])
    parser.add_argument("-alg", "--algorithm", help="algorithm that to use",
                        action="store", dest="alg", required=False, default='logic',
                        choices=['logic'])
    parser.add_argument("-r", "--rules", dest="rules", default='threefish_human_assisted',
                        required=False, choices=['threefish_human_assisted','threefishcolor'])
    parser.add_argument("-env", "--environment", help="environment of game to use",
                        required=False, action="store", dest="env", default='threefish',
                        choices=['threefish', 'threefishcolor'])
    args = parser.parse_args()
    make_deterministic(args.seed)

    env = ProcgenGym3Env(num=1, env_name=args.env, render_mode="rgb_array")
    env = gym3.ViewerWrapper(env, info_key="rgb")

    nsfr = get_nsfr_model(args.env, device=device)

    NB_DONE = 0
    TO_SUCCEED = 1
    # [
    #     ("LEFT",),
    #     ("DOWN",),
    #     (),
    #     ("UP",),
    #     ("RIGHT",),
    # ]
    # action_space = [1, 3, 4, 5, 7]

    rew, obs, done = env.observe()
    extracted_state = extract_logic_state_threefish(obs, args)

    last_explaining = ""
    while NB_DONE < TO_SUCCEED:
        explaining = get_predictions(extracted_state, nsfr)
        action = explaining_to_action(explaining)
        env.act(action)

        rew, obs, done = env.observe()

        if last_explaining != explaining:
            print(explaining)
            last_explaining = explaining

        extracted_state = extract_logic_state_threefish(obs, args)
        if done:
            print("--------------------------new game--------------------------")
        # print(f"reward : {rew}")


if __name__ == '__main__':
    run()
