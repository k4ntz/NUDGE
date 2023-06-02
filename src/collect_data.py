import random
import json
import torch
import os
import pathlib
import gym3

from argparse import ArgumentParser
from environments.getout.getout.imageviewer import ImageViewer
from environments.getout.getout.getout.getout import Getout
from environments.getout.getout.getout.paramLevelGenerator import ParameterizedLevelGenerator
from agents.utils_getout import extract_state, sample_to_model_input, collate
from agents.neural_agent import ActorCritic
from agents.utils_loot import extract_neural_state_loot, simplify_action_loot, extract_logic_state_loot
from agents.utils_threefish import extract_logic_state_threefish, extract_neural_state_threefish
from tqdm import tqdm
from nsfr.utils import extract_for_cgen_explaining

from environments.procgen.procgen import ProcgenGym3Env

KEY_r = 114
device = torch.device('cuda:0')


class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.logic_states = []
        self.neural_states = []
        self.action_probs = []
        self.logprobs = []
        self.reward = []
        self.terminated = []
        self.predictions = []

    def clear(self):
        del self.actions[:]
        del self.logic_states[:]
        del self.neural_states[:]
        del self.action_probs[:]
        del self.logprobs[:]
        del self.reward[:]
        del self.terminated[:]
        del self.predictions[:]

    def save_data(self, args):
        dict = {'actions': self.actions, 'logic_states': self.logic_states, 'neural_states': self.neural_states,
                'action_probs': self.action_probs, 'logprobs': self.logprobs, 'reward': self.reward,
                'terminated': self.terminated, 'predictions': self.predictions}

        current_path = os.path.dirname(__file__)
        dataset = args.m + '.json'
        path = os.path.join(current_path, 'bs_data', dataset)
        with open(path, 'w') as f:
            json.dump(dict, f)
        print('data collected')


def setup_image_viewer(getout):
    viewer = ImageViewer(
        "getout1",
        getout.camera.height,
        getout.camera.width,
        monitor_keyboard=True,
        # relevant_keys=set('W','A','S','D','SPACE')
    )
    return viewer


def create_getout_instance(seed=None):
    seed = random.randint(0, 100000000) if seed is None else seed

    # level_generator = DummyGenerator()
    coin_jump = Getout(start_on_first_action=False)
    level_generator = ParameterizedLevelGenerator()

    level_generator.generate(coin_jump, seed=seed)
    coin_jump.render()

    return coin_jump


def parse_args():
    parser = ArgumentParser("Loads a model and lets it play getout")
    parser.add_argument("-m", "--mode", help="the game mode you want to play with",
                        required=True, action="store", dest="m", default='getout',
                        choices=['getout', 'threefish', 'loot'])
    parser.add_argument("-env", "--environment", help="environment of game to use",
                        required=True, action="store", dest="env", default='GetoutEnv-v1',
                        choices=['getout', 'threefish', 'loot', 'lootcolor'])
    parser.add_argument("-mo", "--model_file", dest="model_file", default=None)
    parser.add_argument("-s", "--seed", dest="seed", default=0, type=int)
    arg = ['-m', 'loot', '-env', 'elootc1']
    args = parser.parse_args(arg)

    if args.model_file is None:
        # read filename from stdin
        current_path = os.path.dirname(__file__)
        model_name = input('Enter file name: ')
        model_file = os.path.join(current_path, 'models', args.m, 'ppo', model_name)
        # model_file = f"../src/ppo_getout_model/{input('Enter file name: ')}"

    else:
        model_file = pathlib.Path(args.model_file)

    return args, model_file


def load_model(model_path, args, set_eval=True):
    with open(model_path, "rb") as f:
        model = ActorCritic(args).to(device)
        model.load_state_dict(state_dict=torch.load(f))
    if isinstance(model, ActorCritic):
        model = model.actor
        model.as_dict = True

    if set_eval:
        model = model.eval()

    return model


def main():
    args, model_file = parse_args()

    model = load_model(model_file, args)

    seed = random.seed() if args.seed is None else int(args.seed)

    buffer = RolloutBuffer()

    # collect data
    max_states = 30000
    save_frequence = 5
    step = 0
    collected_states = 0
    if args.m == 'getout':
        coin_jump = create_getout_instance(seed=seed)
        # viewer = setup_image_viewer(coin_jump)

        # frame rate limiting
        fps = 10
        for i in tqdm(range(max_states)):
            # step game
            step += 1

            if not coin_jump.level.terminated:
                model_input = sample_to_model_input((extract_state(coin_jump), []))
                model_input = collate([model_input])
                state = model_input['state']
                neural_state = torch.cat([state['base'], state['entities']], dim=1)
                prediction = model(neural_state)
                # 1 left 2 right 3 up
                action = torch.argmax(prediction).cpu().item() + 1

                logic_state = extract_for_cgen_explaining(coin_jump)
                if step % save_frequence == 0:
                    collected_states += 1
                    buffer.logic_states.append(logic_state.detach().tolist())
                    buffer.actions.append(torch.argmax(prediction.detach()).tolist())
                    buffer.action_probs.append(prediction.detach().tolist())
                    buffer.neural_states.append(neural_state.tolist())
            else:
                coin_jump = create_getout_instance(seed=seed)
                action = 0

            reward = coin_jump.step(action)

        buffer.save_data(args)
    elif args.m == 'loot':

        env = ProcgenGym3Env(num=1, env_name=args.env, render_mode="rgb_array")
        reward, obs, done = env.observe()
        for i in tqdm(range(max_states)):
            # step game
            step += 1
            neural_state = extract_neural_state_loot(obs, args)
            logic_state = extract_logic_state_loot(obs, args)
            logic_state =logic_state.squeeze(0)
            predictions = model(neural_state)
            action = torch.argmax(predictions)
            action = simplify_action_loot(action)
            env.act(action)
            rew, obs, done = env.observe()
            if step % save_frequence == 0:
                collected_states += 1
                buffer.logic_states.append(logic_state.detach().tolist())
                buffer.actions.append(torch.argmax(predictions.detach()).tolist())
                buffer.action_probs.append(predictions.detach().tolist())
                buffer.neural_states.append(neural_state.tolist())
        buffer.save_data(args)


    elif args.m == 'threefish':
        env = ProcgenGym3Env(num=1, env_name=args.env, render_mode="rgb_array")
        reward, obs, done = env.observe()
        for i in tqdm(range(max_states)):
            # step game
            step += 1
            neural_state = extract_neural_state_threefish(obs, args)
            logic_state = extract_logic_state_threefish(obs, args)
            predictions = model(neural_state)
            action = torch.argmax(predictions)
            action = simplify_action_loot(action)
            env.act(action)
            rew, obs, done = env.observe()
            if step % save_frequence == 0:
                collected_states += 1
                buffer.logic_states.append(logic_state.detach().tolist())
                buffer.actions.append(torch.argmax(predictions.detach()).tolist())
                buffer.action_probs.append(predictions.detach().tolist())
                buffer.neural_states.append(neural_state.tolist())
        buffer.save_data(args)


if __name__ == "__main__":
    main()
