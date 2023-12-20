from enum import Enum
from typing import Sequence

from agents.utils_getout import fixed_size_entity_representation, replace_bools
from nudge.env import NudgeBaseEnv
import numpy as np
import gymnasium as gym
from nudge.utils import simulate_prob
import torch


class NudgeEnv(NudgeBaseEnv):
    pred2action = {
        'stay': 0,
        'idle': 0,
        'left': 1,
        'right': 2,
        'jump': 3,
    }
    pred_names: Sequence

    def __init__(self, mode: str, plusplus=False, noise=False):
        super().__init__(mode)
        self.plusplus = plusplus
        self.noise = noise
        self.env = gym.make("getout", generator_args={"spawn_all_entities": False})  # FIXME

    def reset(self):
        state = self.env.reset()
        return self.convert_state(state)

    def step(self, action):
        state, reward, done, _, _ = self.env.step(action)
        return self.convert_state(state), reward, done

    def extract_logic_state(self, raw_state):
        n_features = 6
        if self.plusplus:
            n_objects = 8
        else:
            n_objects = 4

        representation = raw_state.level.get_representation()
        logic_state = np.zeros((n_objects, n_features))
        for entity in representation["entities"]:
            if entity[0].name == 'PLAYER':
                logic_state[0][0] = 1
                logic_state[0][-2:] = entity[1:3]
            elif entity[0].name == 'KEY':
                logic_state[1][1] = 1
                logic_state[1][-2:] = entity[1:3]
            elif entity[0].name == 'DOOR':
                logic_state[2][2] = 1
                logic_state[2][-2:] = entity[1:3]
            elif entity[0].name == 'GROUND_ENEMY':
                logic_state[3][3] = 1
                logic_state[3][-2:] = entity[1:3]
            elif entity[0].name == 'GROUND_ENEMY2':
                logic_state[4][3] = 1
                logic_state[4][-2:] = entity[1:3]
            elif entity[0].name == 'GROUND_ENEMY3':
                logic_state[5][3] = 1
                logic_state[5][-2:] = entity[1:3]
            elif entity[0].name == 'BUZZSAW1':
                logic_state[6][3] = 1
                logic_state[6][-2:] = entity[1:3]
            elif entity[0].name == 'BUZZSAW2':
                logic_state[7][3] = 1
                logic_state[7][-2:] = entity[1:3]

        if self.noise:
            if sum(logic_state[:, 1]) == 0:
                key_picked = True
            else:
                key_picked = False
            logic_state = simulate_prob(logic_state, n_objects, key_picked)

        return logic_state

    def extract_neural_state(self, raw_state):
        model_input = sample_to_model_input((extract_state(raw_state), []))
        model_input = collate([model_input])
        state = model_input['state']
        state = torch.cat([state['base'], state['entities']], dim=1)
        return state


def for_each_tensor(o, fn):
    if isinstance(o, torch.Tensor):
        return fn(o)
    if isinstance(o, list):
        for i, e in enumerate(o):
            o[i] = for_each_tensor(e, fn)
        return o
    if isinstance(o, dict):
        for k, v in o.items():
            o[k] = for_each_tensor(v, fn)
        return o
    raise ValueError("unexpected object type")


def collate(samples, to_cuda=True, double_to_float=True):
    samples = torch.utils.data._utils.collate.default_collate(samples)
    if double_to_float:
        samples = for_each_tensor(samples, lambda tensor: tensor.float() if tensor.dtype == torch.float64 else tensor)
    if to_cuda:
        samples = for_each_tensor(samples, lambda tensor: tensor.cuda())
    return samples


def extract_state(coin_jump):
    repr = coin_jump.level.get_representation()
    repr["reward"] = coin_jump.level.reward
    repr["score"] = coin_jump.score

    for entity in repr["entities"]:
        # replace all enums (e.g. EntityID) with int values
        for i, v in enumerate(entity):
            if isinstance(v, Enum):
                entity[i] = v.value
            if isinstance(v, bool):
                entity[i] = int(v)

    return repr


def sample_to_model_input(sample, no_dict=False, include_score=False):
    """
    :param sample:  tuple: (representation (use extract_state), explicit_action (use unify_coin_jump_actions))
    :return: {state: {base:[...], entities:[...]}, action:0..9}
    """
    state = sample[0]
    action = sample[1]

    # tr_entities = replace_bools(fixed_size_entity_representation(state))
    tr_entity, swap_coins = fixed_size_entity_representation(state)
    tr_entities, swap_coins = replace_bools(tr_entity, swap_coins)
    if no_dict:
        # ignores the action and returns a single [60] array
        return [
            0,
            0,
            state['level']['reward_key'],
            state['level']['reward_powerup'],
            state['level']['reward_enemy'],
            state['score'] if include_score else 0,
            *tr_entities
        ]

    tr_state = {
        'base': np.array([
            0,
            0,
            state['level']['reward_key'],
            state['level']['reward_powerup'],
            state['level']['reward_enemy'],
            state['score'] if include_score else 0,
        ]),
        'entities': np.array(tr_entities)
    }

    return {
        "state": tr_state,
        "action": action
    }
