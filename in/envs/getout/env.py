from enum import Enum
from typing import Sequence

from nudge.env import NudgeBaseEnv
import numpy as np
import gymnasium
from gymnasium.envs.registration import register
from nudge.utils import simulate_prob
import torch


class NudgeEnv(NudgeBaseEnv):
    name = "getout"
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
        register(id="getout",
                 entry_point="env_src.getout.getout.getout:Getout")
        self.env = gymnasium.make("getout")  # FIXME

    def reset(self):
        state = self.env.reset()
        return self.convert_state(state)

    def step(self, action, is_mapped: bool = False):
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

    def close(self):
        self.env.close()


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


ENTITY_ENCODING_LENGTH = 9
COIN0_IDX = 4 * ENTITY_ENCODING_LENGTH
COIN1_IDX = 5 * ENTITY_ENCODING_LENGTH
IDX_X = 1


def fixed_size_entity_representation(state, swap_coins=None):
    """
    Compresses the list of entities into an fixed size array (MAX_ENTITIES(6)*ENTITY_ENCODING(9))
    Entity order: player, flag, powerup, enemy, coin0, coin1.
    Entity encoding: [x,y, vx,vy, E0..E3]
    :param state:
    :return: int[54] representation of the state
    """

    entities = state['entities']
    MAX_ENTITIES = 6  # maximum number of entities in the level (1*player, 1*flag, 1*powerup, 1*enemy, 2*coin)
    ENTITY_ENCODING = 9  # number of parameters by which each entity is encoded
    tr_entities = [0] * ENTITY_ENCODING * MAX_ENTITIES

    # we assume that player,flag,powerup and enemy occur at max once and coins at max twice
    coin_count = 0
    # IDs: PLAYER = 1
    #      FLAG = 2
    #      COIN = 3
    #      POWERUP = 4
    #      GROUND_ENEMY = 5
    # Position encoding: player, flag, powerup, enemy, coin0, coin1
    for entity in entities:
        id = entity[0]
        if id == 3:
            id = 5 + coin_count
            coin_count += 1
        elif id > 3:
            id -= 1
        id -= 1

        start_pos = id * ENTITY_ENCODING
        tr_entities[start_pos:start_pos + ENTITY_ENCODING] = entity

    if swap_coins is None:
        swap_coins = coin_count == 2 and tr_entities[COIN1_IDX + IDX_X] < tr_entities[COIN0_IDX + IDX_X]

    if swap_coins:
        # swap coins to make coin0 to be left most
        temp_coin = tr_entities[COIN0_IDX:COIN0_IDX + ENTITY_ENCODING_LENGTH]
        tr_entities[COIN0_IDX:COIN0_IDX + ENTITY_ENCODING_LENGTH] = tr_entities[
                                                                    COIN1_IDX:COIN1_IDX + ENTITY_ENCODING_LENGTH]
        tr_entities[COIN1_IDX:COIN1_IDX + ENTITY_ENCODING_LENGTH] = temp_coin

    return tr_entities, swap_coins


def replace_bools(tr_entities, swap_coins):
    # for i, x in enumerate(a):
    #     if isinstance(x, bool):
    #         a[i] = int(x)
    swap_coins = int(swap_coins)

    return tr_entities, swap_coins
