from enum import Enum

import numpy as np
import torch


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
    import ipdb; ipdb.set_trace()
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


ENTITY_ENCODING_LENGTH = 9
PLAYER_IDX = 0 * ENTITY_ENCODING_LENGTH
FLAG_IDX = 1 * ENTITY_ENCODING_LENGTH
POWERUP_IDX = 2 * ENTITY_ENCODING_LENGTH
ENEMY_IDX = 3 * ENTITY_ENCODING_LENGTH
COIN0_IDX = 4 * ENTITY_ENCODING_LENGTH
COIN1_IDX = 5 * ENTITY_ENCODING_LENGTH

IDX_ID = 0
IDX_X = 1
IDX_Y = 2
IDX_VX = 3
IDX_VY = 4
IDX_E0 = IDX_HAS_POWERUP = 5
IDX_E1 = 6
IDX_E2 = 7
IDX_E3 = 8


def fixed_size_entity_representation(state, swap_coins=None):
    """
    Compresses the list of entities into an fixed size array (MAX_ENTITIES(6)*ENTITY_ENCODING(9))
    Entity order: player, flag, powerup, enemy, coin0, coin1.
    Entity encoding: [x,y, vx,vy, E0..E3]
    :param state:
    :return: int[54] representation of the state
    """

    entities = state['entities']
    MAX_ENTITIES = 12  # maximum number of entities in the level (1*player, 1*flag, 1*powerup, 1*enemy, 2*coin)
    ENTITY_ENCODING = 6  # number of parameters by which each entity is encoded
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


def distance(er, e0, e1):
    # return math.sqrt(
    #    (er[e1 + IDX_X] - er[e0 + IDX_X])**2 +
    #    (er[e1 + IDX_Y] - er[e0 + IDX_Y])**2
    # )

    # signed x and y distance (positive: e0 left of e1; negative: e0 right of e1)
    return er[e1 + IDX_X] - er[e0 + IDX_X]


def present_objects(er):
    has_powerup = er[POWERUP_IDX + IDX_ID] != 0
    has_enemy = er[ENEMY_IDX + IDX_ID] != 0
    has_coin0 = er[COIN0_IDX + IDX_ID] != 0
    has_coin1 = er[COIN1_IDX + IDX_ID] != 0
    return has_powerup, has_enemy, has_coin0, has_coin1


def sign(x):
    if x == 0:
        return 0
    if x < 0:
        return -1
    else:
        return 1


def replace_bools(tr_entities, swap_coins):
    # for i, x in enumerate(a):
    #     if isinstance(x, bool):
    #         a[i] = int(x)
    swap_coins = int(swap_coins)

    return tr_entities, swap_coins


def state_to_extended_repr(state, coin_jump, swap_coins=None, coin0_x=None):
    er, swap_coins = fixed_size_entity_representation(state, swap_coins=swap_coins)

    if coin0_x is None:
        coin0_x = er[COIN0_IDX + IDX_X]

    has_powerup, has_enemy, has_coin0, has_coin1 = present_objects(er)

    flag_dist = distance(er, PLAYER_IDX, FLAG_IDX)
    powerup_dist = distance(er, PLAYER_IDX, POWERUP_IDX)
    enemy_dist = distance(er, PLAYER_IDX, ENEMY_IDX)
    coin0_dist = distance(er, PLAYER_IDX, COIN0_IDX) if has_coin0 else 0
    coin1_dist = distance(er, PLAYER_IDX, COIN1_IDX) if has_coin1 else 0

    towards_flag = sign(er[PLAYER_IDX + IDX_VX]) == sign(flag_dist)
    towards_powerup = sign(er[PLAYER_IDX + IDX_VX]) == sign(powerup_dist)
    towards_enemy = sign(er[PLAYER_IDX + IDX_VX]) == sign(enemy_dist)
    towards_coin0 = sign(er[PLAYER_IDX + IDX_VX]) == sign(coin0_dist) if has_coin0 else False
    towards_coin1 = sign(er[PLAYER_IDX + IDX_VX]) == sign(coin1_dist) if has_coin1 else False

    near_dist = 3.0
    near_flag = flag_dist < near_dist
    near_powerup = powerup_dist < near_dist
    near_enemy = enemy_dist < near_dist
    near_coin0 = (coin0_dist < near_dist) if has_coin0 else False
    near_coin1 = (coin1_dist < near_dist) if has_coin1 else False

    er_add = [
        coin_jump.level.reward,
        coin_jump.score,
        coin_jump.level.terminated,
        coin_jump.level.lost,

        has_powerup, has_enemy, has_coin0, has_coin1,  # indicates present objects
        flag_dist, powerup_dist, enemy_dist, coin0_dist, coin1_dist,  # object distance
        near_flag, near_powerup, near_enemy, near_coin0, near_coin1,  # object proximity

        # moving towards entity
        towards_flag, towards_powerup, towards_enemy, towards_coin0, towards_coin1,

        coin_jump.player.collisions[0],  # flag collision
        coin_jump.player.collisions[1],  # powerup collision
        coin_jump.player.collisions[2],  # enemy collision
        coin_jump.player.collisions[3] and coin_jump.player.collision_x == coin0_x,  # coin0 collision
        coin_jump.player.collisions[3] and coin_jump.player.collision_x != coin0_x  # coin1 collision
    ]

    tr_ext = [*er, *er_add]

    return tr_ext, swap_coins, coin0_x
