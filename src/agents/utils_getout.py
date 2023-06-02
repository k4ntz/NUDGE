from enum import Enum
from typing import List, Final
import numpy as np
import torch


def extract_logic_state_getout(coin_jump, args, noise=False):
    if args.env == 'getoutplus':
        num_of_feature = 6
        num_of_object = 8
        representation = coin_jump.level.get_representation()
        # import ipdb; ipdb.set_trace()
        extracted_states = np.zeros((num_of_object, num_of_feature))
        for entity in representation["entities"]:
            if entity[0].name == 'PLAYER':
                extracted_states[0][0] = 1
                extracted_states[0][-2:] = entity[1:3]
                # 27 is the width of map, this is normalization
                # extracted_states[0][-2:] /= 27
            elif entity[0].name == 'KEY':
                extracted_states[1][1] = 1
                extracted_states[1][-2:] = entity[1:3]
                # extracted_states[1][-2:] /= 27
            elif entity[0].name == 'DOOR':
                extracted_states[2][2] = 1
                extracted_states[2][-2:] = entity[1:3]
                # extracted_states[2][-2:] /= 27
            elif entity[0].name == 'GROUND_ENEMY':
                extracted_states[3][3] = 1
                extracted_states[3][-2:] = entity[1:3]
                # extracted_states[3][-2:] /= 27
            elif entity[0].name == 'GROUND_ENEMY2':
                extracted_states[4][3] = 1
                extracted_states[4][-2:] = entity[1:3]
                # extracted_states[3][-2:] /= 27
            elif entity[0].name == 'GROUND_ENEMY3':
                extracted_states[5][3] = 1
                extracted_states[5][-2:] = entity[1:3]
            elif entity[0].name == 'BUZZSAW1':
                extracted_states[6][3] = 1
                extracted_states[6][-2:] = entity[1:3]
            elif entity[0].name == 'BUZZSAW2':
                extracted_states[7][3] = 1
                extracted_states[7][-2:] = entity[1:3]
    else:
        """
        extract state to metric
        input: coin_jump instance
        output: extracted_state to be explained
        set noise to True to add noise
    
        x:  agent, key, door, enemy, position_X, position_Y
        y:  obj1(agent), obj2(key), obj3(door)，obj4(enemy)
    
        To be changed when using object-detection tech
        """
        num_of_feature = 6
        num_of_object = 4
        representation = coin_jump.level.get_representation()
        extracted_states = np.zeros((num_of_object, num_of_feature))
        for entity in representation["entities"]:
            if entity[0].name == 'PLAYER':
                extracted_states[0][0] = 1
                extracted_states[0][-2:] = entity[1:3]
                # 27 is the width of map, this is normalization
                # extracted_states[0][-2:] /= 27
            elif entity[0].name == 'KEY':
                extracted_states[1][1] = 1
                extracted_states[1][-2:] = entity[1:3]
                # extracted_states[1][-2:] /= 27
            elif entity[0].name == 'DOOR':
                extracted_states[2][2] = 1
                extracted_states[2][-2:] = entity[1:3]
                # extracted_states[2][-2:] /= 27
            elif entity[0].name == 'GROUND_ENEMY':
                extracted_states[3][3] = 1
                extracted_states[3][-2:] = entity[1:3]
                # extracted_states[3][-2:] /= 27

    if sum(extracted_states[:, 1]) == 0:
        key_picked = True
    else:
        key_picked = False

    def simulate_prob(extracted_states, num_of_objs, key_picked):
        for i, obj in enumerate(extracted_states):
            obj = add_noise(obj, i, num_of_objs)
            extracted_states[i] = obj
        if key_picked:
            extracted_states[:, 1] = 0
        return extracted_states

    def add_noise(obj, index_obj, num_of_objs):
        mean = torch.tensor(0.2)
        std = torch.tensor(0.05)
        noise = torch.abs(torch.normal(mean=mean, std=std)).item()
        rand_noises = torch.randint(1, 5, (num_of_objs - 1,)).tolist()
        rand_noises = [i * noise / sum(rand_noises) for i in rand_noises]
        rand_noises.insert(index_obj, 1 - noise)

        for i, noise in enumerate(rand_noises):
            obj[i] = rand_noises[i]
        return obj

    if noise:
        extracted_states = simulate_prob(extracted_states, num_of_object, key_picked)
    states = torch.tensor(np.array(extracted_states), dtype=torch.float32, device="cuda:0").unsqueeze(0)
    return states


def extract_neural_state_getout(getout, args):
    model_input = sample_to_model_input((extract_state(getout), []))
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


def preds_to_action_getout(action, prednames):
    """
    map explaining to action
    0:jump
    1:left_go_get_key
    2:right_go_get_key
    3:left_go_to_door
    4:right_go_to_door

    CJA_MOVE_LEFT: Final[int] = 1
    CJA_MOVE_RIGHT: Final[int] = 2
    CJA_MOVE_UP: Final[int] = 3
    """
    if 'jump' in prednames[action]:
        return 3
    elif 'left' in prednames[action]:
        return 1
    elif 'right' in prednames[action]:
        return 2
    elif 'stay' in prednames[action] or 'idle' in prednames[action]:
        return 0


def action_map_getout(prediction, args, prednames=None):
    """map model action to game action"""
    if args.alg == 'ppo':
        # simplified action--- only left right up
        # action = coin_jump_actions_from_unified(torch.argmax(predictions).cpu().item() + 1)
        action = prediction + 1
    elif args.alg == 'logic':
        action = preds_to_action_getout(prediction, prednames)
    return action
