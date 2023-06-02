import numpy as np
import torch

device = torch.device('cuda:0')


def extract_logic_state_loot(state, args):
    """
    [X,Y]
    [agent,key_b,door_b,key_g,door_g,key_r,door_r]
    """
    states = torch.from_numpy(state['positions']).squeeze()
    if args.env == 'lootplus':
        # input shape: [X,Y]* [agent,key_b,door_b,key_g,door_g,key_r,door_r]
        # output shape:[agent, key, door, blue, green, red ,got_key, X, Y]
        extracted_state = torch.tensor([
            [1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0, 1, 0, 0, 0]], dtype=torch.float64)
        extracted_state[:, -2:] = states[:]
        for i, state in enumerate(extracted_state):
            if state[-1] == 0:
                extracted_state[i] = torch.zeros((1, 9))
            elif i in [2, 4, 6] and state[-1] != 0 and extracted_state[i - 1][1] == 0:
                extracted_state[i][-3] = 1
    elif args.env == "loot":
        # input shape: [X,Y]* [agent,key_b,door_b,key_g,door_g]
        # output shape:[agent, key, door, blue, red ,got_key, X, Y]
        extracted_state = torch.tensor([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 1, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 0, 0, 0],
            [0, 1, 0, 0, 1, 0, 0, 0],
            [0, 0, 1, 0, 1, 0, 0, 0]], dtype=torch.float64)
        states = states[0:5]
        extracted_state[:, -2:] = states[:]
        for i, state in enumerate(extracted_state):
            # 0 mean object does not exist
            if state[-1] == 0:
                # then set to all attributes 0
                extracted_state[i] = torch.zeros((1, 8))
            # if key = 0 but door !=0, means key of this door has picked
            elif i in [2, 4] and state[-1] != 0 and extracted_state[i - 1][1] == 0:
                extracted_state[i][-3] = 1
    elif args.env == "lootcolor":
        # input shape: [X,Y]* [agent,key_b,door_b,key_g,door_g]
        # output shape:[agent, key, door, green, brown ,got_key, X, Y]
        extracted_state = torch.tensor([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 1, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 0, 0, 0],
            [0, 1, 0, 0, 1, 0, 0, 0],
            [0, 0, 1, 0, 1, 0, 0, 0]], dtype=torch.float64)
        states = states[0:5]
        extracted_state[:, -2:] = states[:]
        for i, state in enumerate(extracted_state):
            # 0 mean object does not exist
            if state[-1] == 0:
                # then set to all attributes 0
                extracted_state[i] = torch.zeros((1, 8))
            # if key = 0 but door !=0, means key of this door has picked
            elif i in [2, 4] and state[-1] != 0 and extracted_state[i - 1][1] == 0:
                extracted_state[i][-3] = 1
    extracted_state = extracted_state.unsqueeze(0)
    return extracted_state.to(device)


def extract_neural_state_loot(state, args):
    state = state['positions']

    if args.env == 'lootplus':
        raw_state = np.array([[0, 0, 0, 0],
                              [0, 0, 1, 1],
                              [0, 0, 2, 1],
                              [0, 0, 1, 2],
                              [0, 0, 2, 2],
                              [0, 0, 1, 3],
                              [0, 0, 2, 3]], dtype=np.float32)
        raw_state[:, 0:2] = state[0][:]

    elif args.env == 'loot':

        raw_state = np.array([[0, 0, 0, 0],
                              [0, 0, 1, 1],
                              [0, 0, 2, 1],
                              [0, 0, 1, 2],
                              [0, 0, 2, 2],
                              [0, 0, 0, 0],
                              [0, 0, 0, 0]], dtype=np.float32)
        raw_state[:, 0:2] = state[0][:]

    elif args.env == 'lootcolor':
        raw_state = np.array([[0, 0, 0, 0],
                              [0, 0, 1, 10],
                              [0, 0, 2, 10],
                              [0, 0, 1, 20],
                              [0, 0, 2, 20],
                              [0, 0, 0, 0],
                              [0, 0, 0, 0]], dtype=np.float32)
        raw_state[:, 0:2] = state[0][:]

    state = raw_state.reshape(-1)
    state = state.tolist()
    return torch.tensor(state).to(device)


def simplify_action_loot(action):
    """simplify 9 actions to 5 actions
    """
    #          left,down,idle,up,right
    # model_ouput  [0, 1, 2, 3, 4]
    action_space = [1, 3, 4, 5, 7]
    action = action_space[action]
    return np.array([action])


def preds_to_action_loot(action, prednames):
    """
    map explaining to action
    action_space = [1, 3, 4, 5, 7]
    """
    if 'up' in prednames[action]:
        return np.array([5])
    elif 'down' in prednames[action]:
        return np.array([3])
    elif 'left' in prednames[action]:
        return np.array([1])
    elif 'right' in prednames[action]:
        return np.array([7])
    elif 'idle' in prednames[action]:
        return np.array([4])


def action_map_loot(prediction, args, prednames=None):
    """map model action to game action"""
    if args.alg == 'ppo':
        action = simplify_action_loot(prediction)
    elif args.alg == 'logic':
        action = preds_to_action_loot(prediction, prednames)
    return action
