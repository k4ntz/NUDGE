import numpy as np
import torch


def extract_logic_state(state, variant, **kwargs):
    """
    [X,Y]
    [agent,key_b,door_b,key_g,door_g,key_r,door_r]
    """
    states = torch.from_numpy(state['positions']).squeeze()
    if variant == 'lootplus':
        # input shape: [X,Y]* [agent,key_b,door_b,key_g,door_g,key_r,door_r]
        # output shape:[agent, key, door, blue, green, red ,got_key, X, Y]
        state_extracted = torch.tensor([
            [1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0, 1, 0, 0, 0]], dtype=torch.float64)
        state_extracted[:, -2:] = states[:]
        for i, state in enumerate(state_extracted):
            if state[-1] == 0 and state[-2] == 0:
                state_extracted[i] = torch.zeros((1, 9))
            elif i in [2, 4, 6] and state[-1] != 0 and state_extracted[i - 1][1] == 0:
                state_extracted[i][-3] = 1

    elif variant == 'loothard':
        # input shape: [X,Y]* [agent, key_b, door_b, key_g, door_g, exit]
        # output shape:[agent, key, door, blue, green, exit,got_key, X, Y]
        state_extracted = torch.tensor([
            [1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0]], dtype=torch.float64)
        states = states[0:6]
        state_extracted[:, -2:] = states[:]
        for i, state in enumerate(state_extracted):
            if state[-1] == 0 and state[-2] == 0:
                state_extracted[i] = torch.zeros((1, 9))
            elif i in [2, 4] and state[-1] != 0 and state_extracted[i - 1][1] == 0: # setting got_key
                state_extracted[i][-3] = 1

    elif variant in ["loot", "lootcolor"]:
        # input shape: [X,Y]* [agent,key_b,door_b,key_g,door_g]
        # output shape:[agent, key, door, blue/green, red/brown ,got_key, X, Y]
        state_extracted = torch.tensor([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 1, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 0, 0, 0],
            [0, 1, 0, 0, 1, 0, 0, 0],
            [0, 0, 1, 0, 1, 0, 0, 0]], dtype=torch.float64)
        states = states[0:5]
        state_extracted[:, -2:] = states[:]
        for i, state in enumerate(state_extracted):
            # 0 mean object does not exist
            if state[-1] == 0 and state[-2] == 0:
                # then set to all attributes 0
                state_extracted[i] = torch.zeros((1, 8))
            # if key = 0 but door !=0, means key of this door has picked
            elif i in [2, 4] and state[-1] != 0 and state_extracted[i - 1][1] == 0:
                state_extracted[i][-3] = 1

    else:
        raise ValueError(f"Invalid Loot variant '{variant}'.")

    state_extracted = state_extracted.unsqueeze(0)
    return state_extracted


def extract_neural_state(state, variant, **kwargs):
    state = state['positions']

    if variant == 'lootplus':
        state_extracted = np.array([[0, 0, 0, 0],
                              [0, 0, 1, 1],
                              [0, 0, 2, 1],
                              [0, 0, 1, 2],
                              [0, 0, 2, 2],
                              [0, 0, 1, 3],
                              [0, 0, 2, 3]], dtype=np.float32)
        state_extracted[:, 0:2] = state[0][:]

    elif variant == 'loothard':
        state_extracted = np.array([[0, 0, 0, 0],
                              [0, 0, 1, 1],
                              [0, 0, 2, 1],
                              [0, 0, 1, 2],
                              [0, 0, 2, 2],
                              [0, 0, 3, 0],
                              [0, 0, 0, 0]], dtype=np.float32)
        state_extracted[:, 0:2] = state[0][:]

    elif variant == 'loot':
        state_extracted = np.array([[0, 0, 0, 0],
                              [0, 0, 1, 1],
                              [0, 0, 2, 1],
                              [0, 0, 1, 2],
                              [0, 0, 2, 2],
                              [0, 0, 0, 0],
                              [0, 0, 0, 0]], dtype=np.float32)
        state_extracted[:, 0:2] = state[0][:]

    elif variant == 'lootcolor':
        state_extracted = np.array([[0, 0, 0, 0],
                              [0, 0, 1, 10],
                              [0, 0, 2, 10],
                              [0, 0, 1, 20],
                              [0, 0, 2, 20],
                              [0, 0, 0, 0],
                              [0, 0, 0, 0]], dtype=np.float32)
        state_extracted[:, 0:2] = state[0][:]

    else:
        raise ValueError(f"Invalid Loot variant '{variant}'.")

    state_extracted = state_extracted.reshape(-1).tolist()
    return torch.tensor(state_extracted)
