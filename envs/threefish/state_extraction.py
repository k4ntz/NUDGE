import torch


def extract_logic_state(state, variant, **kwargs):
    state = torch.from_numpy(state['positions']).squeeze()

    if variant == "threefish":
        # input shape: [X,Y,radius]
        # output shape:[agent, fish, radius, X, Y]
        state_extracted = torch.zeros((3, 5))
        for i, state in enumerate(state):
            if i == 0:
                state_extracted[i, 0] = 1  # agent
                state_extracted[i, 2] = state[i, 2]  # radius
                state_extracted[i, 3] = state[i, 0]  # X
                state_extracted[i, 4] = state[i, 1]  # Y
            else:
                state_extracted[i, 1] = 1  # fish
                state_extracted[i, 2] = state[i, 2]  # radius
                state_extracted[i, 3] = state[i, 0]  # X
                state_extracted[i, 4] = state[i, 1]  # Y

    elif variant == "threefishcolor":
        # input shape: [X, Y, color, radius]
        # output shape: [agent, fish, green, red,radius, X, Y]
        state_extracted = torch.zeros((3, 7))
        for i, state in enumerate(state):
            if i == 0:
                state_extracted[i, 0] = 1  # agent
                state_extracted[i, -3] = state[i, 3]  # radius
                state_extracted[i, -2] = state[i, 0]  # X
                state_extracted[i, -1] = state[i, 1]  # Y
            else:
                state_extracted[i, 1] = 1  # fish
                if state[i, 2] == 1:
                    state_extracted[i, 2] = 1  # green
                else:
                    state_extracted[i, 3] = 1  # red
                state_extracted[i, -3] = state[i, 3]  # radius
                state_extracted[i, -2] = state[i, 0]  # X
                state_extracted[i, -1] = state[i, 1]  # Y

    else:
        raise ValueError(f"Invalid ThreeFish variant '{variant}'.")

    state_extracted = state_extracted.unsqueeze(0)
    return state_extracted.cuda()


def extract_neural_state(state, **kwargs):
    state = state['positions'][:, :, 0:3].reshape(-1).tolist()
    return torch.tensor(state)
