import numpy as np
import torch

device = torch.device('cuda:0')


def extract_logic_state_threefish(obs, args):
    """
    reshape states for nsfr
    """
    states = torch.from_numpy(obs['positions']).squeeze()
    if args.alg == 'logic':
        if args.env == "threefish":
            # input shape: [X,Y,radius]
            # output shape:[agent, fish, radius, X, Y]
            extracted_state = torch.zeros((3, 5))
            for i, state in enumerate(states):
                if i == 0:
                    extracted_state[i, 0] = 1  # agent
                    extracted_state[i, 2] = states[i, 2]  # radius
                    extracted_state[i, 3] = states[i, 0]  # X
                    extracted_state[i, 4] = states[i, 1]  # Y
                else:
                    extracted_state[i, 1] = 1  # fish
                    extracted_state[i, 2] = states[i, 2]  # radius
                    extracted_state[i, 3] = states[i, 0]  # X
                    extracted_state[i, 4] = states[i, 1]  # Y

            extracted_state = extracted_state.unsqueeze(0)
            return extracted_state.cuda()
        elif args.env == "threefishcolor":
            # input shape: [X, Y, color, radius]
            # output shape: [agent, fish, green, red,radius, X, Y]
            extracted_state = torch.zeros((3, 7))
            for i, state in enumerate(states):
                if i == 0:
                    extracted_state[i, 0] = 1  # agent
                    extracted_state[i, -3] = states[i, 3]  # radius
                    extracted_state[i, -2] = states[i, 0]  # X
                    extracted_state[i, -1] = states[i, 1]  # Y
                else:
                    extracted_state[i, 1] = 1  # fish
                    if states[i, 2] == 1:
                        extracted_state[i, 2] = 1  # green
                    else:
                        extracted_state[i, 3] = 1  # red
                    extracted_state[i, -3] = states[i, 3]  # radius
                    extracted_state[i, -2] = states[i, 0]  # X
                    extracted_state[i, -1] = states[i, 1]  # Y

            extracted_state = extracted_state.unsqueeze(0)
            return extracted_state.cuda()


def extract_neural_state_threefish(state, args):
    state = state['positions'][:,:,0:3].reshape(-1)
    state = state.tolist()

    return torch.tensor(state).to(device)


def simplify_action_bf(action):
    """simplify actions from 9 to 5
    """
    # model_ouput  [0, 1, 2, 3, 4]
    action_space = [1, 3, 4, 5, 7]
    action = action_space[action]
    return np.array([action])


def preds_to_action_threefish(action, prednames):
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


def action_map_threefish(prediction, args, prednames=None):
    """map model action to game action"""
    if args.alg == 'ppo':
        action = simplify_action_bf(prediction)
    elif args.alg == 'logic':
        action = preds_to_action_threefish(prediction, prednames)
    return action
