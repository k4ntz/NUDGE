import numpy as np


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


def map_action(prediction, alg, prednames=None):
    """map model action to game action"""
    if alg == 'ppo':
        action = simplify_action_bf(prediction)
    elif alg == 'logic':
        action = preds_to_action_threefish(prediction, prednames)
    return action
