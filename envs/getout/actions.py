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


def map_action(prediction, alg, prednames=None):
    """map model action to game action"""
    if alg == 'ppo':
        # simplified action--- only left right up
        # action = coin_jump_actions_from_unified(torch.argmax(predictions).cpu().item() + 1)
        action = prediction + 1
    elif alg == 'logic':
        action = preds_to_action_getout(prediction, prednames)
    return action
