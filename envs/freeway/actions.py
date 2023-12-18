def preds_to_action_atari(action, prednames):
    if 'noop' in prednames[action]:
        return 0
    elif 'up' in prednames[action]:
        return 1
    elif 'down' in prednames[action]:
        return 2


def map_action(prediction, alg, prednames=None):
    """map model action to game action"""
    if alg == 'ppo':
        # simplified action--- only left right up
        # action = coin_jump_actions_from_unified(torch.argmax(predictions).cpu().item() + 1)
        action = prediction + 1
    elif alg == 'logic':
        action = preds_to_action_atari(prediction, prednames)
    return action
