def pred2action(action, prednames):
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
