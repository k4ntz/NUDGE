from enum import Enum
from typing import List, Final

class GetoutActions(Enum):

    NOOP = 0
    MOVE_LEFT = 1
    MOVE_RIGHT = 2
    MOVE_UP = 3
    MOVE_DOWN = 4


# GetoutExplicitActions
CJA_NOOP: Final[int] = 0
CJA_MOVE_LEFT: Final[int] = 1
CJA_MOVE_RIGHT: Final[int] = 2
CJA_MOVE_UP: Final[int] = 3
CJA_MOVE_DOWN: Final[int] = 4
CJA_MOVE_LEFT_UP: Final[int] = 5
CJA_MOVE_RIGHT_UP: Final[int] = 6
CJA_MOVE_LEFT_DOWN: Final[int] = 7
CJA_MOVE_RIGHT_DOWN: Final[int] = 8
CJA_NUM_EXPLICIT_ACTIONS = 9


def unify_coin_jump_actions(coin_jump_actions: List[GetoutActions]):
    if len(coin_jump_actions) == 0:
        return CJA_NOOP
    if len(coin_jump_actions) == 1:
        return coin_jump_actions[0].value

    flags = [False] * 5
    for action in coin_jump_actions:
        flags[action.value] = True

    # map left->1 right->2 and left&right->3->0
    left_right_encode = ((1 if flags[CJA_MOVE_LEFT] else 0) + (2 if flags[CJA_MOVE_RIGHT] else 0)) % 3

    if flags[CJA_MOVE_UP]:
        return 4 + left_right_encode if left_right_encode != 0 else CJA_MOVE_UP
    else:
        if flags[CJA_MOVE_DOWN]:
            return 6 + left_right_encode if left_right_encode != 0 else CJA_MOVE_DOWN

    return left_right_encode


def coin_jump_actions_from_unified(unified_coin_jump_action):
    if unified_coin_jump_action <= 4:
        return [GetoutActions(unified_coin_jump_action)]
    if unified_coin_jump_action < 7:
        return [GetoutActions(unified_coin_jump_action-4), GetoutActions.MOVE_UP]
    else:
        return [GetoutActions(unified_coin_jump_action-6), GetoutActions.MOVE_DOWN]
