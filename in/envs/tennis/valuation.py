import torch as th
from nsfr.utils.common import bool_to_probs


def _close_x(player: th.Tensor, ballshadow: th.Tensor, thresh: float = 6.0, temp: float = 8.0) -> th.Tensor:
    # continuous closeness on X (0..1)
    dx = (player[..., 0] - ballshadow[..., 0]).abs()
    margin = thresh - dx                 # >0 means close
    return th.sigmoid(temp * margin)

def _hard_close_x(player: th.Tensor, ballshadow: th.Tensor, thresh: float = 6.0) -> th.Tensor:
    # hard-ish gate (0 or 1 as a prob) for suppressing movement when close
    return bool_to_probs((player[..., 0] - ballshadow[..., 0]).abs() < thresh)


def player_goup(player: th.Tensor, ball: th.Tensor,
                min_sep: float = 90.0, temp: float = 4) -> th.Tensor:
    # go UP when we're too close vertically AND we are already above (increase |dy|)
    dy = player[..., 1] - ball[..., 1]            # + if player above enemy
    too_close = th.sigmoid(temp * (min_sep - dy.abs()))
    above = bool_to_probs(dy >= 0)                 # only push up if we're above
    return too_close * above


def player_godown(player: th.Tensor, ball: th.Tensor,
                  min_sep: float = 90.0, temp: float = 4) -> th.Tensor:
    # go DOWN when we're too close vertically AND we are below (increase |dy|)
    dy = player[..., 1] - ball[..., 1]
    too_close = th.sigmoid(temp * (min_sep - dy.abs()))
    below = bool_to_probs(dy < 0)                  # only push down if we're below
    return too_close * below


def player_goright(player: th.Tensor, ballshadow: th.Tensor) -> th.Tensor:
    player_x = player[..., 0]
    ballshadow_x   = ballshadow[..., 0]
    # when close on X, do NOT move
    hit_gate = 1.0 - _hard_close_x(player, ballshadow, thresh=6.0)
    # positive margin means ball is to the right by >1 px
    margin = (ballshadow_x - player_x) - 1.0
    return hit_gate * th.sigmoid(4.0 * margin)


def player_goleft(player: th.Tensor, ballshadow: th.Tensor) -> th.Tensor:
    player_x = player[..., 0]
    ballshadow_x   = ballshadow[..., 0]
    # when close on X, do NOT move
    hit_gate = 1.0 - _hard_close_x(player, ballshadow, thresh=6.0)
    # positive margin means ball is to the left by >1 px
    margin = (player_x - ballshadow_x) - 1.0
    return hit_gate * th.sigmoid(4.0 * margin)


def ball_closeto_player(player: th.Tensor, ballshadow: th.Tensor,
                        x_thresh: float = 6.0, y_thresh: float = 20.0,
                        temp: float = 8.0) -> th.Tensor:
    dx = (player[..., 0] - ballshadow[..., 0]).abs()  # X distance
    dy = (player[..., 1] - ballshadow[..., 1]).abs()  # Y distance

    margin_x = x_thresh - dx
    margin_y = y_thresh - dy

    close_x = th.sigmoid(temp * margin_x)
    close_y = th.sigmoid(temp * margin_y)

    # fire only when BOTH X and Y are close
    return close_x * close_y



# def ball_comingto_player(ballshadow: th.Tensor) -> th.Tensor:
#     # not needed for this simple policy; keep returning 1 to avoid suppressing fire
#     return bool_to_probs(True)
























# def player_goright(player: th.Tensor, ball: th.Tensor) -> th.Tensor:
#     player_x = player[..., 0]
#     ball_x = ball[..., 0]
#     dist = player_x - ball_x
#     return sigmoid_smoothing(dist < 1, temperature=4)
#     # return bool_to_probs(player_y - ball_y > 8)
#
#
# def player_goleft(player: th.Tensor, ball: th.Tensor) -> th.Tensor:
#     player_x = player[..., 0]
#     ball_x = ball[..., 0]
#     dist = player_x - ball_x
#     return sigmoid_smoothing(dist > 1, temperature=4)
#
#
# def ball_closeto_player(player: th.Tensor, ball: th.Tensor) -> th.Tensor:
#     player_x = player[..., 0]
#     ball_x = ball[..., 0]
#     distance = abs(player_x - ball_x)
#     return sigmoid_smoothing(distance < 4 , temperature= 5.0)
#
#
# def ball_comingto_player(ball: th.Tensor) -> th.Tensor:
#     ball_y = ball[..., 1]
#     ball_y2 = ball[..., 3]
#     return sigmoid_smoothing(ball_y2 - ball_y < 0 , temperature=5.0)



#
#
# def ball_goto_enemy(ball: th.Tensor) -> th.Tensor:
#     ball_x = ball[..., 0]
#     ball_x2 = ball[..., 2]
#     return sigmoid_smoothing(ball_x2 - ball_x > 0, temperature=5.0)
#
#
# def ball_comingto_player(ball: th.Tensor) -> th.Tensor:
#     ball_x = ball[..., 0]
#     ball_x2 = ball[..., 2]
#     return sigmoid_smoothing(ball_x2 - ball_x < 0 , temperature=5.0)

def sigmoid_smoothing(bool_tensor: th.Tensor, temperature: float = 5.0) -> th.Tensor:
    """
    Apply sigmoid smoothing to a boolean tensor, converting True/False into soft probabilities.

    :param bool_tensor: Boolean tensor indicating condition (True = overlap, False = no overlap).
    :param temperature: Controls softness of probability conversion (higher = more binary-like).
    :return: Soft probability tensor (0.0 to 1.0).
    """
    return th.sigmoid(temperature * (bool_tensor.float() - 0.5))  # Adaptive smoothing