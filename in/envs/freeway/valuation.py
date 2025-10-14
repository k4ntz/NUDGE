import torch
from nsfr.utils.common import bool_to_probs


def type(z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    z_type = z[:, 0:2]  # [1, 0, 0, 0] * [1.0, 0, 0, 0] .sum = 0.0  type(obj1, key):0.0
    prob = (a * z_type).sum(dim=1)
    return prob


def closeby(z_1: torch.Tensor, z_2: torch.Tensor) -> torch.Tensor:
    c1 = z_1[:, -2:]
    c2 = z_2[:, -2:]
    dis_x = (c1[:, 0] - c2[:, 0]).abs()
    dis_y = (c1[:, 1] - c2[:, 1]).abs()
    # tune these two numbers to your sprite sizes / lane widths:
    return bool_to_probs((dis_x <= 10) & (dis_y <= 14))


def on_left(z_1: torch.Tensor, z_2: torch.Tensor):
    c_1 = z_1[:, -2]
    c_2 = z_2[:, -2]
    result = bool_to_probs(c_1 < c_2)
    return result


def on_right(z_1: torch.Tensor, z_2: torch.Tensor):
    c_1 = z_1[:, -2]
    c_2 = z_2[:, -2]
    result = bool_to_probs(c_1 > c_2 )
    return result


def same_row(z_1, z_2):
    y1 = z_1[:, -1]
    y2 = z_2[:, -1]
    return bool_to_probs((y1 - y2).abs() <= 6)


def above_row(z_1: torch.Tensor, z_2: torch.Tensor):
    c_1 = z_1[:, -1]
    c_2 = z_2[:, -1]
    diff = c_2 - c_1
    result1 = bool_to_probs(diff < 23)
    result2 = bool_to_probs(diff > 4)
    return result1 * result2


LANE_H = 14.0


def lane_index(y: torch.Tensor) -> torch.Tensor:
    return (y / LANE_H).round()  # int lanes 0..9 (tune rounding vs floor)


def top5car(z):
    ln = lane_index(z[:, -1])
    # e.g., top 5 lanes are 'fast'
    return bool_to_probs((ln <= 4))


def bottom5car(z):
    ln = lane_index(z[:, -1])
    return bool_to_probs((ln >= 5))


LANE_TOL    = 6.0    # px
AHEAD_NEAR  = 6.0    # px
AHEAD_FAR   = 28.0   # px


def no_car_ahead(z_car, z_agent):
    """
    Pairwise predicate evaluated for each grounded (car, agent) pair.
    Returns 1.0 if THIS car is NOT in the ahead window (i.e., it doesn't block),
    0.0 if this car is ahead in the same lane (i.e., it blocks).
    Shapes: z_car [B,F], z_agent [B,F]
    """
    y_ch  = z_agent[:, -1]
    y_car = z_car[:, -1]

    same_row = (y_car - y_ch).abs() <= LANE_TOL
    diff     = y_ch - y_car                    # >0 if car is ahead (above)
    in_win   = (diff > AHEAD_NEAR) & (diff < AHEAD_FAR)
    blocks   = same_row & in_win               # this car blocks
    return bool_to_probs(~blocks)


STEP_Y   = 14.0  # lane height (px)
SAFE_DX  = 10.0
SAFE_DY  = 10.0

def clear_next_cell(z_car, z_agent):
    """
    True if THIS car does NOT occupy the cell the chicken would step into.
    Shapes: [B,F] each.
    """
    x_ch = z_agent[:, -2]; y_ch = z_agent[:, -1]
    x_ca = z_car[:,   -2]; y_ca = z_car[:,   -1]

    y_next = y_ch - STEP_Y
    conflict = (x_ca - x_ch).abs() <= SAFE_DX
    conflict &= (y_ca - y_next).abs() <= SAFE_DY
    return bool_to_probs(~conflict)


AHEAD_NEAR = 4.0
AHEAD_FAR  = 12.0
LANE_TOL   = 6.0

def very_close_ahead(z_car, z_agent):
    y_ch = z_agent[:, -1]; y_ca = z_car[:, -1]
    same = (y_ca - y_ch).abs() <= LANE_TOL
    diff = y_ch - y_ca
    in_win = (diff > AHEAD_NEAR) & (diff < AHEAD_FAR)
    return bool_to_probs(same & in_win)

# def top5car(z_1: torch.Tensor):
#     y = z_1[:, -1]
#     result = bool_to_probs(y > 100)
#     return result
#
#
# def bottom5car(z_1: torch.Tensor):
#     y = z_1[:, -1]
#     result = bool_to_probs(y < 100)
#     return result


# If each lane is ~14px tall and there are 10 lanes:


