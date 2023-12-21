import torch
from nsfr.utils.common import bool_to_probs


def type(z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    z_type = z[:, 0:2]  # [1, 0, 0, 0] * [1.0, 0, 0, 0] .sum = 0.0  type(obj1, key):0.0
    prob = (a * z_type).sum(dim=1)
    return prob


def closeby(z_1: torch.Tensor, z_2: torch.Tensor) -> torch.Tensor:
    c_1 = z_1[:, -2:]
    c_2 = z_2[:, -2:]

    dis_x = abs(c_1[:, 0] - c_2[:, 0]) / 171
    dis_y = abs(c_1[:, 1] - c_2[:, 1]) / 171

    result = bool_to_probs((dis_x < 2.5) & (dis_y <= 0.1))

    return result


def on_left(z_1: torch.Tensor, z_2: torch.Tensor):
    c_1 = z_1[:, -2]
    c_2 = z_2[:, -2]
    diff = c_2 - c_1
    result = bool_to_probs(diff > 0)
    return result


def on_right(z_1: torch.Tensor, z_2: torch.Tensor):
    c_1 = z_1[:, -2]
    c_2 = z_2[:, -2]
    diff = c_2 - c_1
    result = bool_to_probs(diff < 0)
    return result


def same_row(z_1: torch.Tensor, z_2: torch.Tensor):
    c_1 = z_1[:, -1]
    c_2 = z_2[:, -1]
    diff = abs(c_2 - c_1)
    result = bool_to_probs(diff < 6)
    return result


def above_row(z_1: torch.Tensor, z_2: torch.Tensor):
    c_1 = z_1[:, -1]
    c_2 = z_2[:, -1]
    diff = c_2 - c_1
    result1 = bool_to_probs(diff < 23)
    result2 = bool_to_probs(diff > 4)
    return result1 * result2


def top5car(z_1: torch.Tensor):
    y = z_1[:, -1]
    result = bool_to_probs(y > 100)
    return result


def bottom5car(z_1: torch.Tensor):
    y = z_1[:, -1]
    result = bool_to_probs(y < 100)
    return result
