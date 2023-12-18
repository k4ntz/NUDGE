import torch
from nsfr.utils.common import bool_to_probs


def obj_type(z, a):
    z_type = z[:, 0:4]  # [1, 0, 0, 0] * [1.0, 0, 0, 0] .sum = 0.0  type(obj1, key):0.0
    prob = (a * z_type).sum(dim=1)
    return prob


def closeby(z_1, z_2):
    c_1 = z_1[:, 4:]
    c_2 = z_2[:, 4:]

    dis_x = abs(c_1[:, 0] - c_2[:, 0])
    dis_y = abs(c_1[:, 1] - c_2[:, 1])

    result = bool_to_probs((dis_x < 2.5) & (dis_y <= 0.1))

    return result


def on_left(z_1, z_2):
    c_1 = z_1[:, 4]
    c_2 = z_2[:, 4]
    diff = c_2 - c_1
    result = bool_to_probs(diff > 0)
    return result


def on_right(z_1, z_2):
    c_1 = z_1[:, 4]
    c_2 = z_2[:, 4]
    diff = c_2 - c_1
    result = bool_to_probs(diff < 0)
    return result


def have_key(z):
    has_key = torch.ones(z.size(dim=0))
    c = torch.sum(z[:, :, 1], dim=1)
    result = has_key[:] - c[:]

    return result


def not_have_key(z):
    c = torch.sum(z[:, :, 1], dim=1)
    result = c[:]

    return result


def safe(z_1, z_2):
    c_1 = z_1[:, 4:]
    c_2 = z_2[:, 4:]

    dis_x = abs(c_1[:, 0] - c_2[:, 0])
    result = bool_to_probs(dis_x > 2)
    return result
