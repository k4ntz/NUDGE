import torch
from nsfr.utils.common import bool_to_probs
from nsfr.utils.valuation import fuzzy_position


def obj_type(z, a):
    z_type = z[:, 0:3]  # [1, 0, 0, 0] * [1.0, 0, 0, 0] .sum = 0.0  type(obj1, key):0.0
    prob = (a * z_type).sum(dim=1)
    return prob


def color(z, a):
    z_type = z[:, 3:6]  # [1.0, 0] * [1.0, 0] .sum = 0.0  color(obj1, green): 1.0
    prob = (a * z_type).sum(dim=1)

    return prob


def close(z_1, z_2):
    c_1 = z_1[:, -2:]
    c_2 = z_2[:, -2:]

    dis_x = abs(c_1[:, 0] - c_2[:, 0])
    dis_y = abs(c_1[:, 1] - c_2[:, 1])
    dis = dis_x[:] + dis_y[:] + 0.99
    result = 1 / dis
    return result


def closeby_vertical(z_1, z_2):
    c_1 = z_1[:, -2:]
    c_2 = z_2[:, -2:]

    dis_x = abs(c_1[:, 0] - c_2[:, 0])
    dis_y = abs(c_1[:, 1] - c_2[:, 1])

    result = bool_to_probs((dis_y <= 1.1) & (dis_x < 0.7))

    return result


def closeby_horizontal(z_1, z_2):
    c_1 = z_1[:, -2:]
    c_2 = z_2[:, -2:]

    dis_x = abs(c_1[:, 0] - c_2[:, 0])
    dis_y = abs(c_1[:, 1] - c_2[:, 1])

    result = bool_to_probs((dis_x <= 1.1) & (dis_y < 0.7))

    return result


def on_top(z_1, z_2):
    c_1 = z_1[:, -2:]
    c_2 = z_2[:, -2:]

    # dis_y = c_1[:, -1] - c_2[:, -1]
    # result = torch.where(dis_y >= 0, 0.99, 0.01)
    result = fuzzy_position(c_2, c_1, keyword='top')
    # result = result[:] / torch.exp(dis_y[:])
    return result


def at_bottom(z_1, z_2):
    c_1 = z_1[:, -2:]
    c_2 = z_2[:, -2:]

    # dis_y = c_1[:, -1] - c_2[:, -1]

    result = fuzzy_position(c_2, c_1, keyword='bottom')

    return result


def on_left(z_1, z_2):
    c_1 = z_1[:, -2:]
    c_2 = z_2[:, -2:]

    result = fuzzy_position(c_2, c_1, keyword='left')

    return result


def on_right(z_1, z_2):
    c_1 = z_1[:, -2:]
    c_2 = z_2[:, -2:]

    result = fuzzy_position(c_2, c_1, keyword='right')

    return result


def have_key(z):
    c = z[:, -3]
    result = torch.where(c == 1, 0.99, 0.01)
    return result


def not_have_key(z):
    c = z[:, -3]
    result = torch.where(c == 0, 0.99, 0.01)
    return result
