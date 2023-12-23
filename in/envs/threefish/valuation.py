import torch
from nsfr.utils.valuation import fuzzy_position


def obj_type(z, a):
    z_type = z[:, 0:2]  # [1.0, 0] * [1.0, 0] .sum = 0.0  type(obj1, agent): 1.0
    prob = (a * z_type).sum(dim=1)
    return prob


def color(z, a):
    z_type = z[:, 2:4]  # [1.0, 0] * [1.0, 0] .sum = 0.0  color(obj1, green): 1.0
    prob = (a * z_type).sum(dim=1)

    return prob


def on_top(z_1, z_2):
    c_1 = z_1[:, -2:]
    c_2 = z_2[:, -2:]

    result = fuzzy_position(c_2, c_1, keyword='top')
    return result


def high_level(z_1, z_2):
    c_1 = z_1[:, -2:]
    c_2 = z_2[:, -2:]
    diff = c_2[:, 1] - c_1[:, 1]
    # result = utils_bf.fuzzy_position(c_2, c_1, keyword='top')
    result = torch.where(diff <= 0, 99, 0)
    return result


def low_level(z_1, z_2):
    c_1 = z_1[:, -2:]
    c_2 = z_2[:, -2:]
    diff = c_2[:, 1] - c_1[:, 1]
    # result = utils_bf.fuzzy_position(c_2, c_1, keyword='top')
    result = torch.where(diff > 0, 99, 0)
    return result


def on_left(z_1, z_2):
    c_1 = z_1[:, -2:]
    c_2 = z_2[:, -2:]

    result = fuzzy_position(c_2, c_1, keyword='left')
    return result


def at_bottom(z_1, z_2):
    c_1 = z_1[:, -2:]
    c_2 = z_2[:, -2:]

    result = fuzzy_position(c_2, c_1, keyword='bottom')
    return result


def on_right(z_1, z_2):
    c_1 = z_1[:, -2:]
    c_2 = z_2[:, -2:]

    result = fuzzy_position(c_2, c_1, keyword='right')
    return result


def closeby(z_1, z_2):
    c_1 = z_1[:, -2:]
    c_2 = z_2[:, -2:]

    r_1 = z_1[:, 2]
    r_2 = z_2[:, 2]

    dis_x = torch.pow(c_2[:, 0] - c_1[:, 0], 2)
    dis_y = torch.pow(c_2[:, 1] - c_1[:, 1], 2)
    dis = torch.sqrt(dis_x[:] + dis_y[:])
    dis = abs(dis[:] - r_1[:] - r_2[:])
    probs = torch.where(dis <= 2, 0.99, 0.)
    return probs


def is_bigger_than(z_1, z_2):
    r_1 = z_1[:, -3]
    r_2 = z_2[:, -3]
    diff = r_2[:] - r_1[:]
    bigger = torch.where(diff < 0, 0.99, 0.)

    return bigger


def is_smaller_than(z_1, z_2):
    r_1 = z_1[:, 2]
    r_2 = z_2[:, 2]
    diff = r_2[:] - r_1[:]
    smaller = torch.where(diff >= 0, 0.99, 0.)

    return smaller
