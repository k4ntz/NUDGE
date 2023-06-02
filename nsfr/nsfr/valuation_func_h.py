import torch
import torch.nn as nn
import math


################################
# Valuation functions for loot #
################################


class TypeValuationFunction(nn.Module):
    """The function v_object-type
    type(obj1, agent):0.98
    type(obj2, door）：0.87
    """

    def __init__(self):
        super(TypeValuationFunction, self).__init__()

    def forward(self, z, a):
        """
        Args:
            z (tensor): 2-d tensor B * d of object-centric representation.
                [agent, key, door, blue, green, red, got_key, X, Y]
            a (tensor): The one-hot tensor that is expanded to the batch size.

        Returns:
            A batch of probabilities.
        """
        z_type = z[:, 0:3]  # [1, 0, 0, 0] * [1.0, 0, 0, 0] .sum = 0.0  type(obj1, key):0.0
        prob = (a * z_type).sum(dim=1)
        return prob


class ColorValuationFunction(nn.Module):
    """The function v_object-color
    type(obj1, agent):0.98
    type(obj2, fish）：0.87
    """

    def __init__(self):
        super(ColorValuationFunction, self).__init__()

    def forward(self, z, a):
        """
        Args:
            z (tensor): 2-d tensor B * d of object-centric representation.
                [agent, key, door, blue, green, red, got_key, X, Y]
            a (tensor): The one-hot tensor that is expanded to the batch size.

        Returns:
            A batch of probabilities.
        """
        z_type = z[:, 3:6]  # [1.0, 0] * [1.0, 0] .sum = 0.0  color(obj1, green): 1.0
        prob = (a * z_type).sum(dim=1)

        return prob


class CloseValuationFunction(nn.Module):
    """The function v_close.
    """

    def __init__(self):
        super(CloseValuationFunction, self).__init__()

    def forward(self, z_1, z_2):
        """
        Args:
            z_1 (tensor): 2-d tensor (B * D), the object-centric representation.
            [agent, key, door, blue, green, red, got_key, X, Y]

        Returns:
            A batch of probabilities.
        """
        c_1 = z_1[:, -2:]
        c_2 = z_2[:, -2:]

        dis_x = abs(c_1[:, 0] - c_2[:, 0])
        dis_y = abs(c_1[:, 1] - c_2[:, 1])
        dis = dis_x[:] + dis_y[:] + 0.99
        result = 1 / dis
        return result


class ClosebyVerticalValuationFunction(nn.Module):
    """The function v_closeby.
    """

    def __init__(self):
        super(ClosebyVerticalValuationFunction, self).__init__()

    def forward(self, z_1, z_2):
        """
        Args:
            z_1 (tensor): 2-d tensor (B * D), the object-centric representation.
             [agent, key, door, blue, green, red, got_key, X, Y]

        Returns:
            A batch of probabilities.
        """
        c_1 = z_1[:, -2:]
        c_2 = z_2[:, -2:]

        dis_x = abs(c_1[:, 0] - c_2[:, 0])
        dis_y = abs(c_1[:, 1] - c_2[:, 1])

        result = torch.where((dis_y <= 1.1) & (dis_x < 0.7), 0.99, 0.1)

        return result


class ClosebyHorizontalValuationFunction(nn.Module):
    """The function v_closeby.
    """

    def __init__(self):
        super(ClosebyHorizontalValuationFunction, self).__init__()

    def forward(self, z_1, z_2):
        """
        Args:
            z_1 (tensor): 2-d tensor (B * D), the object-centric representation.
            [agent, key, door, blue, green, red, got_key, X, Y]

        Returns:
            A batch of probabilities.
        """
        c_1 = z_1[:, -2:]
        c_2 = z_2[:, -2:]

        dis_x = abs(c_1[:, 0] - c_2[:, 0])
        dis_y = abs(c_1[:, 1] - c_2[:, 1])

        result = torch.where((dis_x <= 1.1) & (dis_y < 0.7), 0.99, 0.1)

        return result


class OnTopValuationFunction(nn.Module):
    """The function v_closeby.
    """

    def __init__(self):
        super(OnTopValuationFunction, self).__init__()

    def forward(self, z_1, z_2):
        """
        Args:
            z_1 (tensor): 2-d tensor (B * D), the object-centric representation.
            [agent, key, door, blue, green, red, got_key, X, Y]

        Returns:
            A batch of probabilities.
        """
        c_1 = z_1[:, -2:]
        c_2 = z_2[:, -2:]

        # dis_y = c_1[:, -1] - c_2[:, -1]
        # result = torch.where(dis_y >= 0, 0.99, 0.01)
        result = fuzzy_position(c_2, c_1, keyword='top')
        # result = result[:] / torch.exp(dis_y[:])
        return result


class AtBottomValuationFunction(nn.Module):
    """The function v_closeby.
    """

    def __init__(self):
        super(AtBottomValuationFunction, self).__init__()

    def forward(self, z_1, z_2):
        """
        Args:
            z_1 (tensor): 2-d tensor (B * D), the object-centric representation.
             [agent, key, door, blue, green, red, got_key, X, Y]

        Returns:
            A batch of probabilities.
        """
        c_1 = z_1[:, -2:]
        c_2 = z_2[:, -2:]

        # dis_y = c_1[:, -1] - c_2[:, -1]

        result = fuzzy_position(c_2, c_1, keyword='bottom')

        return result


class OnLeftValuationFunction(nn.Module):
    """The function v_closeby.
    """

    def __init__(self):
        super(OnLeftValuationFunction, self).__init__()

    def forward(self, z_1, z_2):
        """
        Args:
            z_1 (tensor): 2-d tensor (B * D), the object-centric representation.
             [agent, key, door, blue, green, red, got_key, X, Y]

        Returns:
            A batch of probabilities.
        """
        c_1 = z_1[:, -2:]
        c_2 = z_2[:, -2:]

        result = fuzzy_position(c_2, c_1, keyword='left')

        return result


class OnRightValuationFunction(nn.Module):
    """The function v_closeby.
    """

    def __init__(self):
        super(OnRightValuationFunction, self).__init__()

    def forward(self, z_1, z_2):
        """
        Args: x
            z_1 (tensor): 2-d tensor (B * D), the object-centric representation.
             [agent, key, door, blue, green, red, got_key, X, Y]

        Returns:
            A batch of probabilities.
        """
        c_1 = z_1[:, -2:]
        c_2 = z_2[:, -2:]

        result = fuzzy_position(c_2, c_1, keyword='right')

        return result


class HaveKeyValuationFunction(nn.Module):
    """The function v_closeby.
    """

    def __init__(self):
        super(HaveKeyValuationFunction, self).__init__()

    def forward(self, z):
        """
        Args:
            z (tensor): 2-d tensor B * d of object-centric representation.
                [agent, key, door, blue, green, red, got_key, X, Y]

        Returns:
            A batch of probabilities.
        """
        c = z[:, -3]
        result = torch.where(c == 1, 0.99, 0.01)
        return result


class NotHaveKeyValuationFunction(nn.Module):
    """The function v_closeby.
    """

    def __init__(self):
        super(NotHaveKeyValuationFunction, self).__init__()

    def forward(self, z):
        """
        Args:
            z (tensor): 2-d tensor B * d of object-centric representation.
                [agent, key, door, blue, green, red, got_key, X, Y]

        Returns:
            A batch of probabilities.
        """
        c = z[:, -3]
        result = torch.where(c == 0, 0.99, 0.01)
        return result

def fuzzy_position(pos1, pos2, keyword):
    x = pos2[:, 0] - pos1[:, 0]
    y = pos2[:, 1] - pos1[:, 1]
    tan = torch.atan2(y, x)
    degree = tan[:] / torch.pi * 180
    if keyword == 'top':
        probs = 1 - abs(degree[:] - 90) / 90
        result = torch.where((180 >= degree) & (degree >= 0), probs * 0.9, 0.)
    elif keyword == 'left':
        probs = (abs(degree[:]) - 90) / 90
        result = torch.where((degree <= -90) | (degree >= 90), probs * 0.9, 0.)
    elif keyword == 'bottom':
        probs = 1 - abs(degree[:] + 90) / 90
        result = torch.where((0 >= degree) & (degree >= -180), probs * 0.9, 0.)
    elif keyword == 'right':
        probs = 1 - abs(degree[:]) / 90
        result = torch.where((90 >= degree) & (degree >= -90), probs * 0.9, 0.)

    return result
