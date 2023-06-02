import torch
import torch.nn as nn


################################
# Valuation functions for threefish #
################################


class TypeValuationFunction(nn.Module):
    """The function v_object-type
    type(obj1, agent):0.98
    type(obj2, fish）：0.87
    """

    def __init__(self):
        super(TypeValuationFunction, self).__init__()

    def forward(self, z, a):
        """
        Args:
            z (tensor): 2-d tensor B * d of object-centric representation.
                [agent, fish, radius, x, y]
            a (tensor): The one-hot tensor that is expanded to the batch size.

        Returns:
            A batch of probabilities.
        """
        z_type = z[:, 0:2]  # [1.0, 0] * [1.0, 0] .sum = 0.0  type(obj1, agent): 1.0
        prob = (a * z_type).sum(dim=1)

        return prob


class ColorValuationFunction(nn.Module):
    """The function v_object-type
    type(obj1, agent):0.98
    type(obj2, fish）：0.87
    [agent,fish,green,red,X,Y]
    """

    def __init__(self):
        super(ColorValuationFunction, self).__init__()

    def forward(self, z, a):
        """
        Args:
            z (tensor): 2-d tensor B * d of object-centric representation.
                [agent, fish, green, red, x, y]
            a (tensor): The one-hot tensor that is expanded to the batch size.

        Returns:
            A batch of probabilities.
        """
        z_type = z[:, 2:4]  # [1.0, 0] * [1.0, 0] .sum = 0.0  color(obj1, green): 1.0
        prob = (a * z_type).sum(dim=1)

        return prob


class OnTopValuationFunction(nn.Module):
    """The function v_closeby.
    """

    def __init__(self):
        super(OnTopValuationFunction, self).__init__()

    def forward(self, z_1, z_2):
        """
        Args:
            z_1 (tensor): 2-d tensor (B * D), the object-centric representation.
             [sfish, agent, threefish, x, y]

        Returns:
            A batch of probabilities.
        """
        c_1 = z_1[:, -2:]
        c_2 = z_2[:, -2:]

        result = fuzzy_position(c_2, c_1, keyword='top')
        return result


class HighLevelValuationFunction(nn.Module):
    """The function v_closeby.
    """

    def __init__(self):
        super(HighLevelValuationFunction, self).__init__()

    def forward(self, z_1, z_2):
        """
        Args:
            z_1 (tensor): 2-d tensor (B * D), the object-centric representation.
             [agent, threefish,size, x, y]

        Returns:
            A batch of probabilities.
        """
        c_1 = z_1[:, -2:]
        c_2 = z_2[:, -2:]
        diff = c_2[:, 1] - c_1[:, 1]
        # result = utils_bf.fuzzy_position(c_2, c_1, keyword='top')
        result = torch.where(diff <= 0, 99, 0)
        return result


class LowLevelValuationFunction(nn.Module):
    """The function v_closeby.
    """

    def __init__(self):
        super(LowLevelValuationFunction, self).__init__()

    def forward(self, z_1, z_2):
        """
        Args:
            z_1 (tensor): 2-d tensor (B * D), the object-centric representation.
             [sfish, agent, threefish, x, y]

        Returns:
            A batch of probabilities.
        """
        c_1 = z_1[:, -2:]
        c_2 = z_2[:, -2:]
        diff = c_2[:, 1] - c_1[:, 1]
        # result = utils_bf.fuzzy_position(c_2, c_1, keyword='top')
        result = torch.where(diff > 0, 99, 0)
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
             [sfish, agent, threefish, x, y]

        Returns:
            A batch of probabilities.
        """
        c_1 = z_1[:, -2:]
        c_2 = z_2[:, -2:]

        result = fuzzy_position(c_2, c_1, keyword='left')
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
             [sfish, agent, threefish, x, y]

        Returns:
            A batch of probabilities.
        """
        c_1 = z_1[:, -2:]
        c_2 = z_2[:, -2:]

        result = fuzzy_position(c_2, c_1, keyword='bottom')
        return result


class OnRightValuationFunction(nn.Module):
    """The function v_closeby.
    """

    def __init__(self):
        super(OnRightValuationFunction, self).__init__()

    def forward(self, z_1, z_2):
        """
        Args:
            z_1 (tensor): 2-d tensor (B * D), the object-centric representation.
             [sfish, agent, threefish, x, y]

        Returns:
            A batch of probabilities.
        """
        c_1 = z_1[:, -2:]
        c_2 = z_2[:, -2:]

        result = fuzzy_position(c_2, c_1, keyword='right')
        return result


class ClosebyValuationFunction(nn.Module):
    """The function v_closeby.
    """

    def __init__(self, device):
        super(ClosebyValuationFunction, self).__init__()
        self.device = device

    def forward(self, z_1, z_2):
        """
        Args:
            z_1 (tensor): 2-d tensor (B * D), the object-centric representation.
             [agent, fish, radius, x, y]

        Returns:
            A batch of probabilities.
        """
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


class BiggerValuationFunction(nn.Module):
    """The function v_closeby.
    """

    def __init__(self):
        super(BiggerValuationFunction, self).__init__()

    def forward(self, z_1, z_2):
        """
        Args:
            z (tensor): 2-d tensor B * d of object-centric representation.
                [agent,fish, radius, x, y] or [agent,fish,green,red,radius, x, y]

        Returns:
            A batch of probabilities.
        """
        r_1 = z_1[:, -3]
        r_2 = z_2[:, -3]
        diff = r_2[:] - r_1[:]
        bigger = torch.where(diff < 0, 0.99, 0.)

        return bigger


class SmallerValuationFunction(nn.Module):
    """The function v_closeby.
    """

    def __init__(self):
        super(SmallerValuationFunction, self).__init__()

    def forward(self, z_1, z_2):
        """
        Args:
            z (tensor): 2-d tensor B * d of object-centric representation.
                [agent,fish, radius, x, y]

        Returns:
            A batch of probabilities.
        """
        r_1 = z_1[:, 2]
        r_2 = z_2[:, 2]
        diff = r_2[:] - r_1[:]
        smaller = torch.where(diff >= 0, 0.99, 0.)

        return smaller


def fuzzy_position(pos1, pos2, keyword):
    x = pos2[:, 0] - pos1[:, 0]
    y = pos2[:, 1] - pos1[:, 1]
    tan = torch.atan2(y, x)
    degree = tan[:] / torch.pi * 180

    if keyword == 'top':
        probs = 1 - abs(degree[:] - 90) / 90
        result = torch.where((180 >= degree) & (degree >= 0), probs.double(), 0.)
    elif keyword == 'left':
        probs = (abs(degree[:]) - 90) / 90
        result = torch.where((degree <= -90) | (degree >= 90), probs.double(), 0.)
    elif keyword == 'bottom':
        probs = 1 - abs(degree[:] + 90) / 90
        result = torch.where((0 >= degree) & (degree >= -180), probs.double(), 0.)
    elif keyword == 'right':
        probs = 1 - abs(degree[:]) / 90
        result = torch.where((90 >= degree) & (degree >= -90), probs.double(), 0.)

    return result
