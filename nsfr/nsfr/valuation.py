from typing import Sequence, Dict, Any, Union
from abc import ABC
import inspect
import re

import torch
from torch import nn
from nsfr.fol.language import Language
from nsfr.fol.logic import Atom, Const
from nsfr.utils.common import load_module


class ValuationFunction(nn.Module, ABC):
    """Base class for valuation functions used inside valuation modules."""

    def __init__(self, pred_name: str):
        super().__init__()
        self.pred_name = pred_name

    def forward(self, **kwargs) -> torch.Tensor:
        raise NotImplementedError()

    def bool2probs(self, bool_tensor: torch.Tensor) -> torch.Tensor:
        """Converts a Boolean tensor into a probability tensor by assigning
        probability 0.99 for True
        probability 0.01 for False."""
        return torch.where(bool_tensor, 0.99, 0.01)


class ValuationModule(nn.Module, ABC):
    """Turns logic state representations into valuated atom probabilities according to
    the environment-specific valuation functions.

    Args:
        val_fn_path: The path to the file containing the user-specified valuation functions.
    """

    lang: Language
    device: Union[torch.device, str]
    val_fns: Dict[str, ValuationFunction]  # predicate names to corresponding valuation fn

    def __init__(self, val_fn_path: str, lang: Language, device: Union[torch.device, str],
                 pretrained: bool = True):
        super().__init__()

        # Parse all valuation functions
        val_fn_module = load_module(val_fn_path)
        all_functions = inspect.getmembers(val_fn_module, inspect.isfunction)
        self.val_fns = {fn[0]: fn[1] for fn in all_functions}

        self.lang = lang
        self.device = device
        self.pretrained = pretrained

    def forward(self, zs: torch.Tensor, atom: Atom):
        """Convert the object-centric representation to a valuation tensor.

            Args:
                zs (tensor): The object-centric representation (the output of the YOLO model).
                atom (atom): The target atom to compute its probability.

            Returns:
                A batch of the probabilities of the target atom.
        """
        try:
            val_fn = self.val_fns[atom.pred.name]
        except KeyError as e:
            raise NotImplementedError(f"Missing implementation for valuation function '{atom.pred.name}'.")
        # term: logical term
        # args: the vectorized input evaluated by the value function
        args = [self.ground_to_tensor(term, zs) for term in atom.terms]
        return val_fn(*args)

    def ground_to_tensor(self, const: Const, zs: torch.Tensor):
        """Ground constant (term) into tensor representations.

            Args:
                const (const): The term to be grounded.
                zs (tensor): The object-centric state representation.
        """
        # Check if the constant name is in the reserved style, e.g., "obj1"
        result = re.match("obj([1-9][0-9]*)", const.name)
        if result is not None:
            # The constant is an object constant
            obj_id = result[1]
            obj_index = int(obj_id) - 1
            return zs[:, obj_index]

        elif const.dtype.name == 'object':
            obj_index = self.lang.term_index(const)
            return zs[:, obj_index]

        elif const.dtype.name == 'image':
            return zs

        else:
            return self.term_to_onehot(const, batch_size=zs.size(0))

    def term_to_onehot(self, term, batch_size):
        """Ground terms into tensor representations.

            Args:
                term (term): The term to be grounded.
                zs (tensor): The object-centric representation.

            Return:
                The tensor representation of the input term.
        """
        if term.dtype.name == 'color':
            return self.to_onehot_batch(self.colors.index(term.name), len(self.colors), batch_size)
        elif term.dtype.name == 'shape':
            return self.to_onehot_batch(self.shapes.index(term.name), len(self.shapes), batch_size)
        elif term.dtype.name == 'material':
            return self.to_onehot_batch(self.materials.index(term.name), len(self.materials), batch_size)
        elif term.dtype.name == 'size':
            return self.to_onehot_batch(self.sizes.index(term.name), len(self.sizes), batch_size)
        elif term.dtype.name == 'side':
            return self.to_onehot_batch(self.sides.index(term.name), len(self.sides), batch_size)
        elif term.dtype.name == 'type':
            return self.to_onehot_batch(self.lang.term_index(term), len(self.lang.get_by_dtype_name(term.dtype.name)),
                                        batch_size)
        else:
            assert True, 'Invalid term: ' + str(term)

    def to_onehot_batch(self, i, length, batch_size):
        """Compute the one-hot encoding that is expanded to the batch size."""
        onehot = torch.zeros(batch_size, length).to(self.device)
        onehot[:, i] = 1.0
        return onehot
