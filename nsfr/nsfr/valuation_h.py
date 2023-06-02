from .valuation_func_h import *


class HValuationModule(nn.Module):
    """A module to call valuation functions.
        Attrs:
            lang (language): The language.
            device (device): The device.
            layers (list(nn.Module)): The list of valuation functions.
            vfs (dic(str->nn.Module)): The dictionaty that maps a predicate name to the corresponding valuation function.
            attrs (dic(term->tensor)): The dictionary that maps an attribute term to the corresponding one-hot encoding.
            dataset (str): The dataset.
    """

    def __init__(self, lang, device, pretrained=True):
        super().__init__()
        self.lang = lang
        self.device = device
        self.layers, self.vfs = self.init_valuation_functions(
            device, pretrained)

    def init_valuation_functions(self, device, pretrained):
        """
            Args:
                device (device): The device.
                pretrained (bool): The flag if the neural predicates are pretrained or not.

            Retunrs:
                layers (list(nn.Module)): The list of valuation functions.
                vfs (dic(str->nn.Module)): The dictionaty that maps a predicate name to the corresponding valuation function.
        """
        layers = []
        vfs = {}  # pred name -> valuation function
        v_type = TypeValuationFunction()
        vfs['type'] = v_type
        layers.append(v_type)

        v_color = ColorValuationFunction()
        vfs['color'] = v_color
        layers.append(v_color)

        v_on_top = OnTopValuationFunction()
        vfs['on_top'] = v_on_top
        layers.append(v_on_top)

        v_at_bottom = AtBottomValuationFunction()
        vfs['at_bottom'] = v_at_bottom
        layers.append(v_at_bottom)

        v_on_left = OnLeftValuationFunction()
        vfs['on_left'] = v_on_left
        layers.append(v_on_left)

        v_on_right = OnRightValuationFunction()
        vfs['on_right'] = v_on_right
        layers.append(v_on_right)

        v_have_key = HaveKeyValuationFunction()
        vfs['have_key'] = v_have_key
        layers.append(v_have_key)

        v_not_have_key = NotHaveKeyValuationFunction()
        vfs['not_have_key'] = v_not_have_key
        layers.append(v_have_key)

        v_close = CloseValuationFunction()
        vfs['close'] = v_close
        layers.append(v_close)

        v_closeby_vertical = ClosebyVerticalValuationFunction()
        vfs['closeby_vertical'] = v_closeby_vertical
        layers.append(v_closeby_vertical)

        v_closeby_horizontal = ClosebyHorizontalValuationFunction()
        vfs['closeby_horizontal'] = v_closeby_horizontal
        layers.append(v_closeby_horizontal)

        return nn.ModuleList([v_type, v_color, v_on_top, v_at_bottom, v_on_left, v_on_right, v_have_key,
                              v_not_have_key, v_close, v_closeby_vertical, v_closeby_horizontal]), vfs

    def forward(self, zs, atom):
        """Convert the object-centric representation to a valuation tensor.

            Args:
                zs (tensor): The object-centric representaion (the output of the YOLO model).
                atom (atom): The target atom to compute its proability.

            Returns:
                A batch of the probabilities of the target atom.
        """
        # term: logical term
        # arg: vector representation of the term
        # zs = self.preprocess(zs)
        args = [self.ground_to_tensor(term, zs) for term in atom.terms]
        # call valuation function
        return self.vfs[atom.pred.name](*args)

    def ground_to_tensor(self, term, zs):
        """Ground terms into tensor representations.

            Args:
                term (term): The term to be grounded.
                zs (tensor): The object-centric representation.
        """
        term_index = self.lang.term_index(term)
        if term.dtype.name == 'object':
            return zs[:, term_index]
        elif term.dtype.name == 'image':
            return zs
        else:
            # other attributes
            return self.term_to_onehot(term, batch_size=zs.size(0))

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
        elif term.dtype.name == 'type':
            return self.to_onehot_batch(self.lang.term_index(term), len(self.lang.get_by_dtype_name(term.dtype.name)),
                                        batch_size)
        else:
            assert True, 'Invalid term: ' + str(term)

    def to_onehot_batch(self, i, length, batch_size):
        """Compute the one-hot encoding that is expanded to the batch size.
        """
        onehot = torch.zeros(batch_size, length, ).to(self.device)
        onehot[:, i] = 1.0
        return onehot
