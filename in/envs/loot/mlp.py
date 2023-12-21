import torch


class MLP(torch.nn.Module):

    def __init__(self, has_softmax=False, out_size=5, logic=False, device=None):
        super().__init__()
        self.logic = logic
        self.device = device
        encoding_max_entities = 7
        encoding_entity_features = 4
        self.num_in_features = encoding_entity_features * encoding_max_entities

        modules = [
            torch.nn.Linear(self.num_in_features, 40),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(40, out_size)
        ]

        if has_softmax:
            modules.append(torch.nn.Softmax(dim=-1))

        self.mlp = torch.nn.Sequential(*modules)

    def forward(self, state):
        features = state
        y = self.mlp(features)
        return y
