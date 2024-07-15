import torch


class MLP(torch.nn.Module):
    def __init__(self, device, has_softmax=False, has_sigmoid=False, out_size=6, as_dict=False, logic=False):
        super().__init__()
        self.logic = logic
        self.as_dict = as_dict
        self.device = device
        encoding_base_features = 2
        encoding_entity_features = 2
        encoding_max_entities = 49
        self.num_in_features = (encoding_base_features + encoding_entity_features) * encoding_max_entities

        modules = [
            torch.nn.Linear(self.num_in_features, 120).to(device),
            torch.nn.ReLU(inplace=True).to(device),
            torch.nn.Linear(120, 60).to(device),
            torch.nn.ReLU(inplace=True).to(device),
            torch.nn.Linear(60, out_size).to(device)
            # torch.nn.Linear(120, out_size).to(device)
        ]
        # modules = [
        #     torch.nn.Linear(self.num_in_features, 40),
        #     torch.nn.ReLU(inplace=True),
        #     torch.nn.Linear(40, out_size)
        # ] 

        if has_softmax:
            modules.append(torch.nn.Softmax(dim=-1))
            
        if has_sigmoid:
            modules.append(torch.nn.Sigmoid())

        self.mlp = torch.nn.Sequential(*modules)
        self.mlp.to(device)

    def forward(self, state):
        features = state.float().view(-1, self.num_in_features)
        y = self.mlp(features)
        return y
