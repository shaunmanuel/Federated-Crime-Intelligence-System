import torch
import torch.nn as nn


class CrimeNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(384, 128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 8)
        )

    def forward(self, x):
        return self.network(x)


def get_parameters(model):

    return [
        val.detach().cpu().numpy()
        for _, val in model.state_dict().items()
    ]


def set_parameters(model, parameters):

    params_dict = zip(
        model.state_dict().keys(),
        parameters
    )

    state_dict = {
        k: torch.tensor(v)
        for k, v in params_dict
    }

    model.load_state_dict(
        state_dict,
        strict=True
    )
