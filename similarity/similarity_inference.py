import torch
import torch.nn as nn
import numpy as np


class SimilarityNet(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),

            nn.Linear(256, 64),
            nn.ReLU(),

            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):

        return self.network(x)


model = SimilarityNet()

model.load_state_dict(
    torch.load(
        "models/global_similarity_model.pth",
        map_location="cpu"
    )
)

model.eval()


def predict_similarity(
    emb1,
    emb2
):

    pair = np.concatenate(
        [emb1, emb2],
        axis=0
    )

    pair = torch.tensor(
        pair,
        dtype=torch.float32
    ).unsqueeze(0)

    with torch.no_grad():

        score = model(
            pair
        ).item()

    return scor
