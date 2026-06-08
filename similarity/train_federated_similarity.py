import numpy as np
import pandas as pd
import torch
import torch.nn as nn


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


def load_agency_data(
    agency
):

    X = np.load(
        "models/embeddings.npy"
    )

    pairs = pd.read_csv(
        f"data/{agency}_pairs.csv"
    )

    X1 = X[
        pairs["case1"].values
    ]

    X2 = X[
        pairs["case2"].values
    ]

    y = pairs[
        "label"
    ].values

    features = np.concatenate(
        [X1, X2],
        axis=1
    )

    return features, y


def train_local_model(
    X,
    y,
    global_model=None,
    epochs=10
):

    model = SimilarityNet()

    if global_model is not None:

        model.load_state_dict(
            global_model.state_dict()
        )

    X = torch.tensor(
        X,
        dtype=torch.float32
    )

    y = torch.tensor(
        y,
        dtype=torch.float32
    ).view(-1, 1)

    criterion = nn.BCELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    model.train()

    for _ in range(epochs):

        optimizer.zero_grad()

        outputs = model(X)

        loss = criterion(
            outputs,
            y
        )

        loss.backward()

        optimizer.step()

    return model


def federated_average(
    models
):

    global_model = SimilarityNet()

    global_dict = (
        global_model.state_dict()
    )

    for key in global_dict:

        global_dict[key] = torch.stack(
            [
                m.state_dict()[key].float()
                for m in models
            ]
        ).mean(dim=0)

    global_model.load_state_dict(
        global_dict
    )

    return global_model


def evaluate(
    model,
    X,
    y
):

    X = torch.tensor(
        X,
        dtype=torch.float32
    )

    y = torch.tensor(
        y,
        dtype=torch.float32
    ).view(-1, 1)

    model.eval()

    with torch.no_grad():

        preds = model(X)

        preds = (
            preds > 0.5
        ).float()

        accuracy = (
            preds == y
        ).float().mean()

    return accuracy.item()


print(
    "Loading Agency A..."
)

X_a, y_a = load_agency_data(
    "agency_a"
)

print(
    "Loading Agency B..."
)

X_b, y_b = load_agency_data(
    "agency_b"
)

print(
    "Loading Agency C..."
)

X_c, y_c = load_agency_data(
    "agency_c"
)

global_model = SimilarityNet()

ROUNDS = 5

for round_num in range(
    ROUNDS
):

    print(
        f"\n===== ROUND {round_num + 1} ====="
    )

    model_a = train_local_model(
        X_a,
        y_a,
        global_model
    )

    model_b = train_local_model(
        X_b,
        y_b,
        global_model
    )

    model_c = train_local_model(
        X_c,
        y_c,
        global_model
    )

    global_model = federated_average(
        [
            model_a,
            model_b,
            model_c
        ]
    )

print(
    "\nEvaluating..."
)

acc_a = evaluate(
    global_model,
    X_a,
    y_a
)

acc_b = evaluate(
    global_model,
    X_b,
    y_b
)

acc_c = evaluate(
    global_model,
    X_c,
    y_c
)

global_acc = (
    acc_a +
    acc_b +
    acc_c
) / 3

torch.save(
    global_model.state_dict(),
    "models/global_similarity_model.pth"
)

print("\nResults\n")

print(
    "agency_a_similarity",
    round(acc_a * 100, 2),
    "%"
)

print(
    "agency_b_similarity",
    round(acc_b * 100, 2),
    "%"
)

print(
    "agency_c_similarity",
    round(acc_c * 100, 2),
    "%"
)

print(
    "global_similarity",
    round(global_acc * 100, 2),
    "%"
)

print(
    "\nSaved:"
)

print(
    "models/global_similarity_model.pth"
)
