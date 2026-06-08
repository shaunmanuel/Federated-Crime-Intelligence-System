import sys
import numpy as np
import torch
import torch.nn as nn

import flwr as fl

from federated.model_nn import(
    CrimeNet,
    get_parameters,
    set_parameters
)

# -------------------------
# Load Agency Data
# -------------------------

agency = sys.argv[1]

print(f"\nLoading {agency}...\n")

X = np.load(
    f"data/{agency}_X.npy"
)

y = np.load(
    f"data/{agency}_y.npy"
)

X = torch.tensor(
    X,
    dtype=torch.float32
)

y = torch.tensor(
    y,
    dtype=torch.long
)

# -------------------------
# Model
# -------------------------

model = CrimeNet()

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

# -------------------------
# Flower Client
# -------------------------

class CrimeClient(
    fl.client.NumPyClient
):

    def get_parameters(
        self,
        config
    ):
        return get_parameters(model)

    def fit(
        self,
        parameters,
        config
    ):

        set_parameters(
            model,
            parameters
        )

        model.train()

        for epoch in range(1):

            optimizer.zero_grad()

            outputs = model(X)

            loss = criterion(
                outputs,
                y
            )

            loss.backward()

            optimizer.step()

        print(
            f"{agency} trained."
        )

        return (
            get_parameters(model),
            len(X),
            {}
        )

    def evaluate(
        self,
        parameters,
        config
    ):

        set_parameters(
            model,
            parameters
        )

        model.eval()

        with torch.no_grad():

            outputs = model(X)

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            accuracy = (
                predictions == y
            ).float().mean()

        print(
            f"{agency} accuracy:",
            round(
                accuracy.item() * 100,
                2
            )
        )

        return (
            float(1.0 - accuracy.item()),
            len(X),
            {
                "accuracy":
                float(
                    accuracy.item()
                )
            }
        )

print(
    f"Connecting {agency}..."
)

fl.client.start_numpy_client(
    server_address="127.0.0.1:8080",
    client=CrimeClient()
)
