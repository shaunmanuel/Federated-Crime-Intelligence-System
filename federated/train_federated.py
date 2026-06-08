import numpy as np
import torch
import torch.nn as nn

from federated.model_nn import CrimeNet


def train_local_model(
    X,
    y,
    global_model=None,
    epochs=20
):

    model = CrimeNet()

    if global_model is not None:

        model.load_state_dict(
            global_model.state_dict()
        )

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    X = torch.tensor(
        X,
        dtype=torch.float32
    )

    y = torch.tensor(
        y,
        dtype=torch.long
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


def federated_average(models):

    global_model = CrimeNet()

    global_dict = global_model.state_dict()

    for key in global_dict.keys():

        global_dict[key] = torch.stack(
            [
                models[0].state_dict()[key].float(),
                models[1].state_dict()[key].float(),
                models[2].state_dict()[key].float()
            ]
        ).mean(dim=0)

    global_model.load_state_dict(
        global_dict
    )

    return global_model


def evaluate(model, X, y):

    X = torch.tensor(
        X,
        dtype=torch.float32
    )

    y = torch.tensor(
        y,
        dtype=torch.long
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

    return accuracy.item()


def run_federated_training():

    print("Loading Agency A...")

    X_a = np.load(
        "data/agency_a_X.npy"
    )

    y_a = np.load(
        "data/agency_a_y.npy"
    )

    print("Loading Agency B...")

    X_b = np.load(
        "data/agency_b_X.npy"
    )

    y_b = np.load(
        "data/agency_b_y.npy"
    )

    print("Loading Agency C...")

    X_c = np.load(
        "data/agency_c_X.npy"
    )

    y_c = np.load(
        "data/agency_c_y.npy"
    )

    # Initial global model

    global_model = CrimeNet()

    NUM_ROUNDS = 5

    for round_num in range(
        NUM_ROUNDS
    ):

        print(
            f"\n===== ROUND {round_num + 1} ====="
        )

        print(
            "Training Agency A..."
        )

        model_a = train_local_model(
            X_a,
            y_a,
            global_model=global_model,
            epochs=5
        )

        print(
            "Training Agency B..."
        )

        model_b = train_local_model(
            X_b,
            y_b,
            global_model=global_model,
            epochs=5
        )

        print(
            "Training Agency C..."
        )

        model_c = train_local_model(
            X_c,
            y_c,
            global_model=global_model,
            epochs=5
        )

        print(
            "Performing FedAvg..."
        )

        global_model = federated_average(
            [
                model_a,
                model_b,
                model_c
            ]
        )

    torch.save(
        global_model.state_dict(),
        "models/global_model.pth"
    )

    results = {

        "agency_a_accuracy":
        evaluate(
            global_model,
            X_a,
            y_a
        ),

        "agency_b_accuracy":
        evaluate(
            global_model,
            X_b,
            y_b
        ),

        "agency_c_accuracy":
        evaluate(
            global_model,
            X_c,
            y_c
        )
    }

    results[
        "global_accuracy"
    ] = (
        results["agency_a_accuracy"]
        + results["agency_b_accuracy"]
        + results["agency_c_accuracy"]
    ) / 3

    return results








if __name__ == "__main__":

    results = run_federated_training()

    print("\nResults\n")

    for key, value in results.items():

        print(
            key,
            round(
                value * 100,
                2
            ),
            "%"
        )
