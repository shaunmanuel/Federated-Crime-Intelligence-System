import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

print("Loading embeddings...")

X = np.load(
    "models/embeddings.npy"
)

print("Loading pairs...")

pairs = pd.read_csv(
    "data/similarity_pairs.csv"
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

X_train, X_test, y_train, y_test = train_test_split(
    features,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

X_train = torch.tensor(
    X_train,
    dtype=torch.float32
)

X_test = torch.tensor(
    X_test,
    dtype=torch.float32
)

y_train = torch.tensor(
    y_train,
    dtype=torch.float32
).view(-1, 1)

y_test = torch.tensor(
    y_test,
    dtype=torch.float32
).view(-1, 1)

model = nn.Sequential(
    nn.Linear(768, 256),
    nn.ReLU(),

    nn.Linear(256, 64),
    nn.ReLU(),

    nn.Linear(64, 1),
    nn.Sigmoid()
)

criterion = nn.BCELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

print("Training...")

epochs = 10

for epoch in range(epochs):

    optimizer.zero_grad()

    outputs = model(
        X_train
    )

    loss = criterion(
        outputs,
        y_train
    )

    loss.backward()

    optimizer.step()

    print(
        f"Epoch {epoch+1}",
        "Loss:",
        round(
            loss.item(),
            4
        )
    )

print("\nEvaluating...")

with torch.no_grad():

    preds = model(
        X_test
    )

    preds = (
        preds > 0.5
    ).float()

    accuracy = (
        preds == y_test
    ).float().mean()

print(
    "\nSimilarity Accuracy:",
    round(
        accuracy.item() * 100,
        2
    ),
    "%"
)

torch.save(
    model.state_dict(),
    "models/similarity_model.pth"
)

print(
    "\nSaved model:"
)

print(
    "models/similarity_model.pth"
)
