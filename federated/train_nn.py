import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from federated.model_nn import CrimeNet

print("Loading embeddings...")

X = np.load("models/embeddings.npy")

df = pd.read_csv(
    "data/processed_crime_data.csv"
)

print("Preparing labels...")

encoder = LabelEncoder()

y = encoder.fit_transform(
    df["crime_type"]
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
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
    dtype=torch.long
)

y_test = torch.tensor(
    y_test,
    dtype=torch.long
)

model = CrimeNet()

criterion = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

print("Training...")

epochs = 10

for epoch in range(epochs):

    optimizer.zero_grad()

    outputs = model(X_train)

    loss = criterion(
        outputs,
        y_train
    )

    loss.backward()

    optimizer.step()

    print(
        f"Epoch {epoch+1}/{epochs}",
        "Loss:",
        round(loss.item(), 4)
    )

print("\nEvaluating...")

with torch.no_grad():

    outputs = model(X_test)

    predictions = torch.argmax(
        outputs,
        dim=1
    )

    accuracy = (
        predictions == y_test
    ).float().mean()

print(
    "\nAccuracy:",
    round(
        accuracy.item() * 100,
        2
    ),
    "%"
)
