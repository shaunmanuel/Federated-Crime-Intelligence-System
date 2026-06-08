import numpy as np
import torch
import torch.nn as nn

from model_nn import CrimeNet

print("Loading Agency A...")

X = np.load("data/agency_a_X.npy")
y = np.load("data/agency_a_y.npy")

X = torch.tensor(
    X,
    dtype=torch.float32
)

y = torch.tensor(
    y,
    dtype=torch.long
)

model = CrimeNet()

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

print("Training...")

for epoch in range(5):

    optimizer.zero_grad()

    outputs = model(X)

    loss = criterion(
        outputs,
        y
    )

    loss.backward()

    optimizer.step()

    print(
        f"Epoch {epoch+1}:",
        round(loss.item(), 4)
    )

print("Training completed.")
