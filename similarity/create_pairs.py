import pandas as pd
import numpy as np
import random

print("Loading dataset...")

df = pd.read_csv(
    "data/processed_crime_data.csv"
)

pairs = []

N = 10000

print("Generating pairs...")

for _ in range(N):

    idx1 = random.randint(
        0,
        len(df) - 1
    )

    idx2 = random.randint(
        0,
        len(df) - 1
    )

    row1 = df.iloc[idx1]
    row2 = df.iloc[idx2]

    label = 0

    if (
        row1["crime_type"]
        == row2["crime_type"]
    ):
        label = 1

    pairs.append(
        [
            idx1,
            idx2,
            label
        ]
    )

pairs = pd.DataFrame(
    pairs,
    columns=[
        "case1",
        "case2",
        "label"
    ]
)

pairs.to_csv(
    "data/similarity_pairs.csv",
    index=False
)

print("Saved.")
print(
    pairs["label"].value_counts()
)
