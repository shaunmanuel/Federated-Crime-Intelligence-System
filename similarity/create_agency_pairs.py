import pandas as pd
import random

for agency in [
    "agency_a",
    "agency_b",
    "agency_c"
]:

    print(f"Creating pairs for {agency}")

    df = pd.read_csv(
        f"data/{agency}.csv"
    )

    pairs = []

    N = 5000

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

        label = int(
            row1["crime_type"]
            ==
            row2["crime_type"]
        )

        pairs.append(
            [idx1, idx2, label]
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
        f"data/{agency}_pairs.csv",
        index=False
    )

    print(
        pairs["label"]
        .value_counts()
    )
