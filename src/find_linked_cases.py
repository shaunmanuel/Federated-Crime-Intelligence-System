import pandas as pd

df = pd.read_csv(
    "data/processed_crime_data.csv"
)

gang = df["gang_id"].value_counts().index[0]

cases = df[
    df["gang_id"] == gang
].head(5)

print(cases[
    ["case_id","gang_id","suspect_id"]
])
