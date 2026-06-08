import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv(
    "data/processed_crime_data.csv"
)

agency_a, temp = train_test_split(
    df,
    test_size=0.66,
    random_state=42
)

agency_b, agency_c = train_test_split(
    temp,
    test_size=0.50,
    random_state=42
)

agency_a.to_csv(
    "data/agency_a.csv",
    index=False
)

agency_b.to_csv(
    "data/agency_b.csv",
    index=False
)

agency_c.to_csv(
    "data/agency_c.csv",
    index=False
)

print("Agency datasets created successfully!")

print("Agency A:", len(agency_a))
print("Agency B:", len(agency_b))
print("Agency C:", len(agency_c))
