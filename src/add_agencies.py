import pandas as pd

df = pd.read_csv(
    "data/processed_crime_data.csv"
)

agency_map = {
    "Kochi": "Kochi Police",
    "Chennai": "Chennai Police",
    "Bengaluru": "Bangalore Police",
    "Hyderabad": "Hyderabad Police",
    "Mumbai": "Mumbai Police",
    "Delhi": "Delhi Police",
    "Kolkata": "Kolkata Police",
    "Pune": "Pune Police",
    "Jaipur": "Jaipur Police",
    "Lucknow": "Lucknow Police"
}

df["agency"] = df["city"].map(
    agency_map
)

df.to_csv(
    "data/processed_crime_data.csv",
    index=False
)

print("Agency column added.")
print(df["agency"].value_counts())
