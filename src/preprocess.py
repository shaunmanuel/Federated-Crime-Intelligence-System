import pandas as pd

df = pd.read_csv(
    "data/crime_intelligence_research_grade_v2.csv"
)

df = df.fillna("Unknown")

df["combined_text"] = (
    "Crime Type: " + df["crime_type"].astype(str)
    + ". City: " + df["city"].astype(str)
    + ". Weapon Used: " + df["weapon_used"].astype(str)
    + ". Evidence Type: " + df["evidence_type"].astype(str)
    + ". Case Status: " + df["case_status"].astype(str)
    + ". Gang ID: " + df["gang_id"].astype(str)
    + ". Suspect ID: " + df["suspect_id"].astype(str)
    + ". Associate Group: " + df["associate_group"].astype(str)
    + ". Risk Score: " + df["risk_score"].astype(str)
)

print(df["combined_text"].head())

df.to_csv(
    "data/processed_crime_data.csv",
    index=False
)

print("Processed dataset saved.")
