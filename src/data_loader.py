import pandas as pd

df = pd.read_csv(
    "data/crime_intelligence_research_grade_v2.csv"
)

print("Dataset loaded successfully")
print("Rows:", len(df))
print("Columns:", len(df.columns))

print("\nColumn Names:")
print(df.columns.tolist())

print("\nFirst 5 Rows:")
print(df.head())
