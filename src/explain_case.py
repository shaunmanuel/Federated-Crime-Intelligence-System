import pandas as pd

df = pd.read_csv(
    "data/processed_crime_data.csv"
)

case1 = input("Enter Case ID 1: ").strip()
case2 = input("Enter Case ID 2: ").strip()

row1 = df[df["case_id"] == case1]
row2 = df[df["case_id"] == case2]

if row1.empty or row2.empty:
    print("Case not found.")
    exit()

row1 = row1.iloc[0]
row2 = row2.iloc[0]

reasons = []

checks = [
    ("suspect_id", "Same suspect"),
    ("gang_id", "Same gang"),
    ("city", "Same city"),
    ("weapon_used", "Same weapon"),
    ("phone_number", "Same phone"),
    ("vehicle_number", "Same vehicle"),
    ("bank_account", "Same bank account"),
    ("family_group_id", "Same family group"),
    ("safehouse_id", "Same safehouse"),
    ("associate_group", "Same associate group"),
    ("meeting_location_id", "Same meeting location"),
]

for field, label in checks:
    if str(row1[field]) == str(row2[field]):
        reasons.append(
            f"{label}: {row1[field]}"
        )

print("\nCASE 1")
print("-" * 40)
print(row1[["case_id","crime_type","city","gang_id"]])

print("\nCASE 2")
print("-" * 40)
print(row2[["case_id","crime_type","city","gang_id"]])

print("\nEXPLANATION")
print("-" * 40)

if reasons:
    for reason in reasons:
        print("✓", reason)
else:
    print("No direct links found.")
