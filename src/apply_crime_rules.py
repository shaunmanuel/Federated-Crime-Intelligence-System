import pandas as pd
import random

from crime_rules import (
    WEAPON_MAP,
    EVIDENCE_MAP
)

print("Loading dataset...")

df = pd.read_csv(
    "data/crime_intelligence_research_grade_v2.csv"
)

df = df.fillna("Unknown")


def assign_weapon(crime_type):

    if crime_type in WEAPON_MAP:
        return random.choice(
            WEAPON_MAP[crime_type]
        )

    return "Unknown"


def assign_evidence(crime_type):

    if crime_type in EVIDENCE_MAP:
        return random.choice(
            EVIDENCE_MAP[crime_type]
        )

    return "Unknown"


print("Applying crime rules...")

df["weapon_used"] = df[
    "crime_type"
].apply(
    assign_weapon
)

df["evidence_type"] = df[
    "crime_type"
].apply(
    assign_evidence
)

print(
    df[
        [
            "crime_type",
            "weapon_used",
            "evidence_type"
        ]
    ].head(20)
)

df.to_csv(
    "data/crime_intelligence_research_grade_v2.csv",
    index=False
)

print(
    "\nCrime rules applied successfully."
)
