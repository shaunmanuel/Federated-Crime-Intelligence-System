import random


WEAPON_MAP = {
    "Homicide": [
        "Pistol",
        "Knife",
        "Poison",
        "Iron Rod",
        "Rifle"
    ],

    "Robbery": [
        "Pistol",
        "Knife",
        "Iron Rod",
        "Crowbar"
    ],

    "Kidnapping": [
        "Pistol",
        "Knife",
        "Rope"
    ],

    "Cyber Crime": [
        "Laptop",
        "Phishing Kit"
    ],

    "Fraud": [
        "Laptop",
        "Phishing Kit"
    ],

    "Drug Trafficking": [
        "Pistol",
        "Rifle"
    ],

    "Assault": [
        "Knife",
        "Iron Rod"
    ],

    "Theft": [
        "Crowbar",
        "Unknown"
    ]
}








EVIDENCE_MAP = {
    "Homicide": ["DNA", "Fingerprint"],
    "Robbery": ["Fingerprint", "CCTV"],
    "Kidnapping": ["CCTV", "Witness Statement"],
    "Cyber Crime": ["Digital Evidence"],
    "Financial Fraud": ["Financial Records"],
    "Drug Trafficking": ["Phone Records", "Financial Records"]
}
