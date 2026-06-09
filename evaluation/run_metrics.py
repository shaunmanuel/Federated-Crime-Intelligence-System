from evaluation_metrics import (
    calculate_metrics
)

# Ground Truth
y_true = [
    "Homicide",
    "Fraud",
    "Kidnapping",
    "Cyber Crime",
    "Robbery",
    "Drug Trafficking",
    "Homicide",
    "Fraud",
    "Kidnapping",
    "Cyber Crime"
]

# System Predictions
y_pred = [
    "Homicide",
    "Fraud",
    "Kidnapping",
    "Cyber Crime",
    "Robbery",
    "Drug Trafficking",
    "Homicide",
    "Fraud",
    "Robbery",
    "Cyber Crime"
]

results = calculate_metrics(
    y_true,
    y_pred
)

print(results)
