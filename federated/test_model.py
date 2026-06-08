from model import load_data
from model import create_model

print("Loading Agency A...")

X, y = load_data(
    "data/agency_a.csv"
)

print("Rows:", len(X))
print("Columns:", len(X.columns))

print("Training Logistic Regression...")

model = create_model()

model.fit(X, y)

print("Model trained successfully!")

accuracy = model.score(X, y)

print(
    "Training Accuracy:",
    round(accuracy * 100, 2),
    "%"
)
