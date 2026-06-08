import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

def load_data(csv_file):

    df = pd.read_csv(csv_file)

    features = [
        "city",
        "weapon_used",
        "gang_id",
        "case_status",
        "arrest_status",
        "evidence_type"
    ]

    target = "crime_type"

    X = df[features].copy()
    y = df[target].copy()

    X = X.astype(str)
    y = y.astype(str)

    encoders = {}

    for col in X.columns:

        encoder = LabelEncoder()

        X[col] = encoder.fit_transform(
            X[col]
        )

        encoders[col] = encoder

    target_encoder = LabelEncoder()

    y = target_encoder.fit_transform(
        y
    )

    return X, y


def create_model():

    return LogisticRegression(
        max_iter=1000,
        random_state=42
    )
