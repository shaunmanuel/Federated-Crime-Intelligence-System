import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

print("Loading embeddings...")

X = np.load("models/embeddings.npy")

df = pd.read_csv(
    "data/processed_crime_data.csv"
)

encoder = LabelEncoder()

y = encoder.fit_transform(
    df["crime_type"]
)

print("Saving Agency A...")

np.save(
    "data/agency_a_X.npy",
    X[:34000]
)

np.save(
    "data/agency_a_y.npy",
    y[:34000]
)

print("Saving Agency B...")

np.save(
    "data/agency_b_X.npy",
    X[34000:67000]
)

np.save(
    "data/agency_b_y.npy",
    y[34000:67000]
)

print("Saving Agency C...")

np.save(
    "data/agency_c_X.npy",
    X[67000:]
)

np.save(
    "data/agency_c_y.npy",
    y[67000:]
)

print("Federated datasets prepared.")
