import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans

print("Loading dataset...")

df = pd.read_csv(
    "data/processed_crime_data.csv"
)

features = [
    "crime_type",
    "weapon_used",
    "city",
    "evidence_type"
]

data = df[features].copy()

encoders = {}

for col in features:

    encoder = LabelEncoder()

    data[col] = encoder.fit_transform(
        data[col].astype(str)
    )

    encoders[col] = encoder

print("Running KMeans Clustering...")

kmeans = KMeans(
    n_clusters=5,
    random_state=42,
    n_init=10
)

df["mo_cluster"] = kmeans.fit_predict(data)

df.to_csv(
    "data/mo_results.csv",
    index=False
)

print("\nMO Analysis Completed")

print(
    "\nCluster Distribution:\n"
)

print(
    df["mo_cluster"]
    .value_counts()
    .sort_index()
)

print("\nSample Cases Per Cluster:\n")

for cluster in sorted(
    df["mo_cluster"].unique()
):

    print(
        f"\n===== Cluster {cluster} ====="
    )

    sample = df[
        df["mo_cluster"] == cluster
    ][[
        "crime_type",
        "weapon_used",
        "city",
        "evidence_type"
    ]].head(5)

    print(sample)

df.to_csv(
    "data/mo_results.csv",
    index=False
)

print(
    "\nMO results saved to data/mo_results.csv"
)
