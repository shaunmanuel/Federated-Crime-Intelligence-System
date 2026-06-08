import pickle
import pandas as pd
import community as community_louvain

print("Loading graph...")

with open("graphs/crime_graph.pkl", "rb") as f:
    G = pickle.load(f)

print("Running Louvain Community Detection...")

partition = community_louvain.best_partition(G)

community_df = pd.DataFrame(
    list(partition.items()),
    columns=["node", "community"]
)

community_df.to_csv(
    "data/community_results.csv",
    index=False
)

print("\nCommunity results saved successfully!")

print(
    "Total Communities:",
    community_df["community"].nunique()
)

sizes = (
    community_df["community"]
    .value_counts()
    .sort_values(ascending=False)
)

print("\nTop 10 Largest Communities:\n")

for community_id, size in sizes.head(10).items():
    print(
        f"Community {community_id}: {size} nodes"
    )
