import pickle
import community as community_louvain

print("Loading graph...")

with open("graphs/crime_graph.pkl", "rb") as f:
    G = pickle.load(f)

print("Running Louvain Community Detection...")

partition = community_louvain.best_partition(G)

num_communities = len(set(partition.values()))

print(f"\nTotal Communities Found: {num_communities}")

community_sizes = {}

for node, community_id in partition.items():
    community_sizes[community_id] = (
        community_sizes.get(community_id, 0) + 1
    )

print("\nTop 10 Largest Communities:\n")

largest = sorted(
    community_sizes.items(),
    key=lambda x: x[1],
    reverse=True
)

for community_id, size in largest[:10]:
    print(
        f"Community {community_id}: {size} nodes"
    )
