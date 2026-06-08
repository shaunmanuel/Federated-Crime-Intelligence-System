import pickle

with open(
    "graphs/crime_graph.pkl",
    "rb"
) as f:
    G = pickle.load(f)

print("Nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())

degree = sorted(
    G.degree(),
    key=lambda x: x[1],
    reverse=True
)

print("\nTop 20 Most Connected Nodes\n")

for node, deg in degree[:20]:
    print(f"{node} --> {deg}")
