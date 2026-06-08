import pickle
import networkx as nx
from pyvis.network import Network

print("Loading graph...")

with open(
    "graphs/crime_graph.pkl",
    "rb"
) as f:

    G = pickle.load(f)


def generate_network_html(
    suspect_id
):

    suspect_id = suspect_id.strip()

    target = (
        f"Suspect:{suspect_id}"
    )

    if target not in G:

        return None

    # Create ego graph

    ego = nx.ego_graph(
        G,
        target,
        radius=1
    )

    # Limit graph size

    MAX_NODES = 50

    if ego.number_of_nodes() > MAX_NODES:

        important_nodes = [target]

        neighbors = list(
            ego.neighbors(
                target
            )
        )

        important_nodes.extend(
            neighbors[
                :MAX_NODES - 1
            ]
        )

        ego = ego.subgraph(
            important_nodes
        )

    net = Network(
        height="750px",
        width="100%",
        bgcolor="#ffffff",
        font_color="black"
    )

    # Add nodes

    for node in ego.nodes():

        if node.startswith(
            "Suspect:"
        ):

            color = "red"
            size = 25

        elif node.startswith(
            "Case:"
        ):

            color = "black"
            size = 20

        elif node.startswith(
            "Weapon:"
        ):

            color = "darkred"
            size = 18

        elif node.startswith(
            "City:"
        ):

            color = "cyan"
            size = 15

        elif node.startswith(
            "Family:"
        ):

            color = "pink"
            size = 15

        elif node.startswith(
            "Gang:"
        ):

            color = "orange"
            size = 20

        elif node.startswith(
            "Phone:"
        ):

            color = "green"
            size = 15

        elif node.startswith(
            "Vehicle:"
        ):

            color = "blue"
            size = 15

        elif node.startswith(
            "Bank:"
        ):

            color = "purple"
            size = 15

        elif node.startswith(
            "Safehouse:"
        ):

            color = "brown"
            size = 15

        else:

            color = "gray"
            size = 10

        net.add_node(
            node,
            label=node,
            color=color,
            size=size
        )

    # Add edges

    for source, destination in ego.edges():

        net.add_edge(
            source,
            destination
        )

    html = net.generate_html()

    return {
        "html": html,
        "nodes": ego.number_of_nodes(),
        "edges": ego.number_of_edges()
    }
