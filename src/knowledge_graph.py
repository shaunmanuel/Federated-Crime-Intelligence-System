import pandas as pd
import networkx as nx
import pickle

print("Loading dataset...")

# Load dataset
df = pd.read_csv(
    "data/processed_crime_data.csv"
)

G = nx.Graph()

print("Building graph...")

for _, row in df.iterrows():

    # Main entities
    # Main entities

    case = f"Case:{row['case_id']}"
    suspect = f"Suspect:{row['suspect_id']}"
    gang = f"Gang:{row['gang_id']}"
    phone = f"Phone:{row['phone_number']}"
    vehicle = f"Vehicle:{row['vehicle_number']}"
    bank = f"Bank:{row['bank_account']}"
    weapon = f"Weapon:{row['weapon_used']}"
    city = f"City:{row['city']}"

    # Add nodes
    G.add_node(case, type="case")
    G.add_node(suspect, type="suspect")
    G.add_node(gang, type="gang")
    G.add_node(phone, type="phone")
    G.add_node(vehicle, type="vehicle")
    G.add_node(bank, type="bank")
    G.add_node(weapon, type="weapon")
    G.add_node(city, type="city")

    # Add relationships

    G.add_edge(
    case,
    suspect,
    relation="involves"
    )

    G.add_edge(
    case,
    weapon,
    relation="weapon_used"
    )

    G.add_edge(
    case,
    city,
    relation="occurred_in"
    ) 

    G.add_edge(
        suspect,
        gang,
        relation="member_of"
    )

    G.add_edge(
        suspect,
        phone,
        relation="uses_phone"
    )

    G.add_edge(
        suspect,
        vehicle,
        relation="owns_vehicle"
    )

    G.add_edge(
        suspect,
        bank,
        relation="holds_account"
    )

    # Associate relationship
    associate_id = str(row["associate_id"]).strip()

    if (
        associate_id != ""
        and associate_id.lower() != "unknown"
        and associate_id.lower() != "nan"
        and associate_id != row["suspect_id"]
    ):

        associate = f"Suspect:{associate_id}"

        G.add_node(
            associate,
            type="suspect"
        )

        G.add_edge(
            suspect,
            associate,
            relation="associated_with"
        )

    # Family relationship
    family_group = str(row["family_group_id"]).strip()

    if (
        family_group != ""
        and family_group.lower() != "unknown"
        and family_group.lower() != "nan"
    ):

        family_node = f"Family:{family_group}"

        G.add_node(
            family_node,
            type="family_group"
        )

        G.add_edge(
            suspect,
            family_node,
            relation="belongs_to_family"
        )

    # Safehouse relationship
    safehouse = str(row["safehouse_id"]).strip()

    if (
        safehouse != ""
        and safehouse.lower() != "unknown"
        and safehouse.lower() != "nan"
    ):

        safehouse_node = f"Safehouse:{safehouse}"

        G.add_node(
            safehouse_node,
            type="safehouse"
        )

        G.add_edge(
            suspect,
            safehouse_node,
            relation="uses_safehouse"
        )

print("\nGraph Statistics")
print("----------------")
print("Nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())

# Save graph
with open(
    "graphs/crime_graph.pkl",
    "wb"
) as f:
    pickle.dump(G, f)

print("\nGraph saved successfully!")
print("File: graphs/crime_graph.pkl")
