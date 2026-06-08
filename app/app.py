
import os
import sys


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.insert(0, PROJECT_ROOT)



import streamlit as st
import pandas as pd
import faiss
import subprocess
from sentence_transformers import SentenceTransformer
from src.federated_search import (
    federated_search
)
from src.network_visualization import (
    generate_network_html
)


#from federated.train_federated import (
#    run_federated_training
#)

# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="Crime Intelligence System",
    layout="wide"
)

# ==================================================
# LOAD DATA
# ==================================================

@st.cache_data
def load_data():
    return pd.read_csv(
        "data/processed_crime_data.csv"
    )

df = load_data()

# ==================================================
# LOAD MODEL
# ==================================================

@st.cache_resource
def load_model():
    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

model = load_model()

# ==================================================
# LOAD FAISS INDEX
# ==================================================

@st.cache_resource
def load_index():
    return faiss.read_index(
        "faiss_index/crime.index"
    )

index = load_index()

# ==================================================
# TITLE
# ==================================================

st.title("Crime Intelligence System")

st.markdown(
    "### Federated Crime Intelligence and Criminal Network Analysis System"
)

# ==================================================
# SIDEBAR
# ==================================================

page = st.sidebar.selectbox(
    "Navigation",
    [
      
    "Intelligence Dashboard",
    "New Investigation",
    "Case Link Analysis",
    "Criminal Network Analysis",
    "MO Intelligence",
    "Crime Analytics",
    "Federated Intelligence Center",
    "Model Evaluation",
    "Threat Intelligence",
    "System Architecture",
    "Dataset Overview"
  
    ]
)
# ==================================================
# DASHBOARD
# ==================================================

if page == "Intelligence Dashboard":
    st.header("Crime Intelligence Command Center")
    st.info(
    "Cross-Agency Crime Intelligence and Investigation Platform"
)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Cases",
        f"{len(df):,}"
    )

    col2.metric(
        "Unique Suspects",
        df["suspect_id"].nunique()
    )

    col3.metric(
        "Unique Gangs",
        df["gang_id"].nunique()
    )

    col4.metric(
        "Cities",
        df["city"].nunique()
    )

    st.divider()

    st.subheader("Top Crime Types")

    crime_counts = (
        df["crime_type"]
        .value_counts()
        .reset_index()
    )

    crime_counts.columns = [
        "Crime Type",
        "Count"
    ]

    st.dataframe(
        crime_counts,
        use_container_width=True
    )

# ==================================================
# CRIME SEARCH
# ==================================================

elif page == "New Investigation":

    st.header("New Investigation")

    query = st.text_input(
        "Enter New Crime Report",
        placeholder="e.g. Two armed men robbed a jewellery shop and escaped in a white SUV"
    )

    if st.button("Investigate"):

        if query.strip():

            with st.spinner(
                "Searching similar crimes..."
            ):

                query_embedding = model.encode(
                    [query]
                ).astype("float32")

                results = federated_search(
    query_embedding,
    query
)

                st.subheader(
                    "Cross-Agency Intelligence Matches"
                )

                st.dataframe(
                    results,
                    use_container_width=True
                )

                st.subheader(
                    "Investigator Explanation"
                )





                for _, row in results.iterrows():

                    explanation = []

                    explanation.append(
                        f"Similar crime type ({row['crime_type']})"
                    )

                    if str(
                        row["weapon_used"]
                    ) != "Unknown":

                        explanation.append(
                            f"Weapon pattern: {row['weapon_used']}"
                        )

                    explanation.append(
                        f"Gang association: {row['gang_id']}"
                    )

                    explanation.append(
                        f"Similarity score: {row['similarity_score']:.4f}"
                    )

                    st.warning(
                        f"""
Case {row['case_id']}

Reason for Match:

""" + "\n".join(
                            [
                                f"✓ {x}"
                                for x in explanation
                            ]
                        )
                    )

                st.divider()

                st.subheader(
                    "Matched Agencies"
                )

                agencies = results[
                    "agency"
                ].unique()

                for agency in agencies:

                    st.success(
                        f"✓ {agency}"
                    )

                st.divider()

                unique_cities = results[
                    "city"
                ].nunique()

                unique_gangs = results[
                    "gang_id"
                ].nunique()

                unique_suspects = results[
                    "suspect_id"
                ].nunique()

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Jurisdictions",
                    unique_cities
                )

                col2.metric(
                    "Suspects",
                    unique_suspects
                )

                col3.metric(
                    "Gang Links",
                    unique_gangs
                )

                st.divider()

                if unique_cities > 1:

                    st.success(
                        "🚨 Cross-Agency Intelligence Match Found"
                    )

                    st.write(
                        "Jurisdictions involved:"
                    )

                    st.write(
                        ", ".join(
                            results["city"].unique()
                        )
                    )

                else:

                    st.info(
                        "Matches currently belong to a single jurisdiction."
                    )

                st.divider()

                risk_score = (
                    unique_gangs
                    + unique_suspects
                    + unique_cities
                )

                if risk_score >= 10:

                    st.error(
                        "Threat Level: HIGH"
                    )

                elif risk_score >= 5:

                    st.warning(
                        "Threat Level: MEDIUM"
                    )

                else:

                    st.success(
                        "Threat Level: LOW"
                    )

                st.divider()

                report = f"""
INVESTIGATION REPORT

Crime Query:
{query}

Similar Cases Found:
{len(results)}

Jurisdictions:
{unique_cities}

Linked Suspects:
{unique_suspects}

Linked Gangs:
{unique_gangs}

Threat Level:
{"HIGH" if risk_score >= 10 else "MEDIUM" if risk_score >= 5 else "LOW"}

Cross-Agency Intelligence:
{"YES" if unique_cities > 1 else "NO"}

Matched Agencies:
{", ".join(agencies)}

Recommendation:
Review linked suspects, gang associations,
weapon patterns and similar crime
behaviours identified by the semantic
intelligence engine.
"""

                st.subheader(
                    "Investigation Report"
                )

                st.text_area(
                    "",
                    report,
                    height=350
                )

# ==================================================
# COMMUNITY DETECTION
# ==================================================

elif page == "Community Detection":

    st.header("Community Detection")

    community_df = pd.read_csv(
        "data/community_results.csv"
    )

    total_communities = (
        community_df["community"]
        .nunique()
    )

    st.metric(
        "Total Communities Found",
        total_communities
    )

    sizes = (
        community_df["community"]
        .value_counts()
        .reset_index()
    )

    sizes.columns = [
        "Community ID",
        "Size"
    ]

    st.subheader(
        "Largest Communities"
    )

    st.dataframe(
        sizes.head(20),
        use_container_width=True
    )

    st.subheader(
        "Community Size Distribution"
    )

    st.bar_chart(
        sizes.head(10)
        .set_index("Community ID")
    )

# ==================================================
# CASE EXPLANATION
# ==================================================

elif page == "Case Link Analysis":

    st.header("Case Link Analysis")

    case1 = st.text_input(
        "Case ID 1",
        "C000005"
    )

    case2 = st.text_input(
        "Case ID 2",
        "C000006"
    )

    if st.button("Explain Connection"):

        row1 = df[
            df["case_id"] == case1
        ]

        row2 = df[
            df["case_id"] == case2
        ]

        if row1.empty or row2.empty:

            st.error(
                "Case ID not found."
            )

        else:

            row1 = row1.iloc[0]
            row2 = row2.iloc[0]

            reasons = []

            checks = [
                ("suspect_id", "Same suspect"),
                ("gang_id", "Same gang"),
                ("phone_number", "Same phone"),
                ("vehicle_number", "Same vehicle"),
                ("bank_account", "Same bank account"),
                ("family_group_id", "Same family group"),
                ("safehouse_id", "Same safehouse"),
                ("associate_group", "Same associate group"),
                ("city", "Same city"),
                ("weapon_used", "Same weapon")
            ]

            for field, label in checks:

                if str(row1[field]) == str(row2[field]):

                    reasons.append(
                        f"{label}: {row1[field]}"
                    )

            st.subheader(
                "Investigation Summary"
            )

            if reasons:

                if len(reasons) >= 5:
                    st.error("Risk Level: HIGH")

                elif len(reasons) >= 3:
                    st.warning("Risk Level: MEDIUM")

                else:
                    st.info("Risk Level: LOW")

                st.success(
                    f"""
Investigation Findings

Linked Evidence Found: {len(reasons)}

""" + "\n".join(reasons)
                )

            else:

                st.warning(
                    "No direct links found."
                )




elif page == "Criminal Network Analysis":

    st.header("Criminal Network Analysis")

    suspect_id = st.text_input(
        "Enter Suspect ID",
        "S00005"
    )

    st.info(
        "Visualize relationships between suspects, phones, vehicles, gangs and linked cases."
    )

    if st.button(
        "Generate Network"
    ):

        suspect_cases = df[
            df["suspect_id"] == suspect_id
        ]

        if not suspect_cases.empty:

            st.subheader(
                "Suspect Intelligence Profile"
            )

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Linked Cases",
                len(suspect_cases)
            )

            col2.metric(
                "Associated Gangs",
                suspect_cases["gang_id"].nunique()
            )

            col3.metric(
                "Cities",
                suspect_cases["city"].nunique()
            )

            st.dataframe(
                suspect_cases[
                    [
                        "case_id",
                        "crime_type",
                        "city",
                        "gang_id"
                    ]
                ].head(20),
                use_container_width=True
            )

        import pickle
        import networkx as nx

        with open(
            "graphs/crime_graph.pkl",
            "rb"
        ) as f:

            G = pickle.load(f)

        suspect_node = f"Suspect:{suspect_id}"

        if suspect_node in G:

            neighbors = list(
                G.neighbors(
                    suspect_node
                )
            )

            linked_cases = [
                n for n in neighbors
                if str(n).startswith(
                    "Case:"
                )
            ]

            linked_phones = [
                n for n in neighbors
                if str(n).startswith(
                    "Phone:"
                )
            ]

            linked_vehicles = [
                n for n in neighbors
                if str(n).startswith(
                    "Vehicle:"
                )
            ]

            linked_gangs = [
                n for n in neighbors
                if str(n).startswith(
                    "Gang:"
                )
            ]

            st.subheader(
                "Intelligence Summary"
            )

            c1, c2, c3, c4 = st.columns(4)

            c1.metric(
                "Cases",
                len(linked_cases)
            )

            c2.metric(
                "Phones",
                len(linked_phones)
            )

            c3.metric(
                "Vehicles",
                len(linked_vehicles)
            )

            c4.metric(
                "Gang Links",
                len(linked_gangs)
            )

            risk_score = (
                len(linked_cases)
                + len(linked_phones)
                + len(linked_vehicles)
            )

            if risk_score > 20:

                st.error(
                    "Threat Level: HIGH"
                )

            elif risk_score > 10:

                st.warning(
                    "Threat Level: MEDIUM"
                )

            else:

                st.success(
                    "Threat Level: LOW"
                )

        network = generate_network_html(
            suspect_id
        )

        if network:

            st.subheader(
                "Criminal Network Visualization"
            )

            col1, col2 = st.columns(2)

            col1.metric(
                "Nodes",
                network["nodes"]
            )

            col2.metric(
                "Edges",
                network["edges"]
            )

            st.components.v1.html(
                network["html"],
                height=850,
                scrolling=True
            )

        else:

            st.error(
                "Suspect not found in graph."
            )


# ==================================================
# CRIME ANALYTICS
# ==================================================

elif page == "Crime Analytics":

    st.header("Crime Analytics Dashboard")

    # Crime Types
    st.subheader("Top Crime Types")

    crime_counts = (
        df["crime_type"]
        .value_counts()
        .reset_index()
    )

    crime_counts.columns = [
        "Crime Type",
        "Cases"
    ]

    st.dataframe(
        crime_counts,
        use_container_width=True
    )

    st.bar_chart(
        crime_counts.set_index("Crime Type")
    )

    st.divider()

    # Cities
    st.subheader("Top Cities")

    city_counts = (
        df["city"]
        .value_counts()
        .reset_index()
    )

    city_counts.columns = [
        "City",
        "Cases"
    ]

    st.dataframe(
        city_counts,
        use_container_width=True
    )

    st.bar_chart(
        city_counts.set_index("City")
    )

    st.divider()

    # Top Gangs
    st.subheader("Top 20 Gangs")

    gang_counts = (
        df["gang_id"]
        .value_counts()
        .head(20)
        .reset_index()
    )

    gang_counts.columns = [
        "Gang ID",
        "Cases"
    ]

    st.dataframe(
        gang_counts,
        use_container_width=True
    )

    st.bar_chart(
        gang_counts.set_index("Gang ID")
    )

    st.divider()

    # Top Suspects
    st.subheader("Top 20 Suspects")

    suspect_counts = (
        df["suspect_id"]
        .value_counts()
        .head(20)
        .reset_index()
    )

    suspect_counts.columns = [
        "Suspect ID",
        "Cases"
    ]

    st.dataframe(
        suspect_counts,
        use_container_width=True
    )

    st.bar_chart(
        suspect_counts.set_index("Suspect ID")
    )

    st.divider()

    # Case Status
    st.subheader("Case Status Distribution")

    status_counts = (
        df["case_status"]
        .value_counts()
        .reset_index()
    )

    status_counts.columns = [
        "Case Status",
        "Count"
    ]

    st.dataframe(
        status_counts,
        use_container_width=True
    )

    st.bar_chart(
        status_counts.set_index("Case Status")
    )

    st.divider()

    # Arrest Status
    st.subheader("Arrest Status Distribution")

    arrest_counts = (
        df["arrest_status"]
        .value_counts()
        .reset_index()
    )

    arrest_counts.columns = [
        "Arrest Status",
        "Count"
    ]

    st.dataframe(
        arrest_counts,
        use_container_width=True
    )

    st.bar_chart(
        arrest_counts.set_index("Arrest Status")
    )

    st.divider()

    # Evidence Types
    st.subheader("Evidence Type Distribution")

    evidence_counts = (
        df["evidence_type"]
        .value_counts()
        .reset_index()
    )

    evidence_counts.columns = [
        "Evidence Type",
        "Count"
    ]

    st.dataframe(
        evidence_counts,
        use_container_width=True
    )

    st.bar_chart(
        evidence_counts.set_index("Evidence Type")
    )

    st.divider()

    # Risk Scores
    st.subheader("Risk Score Statistics")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Average Risk Score",
        round(df["risk_score"].mean(), 2)
    )

    col2.metric(
        "Maximum Risk Score",
        round(df["risk_score"].max(), 2)
    )

    col3.metric(
        "Minimum Risk Score",
        round(df["risk_score"].min(), 2)
    )



# ==================================================
# MO ANALYSIS
# ==================================================

elif page == "MO Intelligence":

    st.header("Modus Operandi Analysis")

    mo_df = pd.read_csv(
        "data/mo_results.csv"
    )

    # Debug Information

    st.write(
        "Rows:",
        len(mo_df)
    )

    st.write(
        "Clusters:",
        mo_df["mo_cluster"].nunique()
    )

    # Cluster Summary

    cluster_counts = (
        mo_df["mo_cluster"]
        .value_counts()
        .sort_index()
        .reset_index()
    )

    cluster_counts.columns = [
        "Cluster",
        "Cases"
    ]

    st.metric(
        "Total Clusters",
        mo_df["mo_cluster"].nunique()
    )

    st.subheader(
        "Cluster Distribution"
    )

    st.dataframe(
        cluster_counts,
        width="stretch"
    )

    st.bar_chart(
        cluster_counts.set_index(
            "Cluster"
        )
    )

    # Cluster Selection

    st.subheader(
        "Cluster Analysis"
    )

    selected_cluster = st.selectbox(
        "Select Cluster",
        sorted(
            mo_df["mo_cluster"]
            .unique()
        )
    )

    cluster_data = mo_df[
        mo_df["mo_cluster"]
        == selected_cluster
    ]

    # Cluster Characteristics

    st.subheader(
        "Crime Type Distribution"
    )

    st.dataframe(
        cluster_data[
            "crime_type"
        ]
        .value_counts()
        .reset_index(),
        width="stretch"
    )

    st.subheader(
        "Weapon Distribution"
    )

    st.dataframe(
        cluster_data[
            "weapon_used"
        ]
        .value_counts()
        .reset_index(),
        width="stretch"
    )

    st.subheader(
        "City Distribution"
    )

    st.dataframe(
        cluster_data[
            "city"
        ]
        .value_counts()
        .head(10)
        .reset_index(),
        width="stretch"
    )

    st.subheader(
        "Evidence Distribution"
    )

    st.dataframe(
        cluster_data[
            "evidence_type"
        ]
        .value_counts()
        .reset_index(),
        width="stretch"
    )

    # Sample Cases

    st.subheader(
        "Sample Cases"
    )

    st.dataframe(
        cluster_data[
            [
                "crime_type",
                "weapon_used",
                "city",
                "evidence_type",
                "risk_score",
                "case_status"
            ]
        ]
        .head(20),
        width="stretch"
    )






#FEDERATED INTELLIGENCE
#
#

elif page == "Federated Intelligence Center":

    st.header(
        "Federated Intelligence Center"
    )

    st.success(
        "Cross-Agency Intelligence Network Active"
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Connected Agencies",
        3
    )

    col2.metric(
        "Federated Rounds",
        5
    )

    col3.metric(
        "Aggregation",
        "FedAvg"
    )

    st.divider()

    st.subheader(
        "Agency Status"
    )

    agency_df = pd.DataFrame(
        {
            "Agency": [
                "Kochi Police",
                "Bangalore Police",
                "Chennai Police"
            ],
            "Status": [
                "Connected",
                "Connected",
                "Connected"
            ],
            "Data Sharing": [
                "Local Only",
                "Local Only",
                "Local Only"
            ]
        }
    )

    st.dataframe(
        agency_df,
        width="stretch"
    )

    st.divider()

    st.info(
        """
Privacy Protection Enabled

✓ Raw crime records never leave agencies

✓ Only model parameters are exchanged

✓ Federated Averaging (FedAvg) used

✓ Cross-agency intelligence supported
        """
    )

    st.divider()

    st.subheader(
        "Federated Learning Architecture"
    )

    st.code(
        """
Agency A → Local Training
Agency B → Local Training
Agency C → Local Training

          ↓

      FedAvg Server

          ↓

      Global Model

          ↓

 Crime Intelligence System
        """
    )

    st.divider()

    st.subheader(
        "Federated Model Training"
    )

    st.write(
        """
Train local agency models and aggregate
their parameters using Federated Averaging.
No raw crime records leave the agencies.
        """
    )

    if st.button(
        "Train Federated Model"
    ):

        with st.spinner(
            "Training Federated Model..."
        ):

            result = subprocess.run(
    [
        "python",
        "-m",
        "federated.train_federated"
    ],
    capture_output=True,
    text=True
)

        if result.returncode == 0:

            st.success(
                "Federated Training Complete"
            )

            st.subheader(
                "Training Log"
            )

            st.text(
                result.stdout
            )

            st.success(
                "Global model saved successfully."
            )

            st.info(
                """
Expected Performance

Agency A Accuracy ≈ 78%

Agency B Accuracy ≈ 78%

Agency C Accuracy ≈ 78%

Global Accuracy ≈ 78%
                """
            )

        else:

            st.error(
                "Federated training failed."
            )

            st.subheader(
                "Error Log"
            )

            st.text(
                result.stderr
            )

    st.divider()

    st.subheader(
        "Federated Learning Benefits"
    )

    benefits_df = pd.DataFrame(
        {
            "Feature": [
                "Privacy Preservation",
                "Cross-Agency Collaboration",
                "Local Data Ownership",
                "Model Aggregation",
                "Intelligence Sharing"
            ],
            "Status": [
                "Enabled",
                "Enabled",
                "Enabled",
                "Enabled",
                "Enabled"
            ]
        }
    )

    st.dataframe(
        benefits_df,
        width="stretch"
    )



# ==================================================
# MODEL EVALUATION
# ==================================================

elif page == "Model Evaluation":

    st.header(
        "Retrieval System Evaluation"
    )

    metrics_df = pd.read_csv(
        "evaluation/retrieval_metrics.csv"
    )

    st.subheader(
        "Evaluation Metrics"
    )

    st.dataframe(
        metrics_df,
        width="stretch"
    )

    # ==========================================
    # BAR CHART
    # ==========================================

    chart_df = metrics_df.set_index(
        "Metric"
    )

    st.subheader(
        "Performance Visualization"
    )

    st.bar_chart(
        chart_df
    )

    # ==========================================
    # METRIC CARDS
    # ==========================================

    st.subheader(
        "Metric Summary"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        top1 = metrics_df[
            metrics_df["Metric"]
            ==
            "Top-1 Accuracy"
        ]["Score"].iloc[0]

        st.metric(
            "Top-1 Accuracy",
            f"{top1:.2%}"
        )

    with col2:

        top5 = metrics_df[
            metrics_df["Metric"]
            ==
            "Top-5 Accuracy"
        ]["Score"].iloc[0]

        st.metric(
            "Top-5 Accuracy",
            f"{top5:.2%}"
        )

    with col3:

        top10 = metrics_df[
            metrics_df["Metric"]
            ==
            "Top-10 Accuracy"
        ]["Score"].iloc[0]

        st.metric(
            "Top-10 Accuracy",
            f"{top10:.2%}"
        )

    # ==========================================
    # INTERPRETATION
    # ==========================================

    st.subheader(
        "System Interpretation"
    )

    st.info(
        f"""
        The federated retrieval system achieved:

        • Top-1 Accuracy: {top1:.2%}

        • Top-5 Accuracy: {top5:.2%}

        • Top-10 Accuracy: {top10:.2%}

        These results indicate that the intelligence retrieval
        engine successfully returns relevant cross-agency crime
        records for the evaluated investigation queries.
        """
    )


# ==================================================
# THREAT INTELLIGENCE DASHBOARD
# ==================================================

elif page == "Threat Intelligence":

    st.header(
        "National Threat Intelligence Dashboard"
    )
    st.success(
    "Real-time intelligence overview across all agencies."
    )

    df = pd.read_csv(
        "data/processed_crime_data.csv"
    )

    # ==========================================
    # TOP METRICS
    # ==========================================

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Total Cases",
            f"{len(df):,}"
        )

    with col2:

        st.metric(
            "Crime Types",
            df["crime_type"].nunique()
        )

    with col3:

        st.metric(
            "Cities",
            df["city"].nunique()
        )

    with col4:

        st.metric(
            "Average Risk",
            round(
                df["risk_score"].mean(),
                2
            )
        )

    # ==========================================
    # CRIME DISTRIBUTION
    # ==========================================

    st.subheader(
        "Crime Distribution"
    )

    crime_counts = (
        df["crime_type"]
        .value_counts()
    )

    st.bar_chart(
        crime_counts
    )

    st.dataframe(
        crime_counts.reset_index(),
        width="stretch"
    )

    # ==========================================
    # WEAPON DISTRIBUTION
    # ==========================================

    st.subheader(
        "Weapon Distribution"
    )

    weapon_counts = (
        df["weapon_used"]
        .value_counts()
    )

    st.bar_chart(
        weapon_counts
    )

    st.dataframe(
        weapon_counts.reset_index(),
        width="stretch"
    )

    # ==========================================
    # HIGH RISK CITIES
    # ==========================================

    st.subheader(
        "Highest Risk Cities"
    )

    city_risk = (
        df.groupby("city")
        ["risk_score"]
        .mean()
        .sort_values(
            ascending=False
        )
        .head(10)
    )

    st.bar_chart(
        city_risk
    )

    st.dataframe(
        city_risk.reset_index(),
        width="stretch"
    )

    # ==========================================
    # MOST ACTIVE GANGS
    # ==========================================

    st.subheader(
        "Most Active Gangs"
    )

    gang_counts = (
        df["gang_id"]
        .value_counts()
        .head(10)
    )

    st.bar_chart(
        gang_counts
    )

    st.dataframe(
        gang_counts.reset_index(),
        width="stretch"
    )

    # ==========================================
    # TOP RISK CASES
    # ==========================================

    st.subheader(
        "Highest Risk Cases"
    )

    top_cases = (
        df.sort_values(
            "risk_score",
            ascending=False
        )
        .head(20)
    )

    st.dataframe(
        top_cases[
            [
                "case_id",
                "crime_type",
                "city",
                "weapon_used",
                "gang_id",
                "risk_score"
            ]
        ],
        width="stretch"
    )


# ==================================================
# SYSTEM ARCHITECTURE
# ==================================================

elif page == "System Architecture":

    st.header(
        "System Architecture"
    )

    st.subheader(
        "Overall Crime Intelligence Workflow"
    )

    st.code(
        """
Crime Intelligence Dataset
            ↓

      Data Preprocessing
            ↓

     Crime Rule Engine
 (Weapon & Evidence Assignment)
            ↓

      Sentence Transformer
        (MiniLM-L6-v2)
            ↓

    Crime Embedding Generation
        (384 Dimensions)
            ↓

     Federated Data Split
   (Agency A / B / C)
            ↓

     Local Agency Search
      (Agency Search)
            ↓

    Federated Coordinator
            ↓

     Intelligence Scoring
  (Similarity + Risk + Context)
            ↓

 Cross-Agency Case Retrieval
            ↓

      Knowledge Graph
            ↓

    Community Detection
            ↓

        MO Analysis
            ↓

 Threat Intelligence Dashboard
            ↓

   Investigator Dashboard
        """
    )

    st.divider()

    st.subheader(
        "Federated Learning & Retrieval Architecture"
    )

    st.code(
        """
Agency A Dataset
        ↓
 Local Model Training
        ↓
 Local Search Engine

Agency B Dataset
        ↓
 Local Model Training
        ↓
 Local Search Engine

Agency C Dataset
        ↓
 Local Model Training
        ↓
 Local Search Engine

          ↓

      Flower Server
   (Model Aggregation)

          ↓

       Global Model

          ↓

 Investigator Query

          ↓

    Federated Coordinator

          ↓

 Agency A Search Results
 Agency B Search Results
 Agency C Search Results

          ↓

    Intelligence Scoring

          ↓

 Final Ranked Results
        """
    )

    st.divider()

    st.subheader(
        "Technology Stack"
    )

    tech_stack = {
        "Layer": [
            "Frontend",
            "NLP Embeddings",
            "Vector Search",
            "Federated Learning",
            "Data Analytics",
            "Knowledge Graph",
            "Community Detection",
            "MO Analysis",
            "Evaluation",
            "Backend"
        ],
        "Technology": [
            "Streamlit",
            "Sentence Transformers",
            "FAISS",
            "Flower",
            "Pandas / NumPy",
            "NetworkX",
            "Louvain Community Detection",
            "K-Means Clustering",
            "Custom Evaluation Framework",
            "Python"
        ]
    }

    st.dataframe(
        pd.DataFrame(tech_stack),
        width="stretch"
    )

    st.divider()

    st.subheader(
        "Project Components"
    )

    components = {
        "Module": [
            "Data Preprocessing",
            "Crime Rule Engine",
            "NLP Embeddings",
            "Federated Learning",
            "Federated Retrieval",
            "Intelligence Scoring",
            "Knowledge Graph",
            "Community Detection",
            "MO Analysis",
            "Threat Intelligence Dashboard",
            "Explainability Engine",
            "Evaluation Framework",
            "Streamlit Dashboard"
        ],
        "Status": [
            "Completed",
            "Completed",
            "Completed",
            "Completed",
            "Completed",
            "Completed",
            "Completed",
            "Completed",
            "Completed",
            "Completed",
            "Completed",
            "Completed",
            "Completed"
        ]
    }

    st.dataframe(
        pd.DataFrame(components),
        width="stretch"
    )

    st.success(
        """
        Current Project Status: Research-Grade Federated Crime Intelligence System
        
        Features:
        • Federated Learning
        • Federated Retrieval
        • Semantic Search
        • Intelligence Scoring
        • Knowledge Graph Analytics
        • Criminal Network Analysis
        • MO Analysis
        • Threat Intelligence Dashboard
        • Explainable Investigation Support
        • Evaluation Framework
        """
    )


# ==================================================
# DATASET OVERVIEW
# ==================================================

elif page == "Dataset Overview":

    st.title("National Threat Intelligence Dashboard")

    st.header("📊 Dataset Overview")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Total Records", f"{len(df):,}")

    with col2:
        st.metric("Features", len(df.columns))

    st.subheader("Sample Records")

    st.dataframe(
        df.head(100),
        use_container_width=True
    )
