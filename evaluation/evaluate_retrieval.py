import os
import sys

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)

import pandas as pd
import numpy as np

from sentence_transformers import (
    SentenceTransformer
)

from src.federated_search import (
    federated_search
)

# ==========================================
# LOAD MODEL
# ==========================================

print(
    "\nLoading embedding model..."
)

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# ==========================================
# TEST QUERIES
# ==========================================

test_queries = [

    {
        "query": "homicide poison",
        "crime": "Homicide",
        "weapon": "Poison"
    },

    {
        "query": "robbery crowbar",
        "crime": "Robbery",
        "weapon": "Crowbar"
    },

    {
        "query": "kidnapping rope",
        "crime": "Kidnapping",
        "weapon": "Rope"
    },

    {
        "query": "drug trafficking rifle",
        "crime": "Drug Trafficking",
        "weapon": "Rifle"
    },

    {
        "query": "cyber crime laptop",
        "crime": "Cyber Crime",
        "weapon": "Laptop"
    },

    {
        "query": "fraud phishing kit",
        "crime": "Fraud",
        "weapon": "Phishing Kit"
    }

]

# ==========================================
# METRICS
# ==========================================

top1_correct = 0
top5_correct = 0
top10_correct = 0

total_queries = len(
    test_queries
)

# ==========================================
# EVALUATION LOOP
# ==========================================

for test in test_queries:

    query = test["query"]

    expected_crime = test["crime"]

    expected_weapon = test["weapon"]

    print(
        f"\nEvaluating: {query}"
    )

    query_embedding = model.encode(
        [query]
    ).astype(
        "float32"
    )

    results = federated_search(
        query_embedding,
        query,
        final_top_k=10
    )

    if len(results) == 0:

        print(
            "No results found."
        )

        continue

    # --------------------------
    # TOP 1
    # --------------------------

    first_row = results.iloc[0]

    if (
        first_row["crime_type"]
        ==
        expected_crime
        and
        first_row["weapon_used"]
        ==
        expected_weapon
    ):

        top1_correct += 1

    # --------------------------
    # TOP 5
    # --------------------------

    top5 = results.head(5)

    found_top5 = (

        (
            top5["crime_type"]
            ==
            expected_crime
        )

        &

        (
            top5["weapon_used"]
            ==
            expected_weapon
        )

    ).any()

    if found_top5:

        top5_correct += 1

    # --------------------------
    # TOP 10
    # --------------------------

    found_top10 = (

        (
            results["crime_type"]
            ==
            expected_crime
        )

        &

        (
            results["weapon_used"]
            ==
            expected_weapon
        )

    ).any()

    if found_top10:

        top10_correct += 1

# ==========================================
# RESULTS
# ==========================================

top1_accuracy = (
    top1_correct
    /
    total_queries
)

top5_accuracy = (
    top5_correct
    /
    total_queries
)

top10_accuracy = (
    top10_correct
    /
    total_queries
)

metrics_df = pd.DataFrame(

    {
        "Metric": [

            "Top-1 Accuracy",
            "Top-5 Accuracy",
            "Top-10 Accuracy"

        ],

        "Score": [

            round(
                top1_accuracy,
                4
            ),

            round(
                top5_accuracy,
                4
            ),

            round(
                top10_accuracy,
                4
            )

        ]
    }

)

print(
    "\n========================"
)

print(
    "RETRIEVAL EVALUATION"
)

print(
    "========================"
)

print(
    metrics_df
)

# ==========================================
# SAVE RESULTS
# ==========================================

metrics_df.to_csv(
    "evaluation/retrieval_metrics.csv",
    index=False
)

print(
    "\nSaved:"
)

print(
    "evaluation/retrieval_metrics.csv"
)
