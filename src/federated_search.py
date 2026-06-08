import faiss
import pandas as pd
import numpy as np
#from similarity.similarity_inference import (
#    predict_similarity
#)

from src.federated_coordinator import (
    federated_query
)

# ==========================================
# QUERY FEATURE EXTRACTION
# ==========================================

def extract_query_features(query):

    query = query.lower()

    crime_types = [
        "homicide",
        "robbery",
        "kidnapping",
        "fraud",
        "cyber crime",
        "assault",
        "drug trafficking",
        "theft"
    ]

    weapons = [
        "pistol",
        "knife",
        "poison",
        "iron rod",
        "crowbar",
        "laptop",
        "phishing kit",
        "unknown"
    ]

    cities = [
        "kochi",
        "bangalore",
        "chennai",
        "mumbai",
        "delhi",
        "hyderabad"
    ]

    detected_crime = None
    detected_weapon = None
    detected_city = None

    for crime in crime_types:

        if crime in query:

            detected_crime = crime.title()
            break

    for weapon in weapons:

        if weapon in query:

            detected_weapon = weapon.title()
            break

    for city in cities:

        if city in query:

            detected_city = city.title()
            break

    return (
        detected_crime,
        detected_weapon,
        detected_city
    )






# ==========================================
# FEDERATED SEARCH
# ==========================================

def federated_search(
    query_embedding,
    query_text,
    per_agency_k=50,
    final_top_k=10
):

    query_crime, query_weapon, query_city = (
        extract_query_features(
            query_text
        )
    )

    print(
        "\nDetected Crime:",
        query_crime
    )

    print(
        "Detected Weapon:",
        query_weapon
    )

    print(
        "Detected City:",
        query_city
    )

    results = federated_query(

    query_embedding,

    query_crime,
    query_weapon,
    query_city,

    per_agency_k,
    final_top_k

    )

    if len(results) == 0:

        print(
            "\nNo matching records found."
        )

        return pd.DataFrame()

    results = pd.DataFrame(
        results
    )

    return results
