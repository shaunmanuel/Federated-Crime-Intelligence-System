import pandas as pd

from src.agency_a_service import (
    search_agency_a
)

from src.agency_b_service import (
    search_agency_b
)

from src.agency_c_service import (
    search_agency_c
)


def federated_query(

    query_embedding,

    query_crime,
    query_weapon,
    query_city,

    per_agency_k=50,
    final_top_k=10

):

    results = []

    # ==========================================
    # AGENCY A
    # ==========================================

    results.extend(

        search_agency_a(

            query_embedding,

            query_crime,
            query_weapon,
            query_city,

            per_agency_k

        )

    )

    # ==========================================
    # AGENCY B
    # ==========================================

    results.extend(

        search_agency_b(

            query_embedding,

            query_crime,
            query_weapon,
            query_city,

            per_agency_k

        )

    )

    # ==========================================
    # AGENCY C
    # ==========================================

    results.extend(

        search_agency_c(

            query_embedding,

            query_crime,
            query_weapon,
            query_city,

            per_agency_k

        )

    )

    if len(results) == 0:

        return pd.DataFrame()

    results = pd.DataFrame(
        results
    )

    # ==========================================
    # INTELLIGENCE SCORING
    # ==========================================

    results["risk_component"] = (
        results["risk_score"] / 100
    )

    results["crime_bonus"] = 0

    if query_crime:

        results.loc[
            results["crime_type"]
            .str.lower()
            ==
            query_crime.lower(),
            "crime_bonus"
        ] = 1

    results["weapon_bonus"] = 0

    if query_weapon:

        results.loc[
            results["weapon_used"]
            .str.lower()
            ==
            query_weapon.lower(),
            "weapon_bonus"
        ] = 1

    results["city_bonus"] = 0

    if query_city:

        results.loc[
            results["city"]
            .str.lower()
            ==
            query_city.lower(),
            "city_bonus"
        ] = 1

    results["final_score"] = (

        0.50
        *
        results["similarity_score"]

        +

        0.20
        *
        results["risk_component"]

        +

        0.15
        *
        results["crime_bonus"]

        +

        0.10
        *
        results["weapon_bonus"]

        +

        0.05
        *
        results["city_bonus"]

    )

    results = results.sort_values(
        by="final_score",
        ascending=False
    )

    print(
        "\nTop Federated Matches"
    )

    print(
        results[
            [
                "agency",
                "case_id",
                "crime_type",
                "weapon_used",
                "risk_score",
                "similarity_score",
                "final_score"
            ]
        ].head(final_top_k)
    )

    return results[
        [
            "case_id",
            "agency",
            "crime_type",
            "city",
            "weapon_used",
            "gang_id",
            "suspect_id",
            "risk_score",
            "similarity_score",
            "final_score"
        ]
    ].head(
        final_top_k
    ).reset_index(
        drop=True
    )
