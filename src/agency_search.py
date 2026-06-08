import faiss
import numpy as np


def search_filtered_agency(
    query_embedding,
    embeddings,
    df,
    agency_name,
    query_crime=None,
    query_weapon=None,
    query_city=None,
    k=50
):

    filtered_df = df.copy()

    # ==========================================
    # CRIME FILTER
    # ==========================================

    if query_crime:

        filtered_df = filtered_df[
            filtered_df["crime_type"]
            .str.lower()
            ==
            query_crime.lower()
        ]

    # ==========================================
    # WEAPON FILTER
    # ==========================================

    if query_weapon:

        filtered_df = filtered_df[
            filtered_df["weapon_used"]
            .str.lower()
            ==
            query_weapon.lower()
        ]

    # ==========================================
    # CITY FILTER
    # ==========================================

    if query_city:

        filtered_df = filtered_df[
            filtered_df["city"]
            .str.lower()
            ==
            query_city.lower()
        ]

    print(
        f"{agency_name}: "
        f"{len(filtered_df)} filtered cases"
    )

    if len(filtered_df) == 0:

        return []

    # ==========================================
    # FILTERED EMBEDDINGS
    # ==========================================

    filtered_indices = (
        filtered_df.index.to_numpy()
    )

    filtered_embeddings = embeddings[
        filtered_indices
    ]

    # ==========================================
    # LOCAL INDEX
    # ==========================================

    temp_index = faiss.IndexFlatL2(
        filtered_embeddings.shape[1]
    )

    temp_index.add(
        filtered_embeddings
    )

    k = min(
        k,
        len(filtered_df)
    )

    D, I = temp_index.search(
        query_embedding,
        k
    )

    matches = []

    # ==========================================
    # RESULTS
    # ==========================================

    for distance, idx in zip(
        D[0],
        I[0]
    ):

        row = filtered_df.iloc[idx]

        similarity = float(
            np.exp(-distance)
        )

        matches.append(
            {
                "case_id": row["case_id"],
                "agency": agency_name,
                "crime_type": row["crime_type"],
                "city": row["city"],
                "weapon_used": row["weapon_used"],
                "gang_id": row["gang_id"],
                "suspect_id": row["suspect_id"],
                "risk_score": row["risk_score"],
                "similarity_score": round(
                    similarity,
                    4
                )
            }
        )

    return matches
