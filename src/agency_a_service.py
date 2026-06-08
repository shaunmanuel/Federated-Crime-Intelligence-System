import pandas as pd
import numpy as np

from src.agency_search import (
    search_filtered_agency
)

# ==========================================
# LOAD LOCAL DATA
# ==========================================

agency_a_df = pd.read_csv(
    "data/agency_a.csv"
)

agency_a_X = np.load(
    "data/agency_a_X.npy"
).astype("float32")


# ==========================================
# LOCAL SEARCH SERVICE
# ==========================================

def search_agency_a(

    query_embedding,

    query_crime=None,
    query_weapon=None,
    query_city=None,

    k=50

):

    return search_filtered_agency(

        query_embedding,

        agency_a_X,
        agency_a_df,

        "Agency A",

        query_crime,
        query_weapon,
        query_city,

        k

    )
