import pandas as pd
import numpy as np

from src.agency_search import (
    search_filtered_agency
)

agency_b_df = pd.read_csv(
    "data/agency_b.csv"
)

agency_b_X = np.load(
    "data/agency_b_X.npy"
).astype("float32")


def search_agency_b(

    query_embedding,

    query_crime=None,
    query_weapon=None,
    query_city=None,

    k=50

):

    return search_filtered_agency(

        query_embedding,

        agency_b_X,
        agency_b_df,

        "Agency B",

        query_crime,
        query_weapon,
        query_city,

        k

    )
