import pandas as pd
import numpy as np

from src.agency_search import (
    search_filtered_agency
)

agency_c_df = pd.read_csv(
    "data/agency_c.csv"
)

agency_c_X = np.load(
    "data/agency_c_X.npy"
).astype("float32")


def search_agency_c(

    query_embedding,

    query_crime=None,
    query_weapon=None,
    query_city=None,

    k=50

):

    return search_filtered_agency(

        query_embedding,

        agency_c_X,
        agency_c_df,

        "Agency C",

        query_crime,
        query_weapon,
        query_city,

        k

    )
