from sentence_transformers import (
    SentenceTransformer
)

from src.agency_a_service import (
    search_agency_a
)

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

query = "homicide poison"

embedding = model.encode(
    [query]
).astype("float32")

results = search_agency_a(
    embedding,
    "Homicide",
    "Poison",
    None
)

print(results[:5])
