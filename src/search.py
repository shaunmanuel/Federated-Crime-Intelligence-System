import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer

print("Loading dataset...")
df = pd.read_csv("data/processed_crime_data.csv")

print("Loading model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Loading FAISS index...")
index = faiss.read_index("faiss_index/crime.index")

query = input("\nEnter crime description: ")

query_embedding = model.encode([query]).astype("float32")

D, I = index.search(query_embedding, 5)

print("\n===== TOP 5 SIMILAR CASES =====\n")

for rank, idx in enumerate(I[0], start=1):
    print(f"Result {rank}")
    print(df.iloc[idx][[
        "case_id",
        "crime_type",
        "city",
        "weapon_used",
        "gang_id"
    ]])
    print("-" * 50)
