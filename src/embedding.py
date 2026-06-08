import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

print("Loading dataset...")

df = pd.read_csv(
    "data/processed_crime_data.csv"
)

print("Loading model...")

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("Generating embeddings...")

embeddings = model.encode(
    df["combined_text"].tolist(),
    batch_size=64,
    show_progress_bar=True
)

print("Shape:", embeddings.shape)

np.save(
    "models/embeddings.npy",
    embeddings
)

print("Embeddings saved successfully!")
