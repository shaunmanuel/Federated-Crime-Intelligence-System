import faiss
import numpy as np
import os

agencies = [
    "agency_a",
    "agency_b",
    "agency_c"
]

os.makedirs(
    "faiss_index",
    exist_ok=True
)

for agency in agencies:

    print(f"\nLoading {agency}...")

    embeddings = np.load(
        f"data/{agency}_X.npy"
    ).astype("float32")

    print(
        "Embedding Shape:",
        embeddings.shape
    )

    print(
        "Embedding Dimension:",
        embeddings.shape[1]
    )

    index = faiss.IndexFlatL2(
        embeddings.shape[1]
    )

    index.add(
        embeddings
    )

    print(
        "Vectors Added:",
        index.ntotal
    )

    faiss.write_index(
        index,
        f"faiss_index/{agency}.index"
    )

    print(
        f"Saved: faiss_index/{agency}.index"
    )

print(
    "\nAll agency indexes created successfully."
)
