import faiss, pickle
from sentence_transformers import SentenceTransformer

# Load model and index
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
index = faiss.read_index("vector_index/docs.index")

with open("vector_index/metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

# Ask your question here
query = "How does interval training improve aerobic capacity?"
query_vec = model.encode([query])

# Search for the top 3 most relevant chunks
D, I = index.search(query_vec, k=3)

print("\n🔍 Top 3 results:\n")
for idx, score in zip(I[0], D[0]):
    print(f"{score:.4f} -> {metadata[idx]}")
