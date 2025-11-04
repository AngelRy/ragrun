import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# Directories
input_dir = "data_clean"
index_dir = "vector_index"
os.makedirs(index_dir, exist_ok=True)

# Model for embeddings (small, fast, works offline)
model_name = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(model_name)

texts = []
metadatas = []

# Read all summaries
for topic_folder in os.listdir(input_dir):
    topic_path = os.path.join(input_dir, topic_folder)
    if not os.path.isdir(topic_path):
        continue

    for file in os.listdir(topic_path):
        if not file.endswith(".txt"):
            continue

        file_path = os.path.join(topic_path, file)
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        texts.append(text)
        metadatas.append({"topic": topic_folder, "file": file})

# Convert texts to embeddings
print(f"🔢 Encoding {len(texts)} documents ...")
embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

# Build FAISS index
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings)

# Save everything
faiss.write_index(index, os.path.join(index_dir, "docs.index"))
with open(os.path.join(index_dir, "metadata.pkl"), "wb") as f:
    pickle.dump(metadatas, f)

print("✅ FAISS index built and saved.")
