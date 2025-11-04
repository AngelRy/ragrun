import os
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama

# ===== CONFIG =====
MODEL_PATH = "models/tinyllama/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
INDEX_PATH = "vector_index/docs.index"
META_PATH = "vector_index/metadata.pkl"
TOP_K = 2        # number of chunks to retrieve
# ==================

# Load components
print("🔹 Loading FAISS index and metadata...")
index = faiss.read_index(INDEX_PATH)
with open(META_PATH, "rb") as f:
    metadata = pickle.load(f)

print("🔹 Loading embedding model...")
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

print("🔹 Loading LLM...")
llm = Llama(model_path=MODEL_PATH, n_ctx=1024, n_threads=6)

# --- Retrieval function ---
def retrieve(query, k=TOP_K):
    q_emb = embedder.encode([query])
    D, I = index.search(q_emb, k)
    retrieved_texts = []
    for idx in I[0]:
        meta = metadata[idx]
        topic = meta["topic"]
        file = meta["file"]
        file_path = os.path.join("data_clean", topic, file)
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                retrieved_texts.append(f.read())
    return "\n\n".join(retrieved_texts)

# --- RAG response function ---
def ask_question(query):
    context = retrieve(query)
    prompt = f"""You are a helpful assistant specialized in distance running.
Answer the question based ONLY on the following context.

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:
"""
    response = llm(prompt, max_tokens=256, temperature=0.7, stop=["USER:", "CONTEXT:"])
    print("\n💬", response["choices"][0]["text"].strip(), "\n")

# --- Interactive loop ---
if __name__ == "__main__":
    print("\n📝 TinyLlama RAG chatbot ready. Type 'exit' to quit.")
    while True:
        q = input("\n❓ Ask a question: ")
        if q.lower() in ["exit", "quit"]:
            break
        ask_question(q)

