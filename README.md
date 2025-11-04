# RAG Chatbot Demo

A lightweight **Retrieval-Augmented Generation (RAG)** system that integrates local document search with a compact **LLM** (TinyLlama) to deliver relevant, contextual answers.
The project demonstrates how modern language models can be combined with traditional vector search to enable offline, domain-focused AI assistants.

---

### 🔍 Key Features

* **Local FAISS vector store** for efficient semantic retrieval
* **SentenceTransformers** for text embeddings
* **TinyLlama (via Llama.cpp)** for local inference — no API keys required
* **Python-based modular pipeline** for indexing, summarization, and querying

---

### ⚙️ Core Technologies

* Python
* FAISS
* SentenceTransformers / Hugging Face
* Llama.cpp (TinyLlama-1.1B-Chat)

---

### ⚙️ Setup

1. **Create and activate environment**

   ```bash
   conda create -n ragrun python=3.10
   conda activate ragrun
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Download model**

   * Get a TinyLlama model from [Hugging Face](https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF)
   * Place the `.gguf` file in:

     ```
     models/tinyllama/
     ```

4. **Build the FAISS index (optional)**

   ```bash
   python build_index.py
   ```

5. **Run the chatbot**

   ```bash
   python rag_chatbot.py
   ```

---

### 📂 Project Structure

```
ragrun/
├── data/               # Raw and cleaned text data
│   ├── endurance/      
│   ├── recovery/
│   └── ...             
├── index/              # FAISS index and metadata files
├── models/             # Local TinyLlama GGUF model
├── scripts/            # Data collection, cleaning, and summarization scripts
├── rag_chatbot.py      # Main RAG chatbot interface
└── requirements.txt    # Dependencies
```

---

### 🚀 Example Usage

```bash
python rag_chatbot.py
```

When prompted:

```
❓ Ask a question: How does interval training improve endurance?
💬 Answer: ...
```

---

### 📚 Educational Purpose

This repository is developed for **demonstration and portfolio** use — showcasing a minimal, fully local RAG pipeline that can be extended to any custom knowledge base or domain.
