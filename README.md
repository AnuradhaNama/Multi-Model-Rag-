# Multimodal RAG System

A complete end-to-end Multimodal Retrieval-Augmented Generation (RAG) pipeline using:

- Wikipedia ingestion
- Text, image, audio, and video embeddings
- ChromaDB vector storage
- Semantic similarity retrieval
- Dynamic cold-start expansion
- FastAPI backend
- Streamlit frontend

---

# Features

## Ingestion

- Fetches data strictly from Wikipedia
- Extracts:
  - Text
  - Images
  - Audio references
  - Video references

## Processing

- Meaningful text chunking
- Metadata generation:
  - Article title
  - Section
  - Subsection
  - Source URL

## Embeddings

- Text embeddings using Sentence Transformers
- Image semantic embeddings using Gemini
- Audio semantic embeddings using Gemini
- Video semantic embeddings using Gemini

## Vector Database

- ChromaDB persistent vector storage
- Modality-specific collections:
  - text_collection
  - image_collection
  - audio_collection
  - video_collection

## Retrieval

- Embedding-driven semantic retrieval
- Top-k similarity search
- Multimodal retrieval support

## Dynamic Cold Start

- Automatically detects missing knowledge
- Extracts Wikipedia topic
- Performs ingestion dynamically
- Re-runs retrieval automatically

---

# Tech Stack

- Python
- FastAPI
- Streamlit
- ChromaDB
- Sentence Transformers
- Google Gemini API
- BeautifulSoup
- Wikipedia

---

# Project Structure

```text
multimodal_rag/
│
├── app/
│   ├── main.py
│   ├── rag_pipeline.py
│   ├── wikipedia_fetcher.py
│   ├── vector_store.py
│   ├── embedding.py
│   ├── image_embedding.py
│   ├── audio_embedding.py
│   ├── video_embedding.py
│   ├── topic_extractor.py
│   ├── llm.py
│   ├── chunking.py
│   └── logger_config.py
│
├── chroma_db/
├── streamlit_app.py
├── evaluation.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── README.md
└── .env
```

---

# Run Backend

```bash
uvicorn app.main:app --reload
```

---

# Run Frontend

```bash
streamlit run streamlit_app.py
```

---

# Run Evaluation

```bash
python evaluation.py
```

---

# Run Docker

## Build

```bash
docker compose build
```

## Run

```bash
docker compose up
```

---

# Example Queries

## Text

What are tiger habitats

## Images

show tiger images

## Audio

tiger sounds

## Video

tiger videos

---

# Architecture Flow

Query
↓
Check ChromaDB
↓
Cold-start expansion if needed
↓
Wikipedia ingestion
↓
Chunking + metadata
↓
Embedding generation
↓
ChromaDB storage
↓
Semantic similarity retrieval
↓
LLM grounded response

---

# Evaluation Metrics

- Faithfulness
- Context Precision
- Context Recall

---

# Final Status

Complete end-to-end multimodal RAG pipeline with semantic retrieval.