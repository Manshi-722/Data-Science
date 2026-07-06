# Document Question Answering System (RAG)

A simple, self-contained Retrieval-Augmented Generation pipeline that answers
questions from your own documents (PDF, DOCX, TXT) — built to match the
architecture in the project brief: **Ingestion → Chunking → Embedding →
Vector Store → Retrieval → Generation**.

## How it works

```
Documents (.pdf/.docx/.txt)
        │
        ▼
1. Document Ingestion   (rag/loader.py)      -- extract raw text
        │
        ▼
2. Text Chunking        (rag/chunker.py)     -- split into overlapping chunks
        │
        ▼
3. Embedding Creation    ┐
4. Vector Database        } (rag/vector_store.py) -- embed chunks, store, search
        │
        ▼ (at query time)
5. Query Processing / Context Retrieval  -- embed question, cosine similarity search
        │
        ▼
6. Answer Generation     (rag/generator.py)  -- Claude generates grounded answer
```

All of this is orchestrated by `rag/pipeline.py` (`RAGPipeline` class).

## Setup

```bash
pip install -r requirements.txt
```

To get real generated answers (not just retrieved passages), set your
Anthropic API key:

```bash
export ANTHROPIC_API_KEY="your-key-here"
```

Without a key, the system still runs end-to-end but falls back to showing
the raw retrieved passages instead of a generated answer.

## Usage

**1. Build an index from a folder of documents**

```bash
python cli.py build --folder sample_docs --index storage/index.pkl
```

Options:
- `--backend tfidf` (default) — works fully offline, no model download.
- `--backend sentence-transformers` — better semantic retrieval quality;
  downloads the `all-MiniLM-L6-v2` model from Hugging Face the first time
  you use it (needs internet access), then caches it locally.
- `--chunk-size` / `--chunk-overlap` — tune chunk size in characters (default 800/150).

**2. Ask a single question**

```bash
python cli.py ask --index storage/index.pkl --question "What is the main idea of the document?"
```

**3. Interactive chat**

```bash
python cli.py chat --index storage/index.pkl
```

## Using your own documents

Drop any `.pdf`, `.docx`, or `.txt` files into a folder and point `build` at it:

```bash
python cli.py build --folder /path/to/your/pdfs --index storage/my_index.pkl
```

## Using it as a library

```python
from rag.pipeline import RAGPipeline

pipeline = RAGPipeline(embedding_backend="tfidf")
pipeline.ingest_folder("sample_docs")
result = pipeline.ask("What is RAG?")

print(result["answer"])
print(result["sources"])
```

## Design notes / possible extensions

- **Embeddings**: TF-IDF is used by default so the system runs anywhere
  with zero setup and no external downloads. Swap to `sentence-transformers`
  for real semantic (meaning-based) retrieval once you have internet access
  to Hugging Face — the rest of the pipeline is unchanged.
- **Vector store**: a lightweight in-memory numpy cosine-similarity index.
  For larger corpora, swap in FAISS, Chroma, or a hosted vector DB behind
  the same `VectorStore` interface.
- **Generation**: uses the Anthropic API (`claude-sonnet-5`). Swap the
  `generate_answer` function in `rag/generator.py` to call any other LLM API.
- **Improvements to try** (as suggested in the project brief): hybrid
  search (keyword + vector), re-ranking retrieved chunks, experimenting
  with chunk size/overlap, and trying different embedding models.
