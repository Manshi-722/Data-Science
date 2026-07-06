#!/usr/bin/env python3
"""
Simple CLI for the RAG system.

Build an index from a folder of documents, then ask questions against it.

Usage:
    # Build an index from all supported files in a folder
    python cli.py build --folder sample_docs --index storage/index.pkl

    # Ask a single question
    python cli.py ask --index storage/index.pkl --question "What is RAG?"

    # Interactive Q&A loop
    python cli.py chat --index storage/index.pkl
"""
import argparse
import json

from rag.pipeline import RAGPipeline


def cmd_build(args):
    pipeline = RAGPipeline(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        embedding_backend=args.backend,
    )
    n = pipeline.ingest_folder(args.folder)
    pipeline.save(args.index)
    print(f"Indexed {n} chunks from '{args.folder}' -> saved to '{args.index}'")


def cmd_ask(args):
    pipeline = RAGPipeline()
    pipeline.load(args.index)
    result = pipeline.ask(args.question, top_k=args.top_k)
    print("\nQuestion:", result["question"])
    print("\nAnswer:\n", result["answer"])
    print("\nSources:")
    for s in result["sources"]:
        print(f"  - {s['source']} (chunk {s['chunk_index']}, score={s['score']:.3f})")
    if args.json:
        print("\n" + json.dumps(result, indent=2))


def cmd_chat(args):
    pipeline = RAGPipeline()
    pipeline.load(args.index)
    print("RAG chat ready. Type 'exit' to quit.")
    while True:
        question = input("\n> ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue
        result = pipeline.ask(question, top_k=args.top_k)
        print("\n" + result["answer"])
        srcs = ", ".join(f"{s['source']}#{s['chunk_index']}" for s in result["sources"])
        print(f"\n(sources: {srcs})")


def main():
    parser = argparse.ArgumentParser(description="Simple RAG (Retrieval-Augmented Generation) CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build", help="Build a vector index from a folder of documents")
    p_build.add_argument("--folder", required=True, help="Folder containing .pdf/.docx/.txt files")
    p_build.add_argument("--index", default="storage/index.pkl", help="Path to save the index")
    p_build.add_argument("--chunk-size", type=int, default=800)
    p_build.add_argument("--chunk-overlap", type=int, default=150)
    p_build.add_argument(
        "--backend", default="tfidf", choices=["tfidf", "sentence-transformers"],
        help="Embedding backend: 'tfidf' works fully offline; "
             "'sentence-transformers' gives better semantic retrieval but needs internet the first time.",
    )
    p_build.set_defaults(func=cmd_build)

    p_ask = sub.add_parser("ask", help="Ask a single question against an index")
    p_ask.add_argument("--index", default="storage/index.pkl")
    p_ask.add_argument("--question", required=True)
    p_ask.add_argument("--top-k", type=int, default=4)
    p_ask.add_argument("--json", action="store_true")
    p_ask.set_defaults(func=cmd_ask)

    p_chat = sub.add_parser("chat", help="Interactive question-answering loop")
    p_chat.add_argument("--index", default="storage/index.pkl")
    p_chat.add_argument("--top-k", type=int, default=4)
    p_chat.set_defaults(func=cmd_chat)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
