"""
Answer generation.

Primary path: call the Anthropic API (Claude) with the retrieved context,
so the answer is grounded in the actual document chunks.

Fallback path: if no ANTHROPIC_API_KEY is set, fall back to a simple
extractive "answer" (just surfaces the most relevant chunks) so the
pipeline still runs end-to-end without any API key for demo purposes.
"""
import os
from typing import List, Dict


SYSTEM_PROMPT = (
    "You are a precise question-answering assistant. Answer the user's "
    "question using ONLY the provided context. If the context does not "
    "contain enough information to answer, say so clearly instead of "
    "guessing. Keep answers concise and cite which source each fact came "
    "from using the [source] tags provided."
)


def build_prompt(question: str, context_chunks: List[Dict]) -> str:
    context_blocks = []
    for c in context_chunks:
        context_blocks.append(f"[source: {c['source']} | chunk {c['chunk_index']}]\n{c['text']}")
    context_text = "\n\n---\n\n".join(context_blocks)
    return (
        f"Context:\n{context_text}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the context above."
    )


def generate_answer(
    question: str,
    context_chunks: List[Dict],
    model: str = "claude-sonnet-5",
    max_tokens: int = 500,
) -> str:
    if not context_chunks:
        return "I couldn't find any relevant content in the documents to answer this question."

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    prompt = build_prompt(question, context_chunks)

    if api_key:
        from anthropic import Anthropic
        client = Anthropic(api_key=api_key)
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in response.content if block.type == "text")

    # Fallback: no API key available -> extractive summary of top chunks.
    lines = [
        "[No ANTHROPIC_API_KEY set - showing the most relevant retrieved passages instead of a generated answer]",
        "",
    ]
    for c in context_chunks:
        lines.append(f"- ({c['source']}, chunk {c['chunk_index']}, score={c['score']:.3f}): {c['text'][:300]}...")
    return "\n".join(lines)
