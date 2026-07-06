"""
Document loading utilities.
Supports .pdf, .docx, and .txt files, returning raw text per document.
"""
from pathlib import Path
from typing import List, Dict


def load_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def load_pdf(path: Path) -> str:
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    text_parts = []
    for page in reader.pages:
        text_parts.append(page.extract_text() or "")
    return "\n".join(text_parts)


def load_docx(path: Path) -> str:
    import docx
    document = docx.Document(str(path))
    return "\n".join(p.text for p in document.paragraphs)


LOADERS = {
    ".txt": load_txt,
    ".md": load_txt,
    ".pdf": load_pdf,
    ".docx": load_docx,
}


def load_document(path: str) -> Dict:
    """Load a single document, returning {source, text}."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"No such file: {path}")
    ext = p.suffix.lower()
    if ext not in LOADERS:
        raise ValueError(f"Unsupported file type '{ext}' for {path}")
    text = LOADERS[ext](p)
    return {"source": p.name, "text": text}


def load_documents(paths: List[str]) -> List[Dict]:
    """Load multiple documents. Skips files that fail to load, with a warning."""
    docs = []
    for path in paths:
        try:
            docs.append(load_document(path))
        except Exception as e:
            print(f"[loader] Warning: could not load {path}: {e}")
    return docs


def load_folder(folder: str) -> List[Dict]:
    """Load every supported file directly inside a folder."""
    p = Path(folder)
    paths = [str(f) for f in sorted(p.iterdir()) if f.suffix.lower() in LOADERS]
    return load_documents(paths)
