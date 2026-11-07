"""
Ingest PDFs: load -> split->embed ->store in chroma
 Run from the paper-pilot folder: uv run python -m paper_pilot.ingest

"""

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from paper_pilot.config import get_settings
from paper_pilot.vectorstore import vector_store


def load_papers(paper_dir: Path) -> list[Document]:
    """Load every PDF as one document per page, with clean metadata"""

    pages: list[Document] = []

    for pdf in sorted(paper_dir.glob("*.pdf")):
        year_text, _, slug = pdf.stem.partition("_")
        if not year_text.isdigit() or not slug:
            raise ValueError(f"Rename {pdf.name} to YEAR_title.pdf, e.g: 2017_attention.pdf")

        for page in PyPDFLoader(str(pdf)).load():
            page.metadata = {
                "paper": slug.replace("-", " ").title(),
                "year": int(year_text),
                "page": page.metadata["page"] + 1,  # PyPDF counts from 0, humans from 1
                "source": pdf.name,
            }
            pages.append(page)
    return pages


def split_pages(pages: list[Document], chunk_size: int, chunk_overlap: int) -> list[Document]:
    """Cuts pages into overlapping chunks; each chunk keep its page's metadata"""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(pages)


def chunk_ids(chunks: list[Document]) -> list[str]:
    """Stable IDs like '2017_attention.pdf:p7:c2' so re-running replaces, never duplicates"""
    ids: list[str] = []
    seen: dict[tuple[str, int], int] = {}
    for chunk in chunks:
        key = (chunk.metadata["source"], chunk.metadata["page"])
        seen[key] = seen.get(key, 0) + 1
        ids.append(f"{key[0]}:p{key[1]}:c{seen[key]}")
    return ids


def main() -> None:
    s = get_settings()
    pages = load_papers(s.papera_dir)

    if not pages:
        raise SystemExit(f"No PDFs found in {s.papera_dir}/")

    chunks = split_pages(pages, s.chunk_size, s.chunk_overlap)

    store = vector_store(s)
    store.add_documents(chunks, ids=chunk_ids(chunks))

    print(f"{len(pages)} pages -> {len(chunks)} chunks")
    print(f"Collection now holds {len(store.get()['ids'])} chunks in {s.chroma_dir}/")


if __name__ == "__main__":
    main()
