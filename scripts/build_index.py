"""Phase 1 build step: chunk -> embed -> store (Chroma) + build BM25 index.

Run:  python scripts/build_index.py
Done when: the knowledge base is indexed and ready to query.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from src import chunking, keyword_index, vectorstore  # noqa: E402


def main() -> None:
    chunks = chunking.load_and_chunk()
    if not chunks:
        print(f"No documents found in {config.DOCS_DIR}.")
        print("Add .md / .txt / .pdf fact sheets there, then rerun.")
        return

    docs = {c.doc_id for c in chunks}
    print(f"Loaded {len(chunks)} chunks from {len(docs)} document(s).")

    print("Embedding + storing in Chroma (first run downloads the BGE model)...")
    collection = vectorstore.reset_collection()
    vectorstore.add_chunks(collection, chunks)

    print("Building BM25 keyword index...")
    keyword_index.build(chunks)

    print(f"Done. Indexed {collection.count()} chunks. Query with scripts/query.py")


if __name__ == "__main__":
    main()
