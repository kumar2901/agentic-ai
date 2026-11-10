import sys

from paper_pilot.vectorstore import vector_store


def main() -> None:
    question = " ".join(sys.argv[1:]) or "What optimizer did the Transformer use?"
    store = vector_store()

    for doc, distance in store.similarity_search_with_score(question, k=4):
        m = doc.metadata
        print(f"\n--- {m['paper']} ({m['year']}) p.{m['page']} distance {distance:.3f}")
        print(doc.page_content[:400])


if __name__ == "__main__":
    main()
