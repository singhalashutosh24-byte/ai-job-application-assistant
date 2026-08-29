import chromadb

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="resume_examples")


def get_similar_examples(job_role: str, n_results: int = 3) -> list[str]:
    """
    Given a job role, retrieves the most semantically similar 
    example resume bullets from the knowledge base.
    """
    results = collection.query(
        query_texts=[job_role],
        n_results=n_results
    )

    # results["documents"] is a list of lists (one list per query) - 
    # since we only passed one query, we take the first inner list
    similar_bullets = results["documents"][0]

    return similar_bullets


if __name__ == "__main__":
    query = "Backend Engineer with experience in Python, REST APIs, PostgreSQL, Docker, and AWS"
    results = collection.query(
        query_texts=[query],
        n_results=5,
        include=["documents", "distances", "metadatas"]
    )
    for doc, dist, meta in zip(results["documents"][0], results["distances"][0], results["metadatas"][0]):
        print(f"[{meta['role']}] (distance: {dist:.4f}) {doc}")