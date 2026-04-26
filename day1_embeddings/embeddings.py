import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from snippets import snippets

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def dot_product(a: list[float], b: list[float]) -> float:
    """
    Multiply corresponding elements and sum them.
    This is the numerator of the cosine similarity formula.
    """
    return sum(x * y for x, y in zip(a, b))


def magnitude(v: list[float]) -> float:
    """
    The length of a vector: square root of the sum of squared elements.
    This is ||v|| in the formula.
    """
    return sum(x ** 2 for x in v) ** 0.5


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """
    How similar are two vectors in terms of direction?
    Result: 1.0 = identical direction, 0.0 = orthogonal, -1.0 = opposite
    """
    return dot_product(a, b) / (magnitude(a) * magnitude(b))

def get_embeddings(texts: list[str]) -> list[list[float]]:
    """
    Send a list of texts to OpenAI and return a list of embedding vectors.
    The API accepts multiple inputs in one call — more efficient than looping.
    """
    response = client.embeddings.create(
        model="text-embedding-3-small",  # 1536 dimensions, cheap, fast
        input=texts
    )

    # response.data is a list of EmbeddingObject, each has an .embedding attribute
    # We sort by index to guarantee order matches input order
    embeddings = [item.embedding for item in sorted(response.data, key=lambda x: x.index)]
    return embeddings

def find_most_similar(query: str, snippets: list[str], snippet_vectors: list[list[float]]) -> None:
    """
    Embed the query, compute cosine similarity against all snippets,
    then print them ranked from most to least similar.
    """
    print(f"\nQuery: '{query}'")
    print("-" * 60)

    # Embed just the query — same API call, single string
    query_vector = get_embeddings([query])[0]

    # Compute similarity score for every snippet
    scored = []
    for i, (snippet, vector) in enumerate(zip(snippets, snippet_vectors)):
        score = cosine_similarity(query_vector, vector)
        scored.append((score, i, snippet))

    # Sort descending — highest similarity first
    scored.sort(reverse=True)

    for rank, (score, idx, snippet) in enumerate(scored, 1):
        print(f"{rank}. [{score:.4f}] {snippet[:80]}")

def save_embeddings(vectors: list[list[float]], path: str = "embeddings_cache.json") -> None:
    with open(path, "w") as f:
        json.dump(vectors, f)
    print(f"Saved {len(vectors)} vectors to {path}")


def load_embeddings(path: str = "embeddings_cache.json") -> list[list[float]] | None:
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)

if __name__ == "__main__":

    # Check cache FIRST — only call the API if no cache exists
    vectors = load_embeddings()
    if vectors is None:
        print(f"No cache found. Generating embeddings for {len(snippets)} snippets...")
        vectors = get_embeddings(snippets)
        save_embeddings(vectors)
    else:
        print("Loaded embeddings from cache.")

    print(f"Each vector has {len(vectors[0])} dimensions.\n")

    queries = [
        "How do deep learning models work?",
        "What software do VFX artists use?",
        "How should I cook Italian food?",
        "How does interest work in investing?",
    ]

    for query in queries:
        find_most_similar(query, snippets, vectors)
