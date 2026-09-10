import chromadb  # Vector database
from sentence_transformers import SentenceTransformer

# Initialize the embedding model (runs locally)
embedder = SentenceTransformer('all-MiniLM-L6-v2')
client = chromadb.PersistentClient(path="memory/chroma_db")

def store_conversation_summary(summary):
    """Stores a summary of a conversation for later retrieval."""
    collection = client.get_or_create_collection("memories")
    embedding = embedder.encode(summary).tolist()
    collection.add(
        embeddings=[embedding],
        documents=[summary],
        ids=[f"mem_{len(collection.get()['ids'])}"]
    )

def retrieve_relevant_memories(query, n_results=3):
    """Finds past conversations relevant to the current query."""
    collection = client.get_collection("memories")
    embedding = embedder.encode(query).tolist()
    results = collection.query(
        query_embeddings=[embedding],
        n_results=n_results
    )
    return results['documents'][0]