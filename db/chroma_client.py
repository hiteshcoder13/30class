import chromadb
from config.settings import CHROMA_COLLECTION, CHROMA_DIR

def get_chroma_collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "l2"}  # Euclidean distance
    )

    return collection