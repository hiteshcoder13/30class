from collections import Counter
from os import name
from features.color_extractor import extract_color_vector
from db.chroma_client import get_chroma_collection

def query_image(image_path, top_k=200):
    """
    top_k = number of images retrieved
    later we aggregate → top 20 families
    """

    collection = get_chroma_collection()

    query_vec = extract_color_vector(image_path)

    if query_vec is None:
        print("❌ Could not extract features")
        return []

    results = collection.query(
        query_embeddings=[query_vec],
        n_results=top_k
    )

    families = [meta["family"] for meta in results["metadatas"][0]]

    # Count frequency
    counter = Counter(families)

    # Top 20 families
    top_families = counter.most_common(30)

    return top_families


