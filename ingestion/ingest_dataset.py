from tqdm import tqdm
from features.color_extractor import extract_color_vector
from db.chroma_client import get_chroma_collection
from utils.file_utils import get_all_images

from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import uuid

# ---- CONFIG ----
BATCH_SIZE = 256
NUM_WORKERS = max(1, multiprocessing.cpu_count() - 1)


# ---- WORKER FUNCTION (IMPORTANT: must be top-level) ----
def process_image(data):
    img_path, family = data
    vec = extract_color_vector(img_path)

    if vec is None:
        return None

    return {
        "id": str(uuid.uuid4()),
        "embedding": vec,
        "metadata": {
            "family": family,
            "path": str(img_path)
        }
    }


# ---- MAIN INGEST FUNCTION ----
def ingest_dataset(parent_folder):
    collection = get_chroma_collection()

    data = get_all_images(parent_folder)
    print(f"Total images found: {len(data)}")
    print(f"Using {NUM_WORKERS} workers 🚀")

    ids, embeddings, metadatas = [], [], []

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [executor.submit(process_image, item) for item in data]

        for future in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()

            if result is None:
                continue

            ids.append(result["id"])
            embeddings.append(result["embedding"])
            metadatas.append(result["metadata"])

            # ---- BATCH INSERT ----
            if len(ids) >= BATCH_SIZE:
                collection.add(
                    ids=ids,
                    embeddings=embeddings,
                    metadatas=metadatas
                )
                ids, embeddings, metadatas = [], [], []

    # ---- FINAL FLUSH ----
    if ids:
        collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas
        )

    print("✅ Parallel ingestion completed")