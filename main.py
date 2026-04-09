from ingestion.ingest_dataset import ingest_dataset
from query.query_image import query_image

if __name__ == "__main__":

    # STEP 1: Ingest dataset (run once)
    parent_folder = "/home/Unthinkable/Downloads/onedrive(mayank) + ssd with reflection/159folder"
    # ingest_dataset(parent_folder)

    # STEP 2: Query
    query_img = "/home/Unthinkable/Documents/poc_stone_color/SI_17207_A-050.jpg"

    results = query_image(query_img)

    print("\n🎯 Top Matching Stone Families:\n")

    for i, (family, count) in enumerate(results, 1):
        print(f"{i}. {family} (score: {count})")