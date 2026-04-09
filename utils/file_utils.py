from pathlib import Path

def get_all_images(parent_dir):
    parent = Path(parent_dir)

    image_paths = []
    for subfolder in parent.iterdir():
        if subfolder.is_dir():
            for img in subfolder.glob("*.*"):
                if img.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                    image_paths.append((img, subfolder.name))

    return image_paths