from pathlib import Path
import shutil

root = Path(r"C:\Users\amira\Desktop\Local Test\tiny-imagenet-200")
val_dir = root / "val"
images_dir = val_dir / "images"
annotations = val_dir / "val_annotations.txt"

with open(annotations) as f:
    for line in f:
        filename, class_id = line.split("\t")[:2]
        class_dir = val_dir / class_id
        class_dir.mkdir(exist_ok=True)
        src = images_dir / filename
        dst = class_dir / filename
        if src.exists():
            shutil.move(str(src), str(dst))

# Optional: remove empty images directory
if images_dir.exists() and not any(images_dir.iterdir()):
    images_dir.rmdir()

print("Validation folder prepared.")