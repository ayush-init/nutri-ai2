"""
FoodLens AI - Full FoodSeg103 Dataset Preparer
Prepares all 103 food classes into YOLO format for GPU training.
"""

from datasets import load_dataset
from pathlib import Path
import random
import yaml
import sys
import os

def main():
    print("=" * 65)
    print("🍕 FoodLens AI - Full FoodSeg103 Dataset Pipeline (103 Classes)")
    print("=" * 65)

    OUTPUT_DIR = Path("datasets/food_yolo_103")

    for split in ["train", "val", "test"]:
        (OUTPUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    print("\n[Step 1/5] Downloading & Loading pictograph/foodseg103 from Hugging Face...")
    dataset = load_dataset("pictograph/foodseg103", split="train")
    print(f"✓ Dataset loaded successfully! Total images available: {len(dataset)}")

    print("\n[Step 2/5] Extracting & Indexing all 103 Food Categories...")
    all_categories = set()
    for sample in dataset:
        for name in sample["objects"]["category_names"]:
            if name and name.strip():
                all_categories.add(name.strip())

    sorted_classes = sorted(list(all_categories))
    CLASS_TO_ID = {name: idx for idx, name in enumerate(sorted_classes)}
    print(f"✓ Discovered {len(sorted_classes)} distinct food classes.")
    print("Top classes sample:", sorted_classes[:15])

    print("\n[Step 3/5] Splitting into Train (80%), Validation (10%), and Test (10%)...")
    total_images = len(dataset)
    indices = list(range(total_images))
    random.seed(42)
    random.shuffle(indices)

    train_end = int(total_images * 0.8)
    val_end = int(total_images * 0.9)

    splits = {
        "train": indices[:train_end],
        "val": indices[train_end:val_end],
        "test": indices[val_end:]
    }

    print(f"  • Train set: {len(splits['train'])} images")
    print(f"  • Val set:   {len(splits['val'])} images")
    print(f"  • Test set:  {len(splits['test'])} images")

    def convert_bbox_to_yolo(bbox, img_w, img_h):
        x, y, w, h = bbox
        x_center = (x + w / 2.0) / img_w
        y_center = (y + h / 2.0) / img_h
        norm_w = w / img_w
        norm_h = h / img_h
        return max(0.0, min(1.0, x_center)), max(0.0, min(1.0, y_center)), max(0.0, min(1.0, norm_w)), max(0.0, min(1.0, norm_h))

    print("\n[Step 4/5] Converting bounding boxes & saving images to disk...")
    for split_name, split_indices in splits.items():
        print(f"\nProcessing {split_name.upper()} split ({len(split_indices)} images)...")
        saved_count = 0

        for count, idx in enumerate(split_indices):
            sample = dataset[idx]
            image = sample["image"].convert("RGB")
            img_w, img_h = image.size

            objects = sample["objects"]
            boxes = objects["bbox"]
            cat_names = objects["category_names"]

            annotations = []
            for bbox, food_name in zip(boxes, cat_names):
                food_name = food_name.strip()
                if food_name not in CLASS_TO_ID:
                    continue
                cid = CLASS_TO_ID[food_name]
                xc, yc, nw, nh = convert_bbox_to_yolo(bbox, img_w, img_h)
                annotations.append(f"{cid} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")

            if not annotations:
                continue

            img_filename = f"food103_{idx:05d}.jpg"
            lbl_filename = f"food103_{idx:05d}.txt"

            image.save(OUTPUT_DIR / "images" / split_name / img_filename, quality=90)
            with open(OUTPUT_DIR / "labels" / split_name / lbl_filename, "w", encoding="utf-8") as lf:
                lf.write("\n".join(annotations))

            saved_count += 1
            if (count + 1) % 500 == 0:
                print(f"  Processed {count + 1}/{len(split_indices)} images...")

        print(f"✓ Finished {split_name.upper()}: Saved {saved_count} annotated images.")

    print("\n[Step 5/5] Generating data.yaml with all 103 class names...")
    yaml_dict = {
        "path": str(OUTPUT_DIR.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {idx: name for idx, name in enumerate(sorted_classes)}
    }

    yaml_path = OUTPUT_DIR / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as yf:
        yaml.dump(yaml_dict, yf, default_flow_style=False, sort_keys=False)

    print(f"✓ data.yaml created at: {yaml_path.resolve()}")
    print("\n" + "=" * 65)
    print("🎉 FULL 103-CLASS DATASET READY FOR TRAINING!")
    print(f"📁 Dataset Path: {OUTPUT_DIR.resolve()}")
    print("=" * 65)

if __name__ == "__main__":
    main()
