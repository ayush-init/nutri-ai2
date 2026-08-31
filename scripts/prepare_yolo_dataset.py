from datasets import load_dataset
from pathlib import Path
import random

# --------------------------------------------------
# 1. Classes we want for our MVP
# --------------------------------------------------

SELECTED_CLASSES = [
    "bread",
    "tomato",
    "carrot",
    "chicken_duck",
    "steak",
    "potato",
    "broccoli",
    "ice_cream",
    "strawberry",
    "lettuce",
    "onion",
    "rice",
    "cucumber",
    "pepper",
    "pie",
]

CLASS_TO_ID = {
    name: index
    for index, name in enumerate(SELECTED_CLASSES)
}

# --------------------------------------------------
# 2. Dataset paths
# --------------------------------------------------

OUTPUT_DIR = Path("datasets/food_yolo")

for split in ["train", "val", "test"]:
    (OUTPUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 3. Load dataset
# --------------------------------------------------

print("Loading FoodSeg103...")

dataset = load_dataset(
    "pictograph/foodseg103",
    split="train"
)

print("Dataset loaded!")
print("Total images:", len(dataset))

# --------------------------------------------------
# 4. Find useful images
# --------------------------------------------------

print("\nFiltering images...")

usable_samples = []

for index, sample in enumerate(dataset):

    food_names = sample["objects"]["category_names"]

    if any(food in CLASS_TO_ID for food in food_names):
        usable_samples.append(index)

print("Usable images:", len(usable_samples))

# --------------------------------------------------
# 5. Shuffle and split
# --------------------------------------------------

random.seed(42)
random.shuffle(usable_samples)

total = len(usable_samples)

train_end = int(total * 0.8)
val_end = int(total * 0.9)

train_indices = usable_samples[:train_end]
val_indices = usable_samples[train_end:val_end]
test_indices = usable_samples[val_end:]

splits = {
    "train": train_indices,
    "val": val_indices,
    "test": test_indices,
}

print("\nSplit sizes:")
print("Train:", len(train_indices))
print("Val:", len(val_indices))
print("Test:", len(test_indices))

# --------------------------------------------------
# 6. Convert bounding boxes to YOLO format
# --------------------------------------------------

def convert_bbox_to_yolo(bbox, image_width, image_height):
    x, y, width, height = bbox

    x_center = x + width / 2
    y_center = y + height / 2

    x_center /= image_width
    y_center /= image_height
    width /= image_width
    height /= image_height

    return x_center, y_center, width, height


# --------------------------------------------------
# 7. Save images and labels
# --------------------------------------------------

print("\nCreating YOLO dataset...")

for split_name, indices in splits.items():

    print(f"\nProcessing {split_name}...")

    for counter, dataset_index in enumerate(indices):

        sample = dataset[dataset_index]

        image = sample["image"].convert("RGB")
        image_width, image_height = image.size

        objects = sample["objects"]

        boxes = objects["bbox"]
        food_names = objects["category_names"]

        yolo_annotations = []

        for bbox, food_name in zip(boxes, food_names):

            if food_name not in CLASS_TO_ID:
                continue

            class_id = CLASS_TO_ID[food_name]

            x_center, y_center, width, height = (
                convert_bbox_to_yolo(
                    bbox,
                    image_width,
                    image_height
                )
            )

            annotation = (
                f"{class_id} "
                f"{x_center:.6f} "
                f"{y_center:.6f} "
                f"{width:.6f} "
                f"{height:.6f}"
            )

            yolo_annotations.append(annotation)

        # Skip if this image somehow has no selected objects
        if not yolo_annotations:
            continue

        image_name = f"food_{dataset_index:05d}"

        image_path = (
            OUTPUT_DIR
            / "images"
            / split_name
            / f"{image_name}.jpg"
        )

        label_path = (
            OUTPUT_DIR
            / "labels"
            / split_name
            / f"{image_name}.txt"
        )

        image.save(image_path)

        with open(label_path, "w", encoding="utf-8") as file:
            file.write("\n".join(yolo_annotations))

        if (counter + 1) % 500 == 0:
            print(
                f"Processed {counter + 1}/{len(indices)}"
            )

# --------------------------------------------------
# 8. Create data.yaml
# --------------------------------------------------

yaml_path = OUTPUT_DIR / "data.yaml"

with open(yaml_path, "w", encoding="utf-8") as file:

    file.write(f"path: {OUTPUT_DIR.resolve()}\n")
    file.write("train: images/train\n")
    file.write("val: images/val\n")
    file.write("test: images/test\n\n")

    file.write("names:\n")

    for class_id, class_name in enumerate(SELECTED_CLASSES):
        file.write(f"  {class_id}: {class_name}\n")

print("\n--------------------------------")
print("YOLO DATASET CREATED!")
print("--------------------------------")

print("Location:", OUTPUT_DIR.resolve())
print("data.yaml:", yaml_path.resolve())