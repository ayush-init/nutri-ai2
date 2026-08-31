from pathlib import Path
import random
import cv2

DATASET_DIR = Path("datasets/food_yolo")

IMAGE_DIR = DATASET_DIR / "images" / "train"
LABEL_DIR = DATASET_DIR / "labels" / "train"

CLASS_NAMES = [
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

images = list(IMAGE_DIR.glob("*.jpg"))

image_path = random.choice(images)

label_path = LABEL_DIR / f"{image_path.stem}.txt"

image = cv2.imread(str(image_path))

height, width = image.shape[:2]

with open(label_path, "r", encoding="utf-8") as file:
    lines = file.readlines()

for line in lines:
    class_id, x_center, y_center, box_width, box_height = map(
        float,
        line.strip().split()
    )

    class_id = int(class_id)

    x_center *= width
    y_center *= height
    box_width *= width
    box_height *= height

    x1 = int(x_center - box_width / 2)
    y1 = int(y_center - box_height / 2)
    x2 = int(x_center + box_width / 2)
    y2 = int(y_center + box_height / 2)

    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )

    cv2.putText(
        image,
        CLASS_NAMES[class_id],
        (x1, max(y1 - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )

output_path = "datasets/food_yolo/visualized_sample.jpg"

cv2.imwrite(output_path, image)

print("Image:", image_path)
print("Saved visualization:", output_path)