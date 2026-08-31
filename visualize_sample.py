from datasets import load_dataset
from PIL import ImageDraw, ImageFont

print("Loading FoodSeg103...")

dataset = load_dataset(
    "pictograph/foodseg103",
    split="train"
)

sample = dataset[0]

image = sample["image"].convert("RGB")
draw = ImageDraw.Draw(image)

objects = sample["objects"]

boxes = objects["bbox"]
food_names = objects["category_names"]

for box, food_name in zip(boxes, food_names):
    x, y, width, height = box

    x1 = int(x)
    y1 = int(y)
    x2 = int(x + width)
    y2 = int(y + height)

    draw.rectangle(
        [x1, y1, x2, y2],
        outline="red",
        width=3
    )

    draw.text(
        (x1, max(0, y1 - 15)),
        food_name,
        fill="red"
    )

output_path = "backend/data/annotated_sample.jpg"

image.save(output_path)

print("Annotated image saved:", output_path)