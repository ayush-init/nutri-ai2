from datasets import load_dataset

print("Loading FoodSeg103...")

dataset = load_dataset(
    "pictograph/foodseg103",
    split="train"
)

print("\nDataset loaded!")
print("Number of images:", len(dataset))
print("Columns:", dataset.column_names)

sample = dataset[0]

print("\n--- IMAGE ---")
print("Type:", type(sample["image"]))
print("Size:", sample["image"].size)

print("\n--- OBJECTS ---")

objects = sample["objects"]

print("Bounding boxes:", objects["bbox"])
print("Categories:", objects["categories"])
print("Food names:", objects["category_names"])

print("\n--- SEGMENTATION ---")
print("Number of segmentation objects:", len(sample["segmentation"]))