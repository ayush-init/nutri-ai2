from collections import Counter
from datasets import load_dataset

print("Loading FoodSeg103...")

dataset = load_dataset(
    "pictograph/foodseg103",
    split="train"
)

print("Dataset loaded!")
print("Total images:", len(dataset))

food_counts = Counter()

for sample in dataset:
    food_names = sample["objects"]["category_names"]

    for food in food_names:
        food_counts[food] += 1

print("\nTotal food classes:", len(food_counts))

print("\nTop 20 food classes:")
for food, count in food_counts.most_common(20):
    print(f"{food:25} {count}")