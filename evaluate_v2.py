"""
FoodLens AI - v2 Evaluation & v1 vs v2 Benchmark Comparison
"""

from ultralytics import YOLO
import os
import sys

def main():
    print("=" * 65)
    print("📊 FoodLens AI - Model Evaluation & Benchmark Comparison")
    print("=" * 65)

    v2_weights = "runs/detect/foodlens_v2_103/weights/best.pt"
    dataset_yaml = "datasets/food_yolo_103/data.yaml"

    if not os.path.exists(v2_weights):
        print(f"❌ Error: Model weights not found at '{v2_weights}'")
        print("Please complete v2 training first using 'python train_food_model_v2.py'.")
        sys.exit(1)

    if not os.path.exists(dataset_yaml):
        print(f"❌ Error: Dataset YAML not found at '{dataset_yaml}'")
        sys.exit(1)

    print(f"✓ Loading FoodLens v2 weights from: {v2_weights}")
    model = YOLO(v2_weights)

    print("\nRunning validation on Test Split (103 classes)...")
    metrics = model.val(
        data=dataset_yaml,
        split="test",
        imgsz=480,
        batch=16,
        device=0,
        plots=True
    )

    p_v2 = round(float(metrics.box.mp), 4)
    r_v2 = round(float(metrics.box.mr), 4)
    map50_v2 = round(float(metrics.box.map50), 4)
    map50_95_v2 = round(float(metrics.box.map), 4)

    # v1 Baseline Metrics (from record)
    p_v1 = 0.579
    r_v1 = 0.551
    map50_v1 = 0.562
    map50_95_v1 = 0.458

    print("\n" + "=" * 65)
    print("📈 PERFORMANCE BENCHMARK: FoodLens v1 vs v2")
    print("=" * 65)
    print(f"{'Metric':<20} | {'v1 Baseline (15 Cls)':<20} | {'v2 Model (103 Cls)':<20}")
    print("-" * 65)
    print(f"{'Precision':<20} | {p_v1:<20.4f} | {p_v2:<20.4f}")
    print(f"{'Recall':<20} | {r_v1:<20.4f} | {r_v2:<20.4f}")
    print(f"{'mAP@0.50':<20} | {map50_v1:<20.4f} | {map50_v2:<20.4f}")
    print(f"{'mAP@0.50:0.95':<20} | {map50_95_v1:<20.4f} | {map50_95_v2:<20.4f}")
    print("=" * 65)

    print("\n✓ Evaluation complete! Validation plots saved in runs/detect/val/")

if __name__ == "__main__":
    main()
