"""
FoodLens AI - v2 (103-Class) YOLO11 Training Pipeline
Experiment: foodlens_v2_103
Dataset: datasets/food_yolo_103/data.yaml
"""

from ultralytics import YOLO
import torch
import sys
import os

def main():
    print("=" * 60)
    print("🚀 FoodLens AI - v2 (103-Class) Training Pipeline")
    print("=" * 60)

    # 1. GPU Check
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        device_id = 0
        print(f"✓ GPU Active: {gpu_name} (Device: {device_id})")
    else:
        device_id = "cpu"
        print("⚠ CUDA not detected. Using CPU.")

    # 2. Strict Dataset Verification (Prevent falling back to 15-class v1 dataset)
    dataset_yaml = "datasets/food_yolo_103/data.yaml"

    if not os.path.exists(dataset_yaml):
        print(f"\n❌ Error: Dataset YAML not found: {dataset_yaml}")
        print("Please run 'python scripts/prepare_foodseg103_full.py' first.")
        sys.exit(1)

    print(f"✓ Verified 103-Class Dataset at: {dataset_yaml}")

    # 3. Load Pretrained YOLO11n
    print("\n[Step 1/2] Loading YOLO11n pretrained backbone...")
    model = YOLO("yolo11n.pt")

    # 4. Launch Training
    print("\n[Step 2/2] Launching FoodLens v2 Training (103 Classes)...")
    print("  • Experiment Name: foodlens_v2_103")
    print("  • Target Epochs: 30 (Patience: 5)")
    print("  • Resolution (imgsz): 480")
    print("  • Batch Size: 16 (If CUDA OOM occurs, reduce to 8)")
    print("  • Optimizer: AdamW (lr0: 0.002)")
    print("  • Mixed Precision (AMP): True")
    print("-" * 60)

    results = model.train(
        data=dataset_yaml,
        epochs=30,
        imgsz=480,
        batch=16,
        device=device_id,
        workers=2,
        amp=True,
        patience=5,
        optimizer="AdamW",
        lr0=0.002,
        name="foodlens_v2_103",
        save=True,
        plots=True
    )

    print("\n" + "=" * 60)
    print("🎉 FOODLENS v2 TRAINING COMPLETE!")
    print("Best weights saved at:")
    print("  runs/detect/foodlens_v2_103/weights/best.pt")
    print("=" * 60)

if __name__ == "__main__":
    main()
