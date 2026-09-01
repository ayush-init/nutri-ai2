"""
FoodLens AI - Full 103-Class YOLO11 Training Pipeline
Runs accelerated GPU training on datasets/food_yolo_103/data.yaml
"""

from ultralytics import YOLO
import torch
import sys
import os

def main():
    print("=" * 60)
    print("🔥 FoodLens AI - 103-Class YOLO11 Training Engine")
    print("=" * 60)

    # Check CUDA Availability
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        device_id = 0
        print(f"✓ NVIDIA GPU Detected: {gpu_name} (Device: {device_id})")
    else:
        device_id = "cpu"
        print("⚠ No CUDA GPU found, using CPU.")

    dataset_yaml = "datasets/food_yolo_103/data.yaml"
    if not os.path.exists(dataset_yaml):
        print(f"❌ Error: {dataset_yaml} not found!")
        print("Please run 'python scripts/prepare_foodseg103_full.py' first.")
        sys.exit(1)

    print("
[Step 1/2] Initializing YOLO11 pretrained architecture...")
    model = YOLO("yolo11n.pt")

    print("
[Step 2/2] Launching 103-Class Food Training...")
    print("  • Epochs: 40")
    print("  • Image Resolution: 640x640")
    print("  • Batch Size: 16 (optimized for GPU)")
    print("  • Output Run: runs/detect/foodlens_103_gpu")
    print("-" * 60)

    results = model.train(
        data=dataset_yaml,
        epochs=40,
        imgsz=640,
        batch=16,
        device=device_id,
        workers=2,
        name="foodlens_103_gpu",
        save=True,
        plots=True
    )

    print("
" + "=" * 60)
    print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
    print("Best weights saved at:")
    print("  runs/detect/foodlens_103_gpu/weights/best.pt")
    print("=" * 60)

if __name__ == "__main__":
    main()
