"""
FoodLens AI - Ultra-Fast GPU Training Pipeline (Optimized)
"""

from ultralytics import YOLO
import torch
import sys
import os

def main():
    print("=" * 60)
    print("⚡ FoodLens AI - Fast GPU Training Pipeline")
    print("=" * 60)

    # 1. Detect GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        device_id = 0
        print(f"✓ NVIDIA GPU Active: {gpu_name} (Device: {device_id})")
    else:
        device_id = "cpu"
        print("⚠ CUDA not detected, falling back to CPU.")

    dataset_yaml = "datasets/food_yolo/data.yaml"
    if not os.path.exists(dataset_yaml):
        dataset_yaml = "datasets/food_yolo_103/data.yaml"

    if not os.path.exists(dataset_yaml):
        print(f"❌ Error: Dataset YAML not found ({dataset_yaml})")
        print("Please run 'python scripts/prepare_yolo_dataset.py' first.")
        sys.exit(1)

    print(f"✓ Using dataset: {dataset_yaml}")
    print("\n[Speed Optimizations Applied]:")
    print("  • Epochs: 15 (Fast convergence)")
    print("  • Resolution (imgsz): 480 (50% faster than 640)")
    print("  • Batch Size: 16 (Max GPU Tensor Core throughput)")
    print("  • Mixed Precision (FP16/AMP): ENABLED")
    print("  • Dataloader Workers: 2 (Prevents GPU starvation)")
    print("  • Early Stopping Patience: 5 epochs")
    print("-" * 60)

    # Load YOLO11 nano (lightweight & ultra fast)
    model = YOLO("yolo11n.pt")

    results = model.train(
        data=dataset_yaml,
        epochs=15,              # Fast 15 epochs
        imgsz=480,              # 480x480 is ~50% faster than 640x640
        batch=16,               # Higher batch = faster training on CUDA
        device=device_id,
        workers=2,              # Parallel dataloading
        amp=True,               # FP16 mixed precision
        patience=5,             # Stop early if converged
        optimizer="AdamW",      # Fast learning rate adaptation
        lr0=0.002,
        name="foodlens_v1_gpu",
        save=True,
        plots=True
    )

    print("\n" + "=" * 60)
    print("🎉 FAST TRAINING COMPLETED!")
    print("Best weights saved at:")
    print("  runs/detect/foodlens_v1_gpu/weights/best.pt")
    print("=" * 60)

if __name__ == "__main__":
    main()
