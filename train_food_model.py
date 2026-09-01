from ultralytics import YOLO


def main():
    print("Loading YOLO model...")

    model = YOLO("yolo11n.pt")

    print("Starting FoodLens GPU training...")

    results = model.train(
        data="datasets/food_yolo/data.yaml",
        epochs=30,
        imgsz=640,
        batch=8,
        device=0,
        workers=0,
        name="foodlens_v1_gpu"
    )

    print("Training completed!")


if __name__ == "__main__":
    main()