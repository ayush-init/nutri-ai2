from ultralytics import YOLO


MODEL_PATH = "runs/detect/foodlens_v1_gpu-3/weights/best.pt"
IMAGE_PATH = "test_images/myfood.png"


print("Loading FoodLens model...")

model = YOLO(MODEL_PATH)

print("Running food detection...")

results = model.predict(
    source=IMAGE_PATH,
    conf=0.25,
    device=0,
    save=True
)

for result in results:

    print("\nDetected foods:")

    for box in result.boxes:

        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        class_name = result.names[class_id]

        print(
            f"{class_name}: "
            f"{confidence:.2f}"
        )

print("\nPrediction completed!")