from ultralytics import YOLO

model = YOLO("yolo11n.pt")

results = model("data/test_food.jpg")

for result in results:
    result.save(filename="data/detected_food.jpg")

    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        class_name = result.names[class_id]

        print(f"{class_name}: {confidence:.2f}")