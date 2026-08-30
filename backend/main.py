import cv2

image = cv2.imread("data/test_food.jpg")

if image is None:
    print("Could not read image")
else:
    print("Image loaded successfully")
    print("Original shape:", image.shape)

    resized = cv2.resize(image, (640, 640))

    cv2.imwrite("data/resized_food.jpg", resized)

    print("Resized image saved successfully")
    print("New shape:", resized.shape)