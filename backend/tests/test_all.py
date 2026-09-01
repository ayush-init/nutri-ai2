import sys
import os
import pytest
import numpy as np
import cv2
from fastapi.testclient import TestClient

# Ensure backend directory is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.main import app

@pytest.fixture
def client():
    return TestClient(app)

def test_health_check(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["healthy", "degraded"]
    assert "database" in data
    assert data["database"]["type"] == "Neon PostgreSQL"

def test_image_upload_validation(client):
    res_bad = client.post("/api/v1/image/upload", files={"file": ("hack.exe", b"binary", "application/octet-stream")})
    assert res_bad.status_code == 400

def test_food_nutrition_items(client):
    response = client.get("/api/v1/food/items")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] >= 15
    assert any("rice" in i["label"] for i in data["items"])
    assert any("dal" in i["label"] or "broccoli" in i["label"] for i in data["items"])

def test_multi_dish_analyze(client):
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.circle(img, (200, 200), 100, (255, 255, 255), -1)
    cv2.circle(img, (450, 200), 80, (0, 200, 255), -1)
    _, encoded = cv2.imencode(".png", img)

    response = client.post(
        "/api/v1/food/analyze?conf_threshold=0.10",
        files={"file": ("test_plate.png", encoded.tobytes(), "image/png")}
    )
    assert response.status_code == 200
    data = response.json()
    assert "total_calories" in data
    assert "total_protein" in data
    assert "detected_items" in data
    assert "annotated_image_url" in data

def test_package_ocr_parsing(client):
    label_text = """
    Nutrition Facts
    Serving Size 1 cup (240ml)
    Calories 150
    Total Fat 8g
    Total Carbohydrate 12g
    Protein 8g
    Sodium 125mg
    Ingredients: Milk, Vitamin D3.
    Contains: Milk.
    """
    response = client.post("/api/v1/package/analyze-text", data={"label_text": label_text})
    assert response.status_code == 200
    data = response.json()
    assert data["calories"]["value"] == 150.0
    assert data["protein"]["value"] == 8.0
    assert data["fat"]["value"] == 8.0
    assert "Milk" in data["allergens"]

def test_menu_recommendation(client):
    menu_sample = """
    Daily Menu:
    1. Grilled Chicken Salad - $12.99
    2. Deep Fried Double Cheese Burger with Loaded Fries - $15.99
    3. Steamed Broccoli with Brown Rice - $9.99
    """
    response = client.post(
        "/api/v1/menu/recommend",
        json={"menu_text": menu_sample, "preference": "balanced"}
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["top_recommendations"]) > 0

def test_food_comparison(client):
    response = client.post(
        "/api/v1/menu/compare",
        json={"food_names": ["Grilled Chicken Salad", "Loaded Bacon Cheeseburger"], "preference": "balanced"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "winner" in data
    assert "summary_reason" in data
    assert "comparison_table" in data
    assert len(data["comparison_table"]) >= 2

def test_history_logging(client):
    response = client.get("/api/v1/history?limit=5")
    assert response.status_code == 200
    data = response.json()
    assert "history" in data
    assert isinstance(data["history"], list)
