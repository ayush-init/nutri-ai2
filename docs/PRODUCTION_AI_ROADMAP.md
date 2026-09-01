# 🥗 FoodLens AI — Production Multi-Dish AI & Training Roadmap

## 🎯 Goal & Architecture Overview
Transform the initial 15-class MVP into a production-ready **Open-World Multi-Dish Computer Vision Engine** capable of:
1. Identifying **multiple individual dishes and bowls on a single plate** (e.g. *Rice, Dal, Papad, Raita, Achar, Roti, Paneer, Chicken Curry, Chaat*).
2. Drawing **accurate bounding boxes & confidence scores** for every detected item.
3. Calculating **portion multipliers, calories, and macronutrients** stored in **Neon PostgreSQL**.
4. Utilizing **Hugging Face Pre-trained Models & Datasets** (FoodSeg103, Food-101) + **YOLO-World (Open-Vocabulary Object Detection)** for zero-lag local inference.
5. Providing full **Cloud & Container Deployment Readiness** (Docker, Render, AWS).

---

## 🏗️ System Architecture

```
                                  [ User Meal Photo ]
                                           │
                        FastAPI Backend (/api/v1/food/analyze)
                                           │
                     ┌─────────────────────┴─────────────────────┐
                     ▼                                           ▼
       [ YOLO-World Multi-Dish Engine ]             [ FoodSeg103 Fine-Tuned Model ]
     • Open-Vocabulary Plate Detection           • 103 Ingredient-Level Segmenter
     • Detects: Rice, Dal, Papad, Raita,         • Detects: Garnishes, Toppings,
       Achar, Roti, Curries, Chaat, etc.           Meats, Veggies, Cheeses
                     │                                           │
                     └─────────────────────┬─────────────────────┘
                                           ▼
                            [ Nutrition Calculation Engine ]
                        • Neon Serverless PostgreSQL Database
                        • Standard Portions (katori, cup, grams)
                        • Range Estimator (±15%) & Medical Disclaimer
                                           │
                                           ▼
                            [ Modern Web Dashboard & API ]
                        • Viewer.js Fullscreen Zoom Lightbox
                        • Annotated Bounding Box Downloader
                        • Chart.js Macro Donut Chart
```

---

## 🗺️ Step-by-Step Implementation Phases

### 📍 Phase 1: YOLO-World Multi-Dish Engine Integration
* **Objective:** Enable multi-item bounding box detection for comprehensive dishes on plates & thalis.
* **Tasks:**
  1. Integrate `yolov8x-worldv2` / `yolo11s-world` from Ultralytics & Hugging Face.
  2. Define an extensive Food & Dish vocabulary dictionary (covering Indian thali items, Asian bowls, Western dishes, fast food, and healthy meals).
  3. Update `backend/app/services/vision/yolo_service.py` to support dynamic vocabulary matching and multi-dish bounding box rendering.

---

### 📍 Phase 2: FoodSeg103 Full 103-Class Local Training Pipeline
* **Objective:** Train a custom localized YOLO11-Small model on all 103 FoodSeg103 classes from Hugging Face (`pictograph/foodseg103`).
* **Tasks:**
  1. Create `scripts/prepare_foodseg103_all.py` to extract all 103 categories with bounding boxes into `datasets/foodseg103_yolo/`.
  2. Implement `train_foodseg103.py` with Mosaic (1.0), MixUp (0.15), and HSV color augmentations.
  3. Train on local GPU (`device=0`) and save checkpoints to `weights/foodseg103_best.pt`.

---

### 📍 Phase 3: Neon PostgreSQL Global Nutrition Catalog Seeding
* **Objective:** Expand database nutrition profiles to cover all dishes, curries, thali components, and 103 ingredients.
* **Tasks:**
  1. Create comprehensive database migration `backend/app/db/seed_expanded_foods.py`.
  2. Populate accurate calorie, protein, carbohydrate, fat, serving unit (katori, bowl, tbsp, slice, grams) for all items.
  3. Ensure fast indexing on food names and canonical aliases.

---

### 📍 Phase 4: Frontend UI Enhancements for Multi-Dish Visualization
* **Objective:** Provide visual clarity when multiple bounding boxes are detected on one meal plate.
* **Tasks:**
  1. Color-coded bounding box badges mapped directly to table rows.
  2. Individual item calorie tags overlaid directly on the image viewer.
  3. Preset multi-dish test samples (*Indian Thali: Rice + Dal + Papad + Raita + Achar*, *Burger + Fries + Dip*, *Healthy Salad Bowl*).

---

### 📍 Phase 5: Verification, Pytest Suite & Real Multi-Dish Testing
* **Objective:** Verify 100% test coverage and test real multi-dish images (including Dahi Chaat, Indian Thali, and Breakfast Platters).
* **Tasks:**
  1. Add automated Pytest tests in `backend/tests/test_multi_dish.py`.
  2. Benchmark inference latency (< 100ms on GPU, < 300ms on CPU).
  3. Test bounding box IoU and confidence filtering.

---

### 📍 Phase 6: Production Dockerization & Cloud Deployment
* **Objective:** Prepare production build for deployment on Render, Railway, Hugging Face Spaces, or AWS EC2.
* **Tasks:**
  1. Optimize multi-stage `Dockerfile` with caching for PyTorch and YOLO weights.
  2. Add production environment variables and gunicorn/uvicorn worker configurations.
  3. Write step-by-step deployment documentation in `DEPLOYMENT.md`.
