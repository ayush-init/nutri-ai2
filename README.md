# FoodLens --- AI Food Intelligence & Menu Assistant

## Complete Phase-Wise Development Specification

> **Purpose:** This document is the master implementation plan for
> building FoodLens with Antigravity. Build it **phase by phase**, not
> in one shot. Each phase must be implemented, tested, and verified
> before moving to the next phase.

------------------------------------------------------------------------

# 1. Project Vision

Build a production-style web application called **FoodLens**.

FoodLens allows a user to:

1.  Upload/capture a photo of a food dish.
2.  Detect visible food items using Computer Vision.
3.  Estimate nutritional information for detected foods.
4.  Upload a packaged-food image and extract printed
    nutrition/ingredient information using OCR.
5.  Upload or paste a menu and identify menu items.
6.  Rank/recommend menu items based on the user's selected preference.
7.  Explain why an item is recommended.
8.  Keep results and reports in a database.
9.  Expose the AI pipeline through a clean backend API.
10. Provide a polished web interface.

## Important scientific constraint

FoodLens must **never present photo-based calorie/macronutrient
estimates as exact medical or laboratory measurements**.

For food photos, show estimates/ranges and clearly label them as
estimates.

For packaged food, when nutrition values are read from the package label
using OCR, distinguish **OCR-extracted label values** from AI-estimated
values.

------------------------------------------------------------------------

# 2. Main Product Flows

## Flow A --- Food Photo Analysis

``` text
User
  ↓
Upload / Capture Food Photo
  ↓
Image Validation
  ↓
OpenCV Preprocessing
  ↓
Food Detection Model
  ↓
Detected Food Items + Bounding Boxes + Confidence
  ↓
Nutrition Lookup / Estimation
  ↓
Nutrition Summary
  ↓
Recommendation / Explanation
  ↓
Save Result
```

## Flow B --- Packaged Food Analysis

``` text
User
  ↓
Upload Package Photo
  ↓
OpenCV Preprocessing
  ↓
OCR
  ↓
Extract Text
  ↓
Parse Nutrition Facts / Ingredients
  ↓
Structured Nutrition Result
  ↓
Confidence / Review Flags
  ↓
Save Result
```

## Flow C --- Menu Analysis

``` text
User
  ↓
Upload Menu Image / Enter Menu Text
  ↓
OCR (if image)
  ↓
Menu Item Extraction
  ↓
Normalize Food Names
  ↓
Nutrition Lookup / Estimation
  ↓
Apply User Preference
  ↓
Rank Items
  ↓
Explain Recommendations
```

------------------------------------------------------------------------

# 3. Target Technology Stack

Use the following stack unless a genuine technical reason requires a
change.

## Backend

-   Python
-   FastAPI
-   Pydantic
-   SQLAlchemy (ORM)
-   PostgreSQL (Hosted on Neon DB) with connection pooling

## Computer Vision / ML

-   OpenCV
-   NumPy
-   PyTorch
-   Ultralytics YOLO
-   Pillow where useful

## OCR

-   PaddleOCR preferred
-   Keep OCR implementation modular so another OCR engine can be
    substituted later

## Frontend

-   Next.js / React
-   TypeScript
-   Clean responsive UI

## Developer Tools

-   Git
-   GitHub
-   `.env` configuration
-   Docker where useful

------------------------------------------------------------------------

# 4. Architecture

Use a modular architecture.

``` text
foodlens/
├── frontend/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   ├── core/
│   │   ├── models/
│   │   ├── schemas/
│   │   ├── services/
│   │   │   ├── vision/
│   │   │   ├── ocr/
│   │   │   ├── nutrition/
│   │   │   └── recommendation/
│   │   └── main.py
│   └── tests/
├── ml/
│   ├── datasets/
│   ├── training/
│   ├── inference/
│   └── notebooks/
├── data/
├── docs/
├── scripts/
├── .env.example
├── .gitignore
├── README.md
└── docker-compose.yml
```

Do not put all logic into one Python file.

Keep: - API routes - model inference - OCR - nutrition logic -
recommendation logic - database access

separate.

------------------------------------------------------------------------

# PHASE 0 --- Project Definition & Repository Setup

## Goal

Create a clean repository and development foundation.

## Tasks

1.  Create Git repository.
2.  Create the folder structure.
3.  Initialize frontend.
4.  Initialize Python backend.
5.  Create Python virtual environment.
6.  Add dependency management.
7.  Create `.env.example`.
8.  Configure `.gitignore`.
9.  Add basic README.
10. Add health-check endpoint.
11. Add basic frontend page.
12. Confirm frontend can communicate with backend.

## Backend acceptance test

`GET /health`

Expected:

``` json
{
  "status": "ok"
}
```

## Git checkpoint

Commit:

``` text
chore: initialize FoodLens project
```

Do not proceed until: - frontend starts - backend starts - `/health`
works - repository is clean and understandable

------------------------------------------------------------------------

# PHASE 1 --- Learn & Implement Image Upload Pipeline

## Goal

Create the first real FoodLens workflow without AI.

## Tasks

1.  Build image upload UI.
2.  Validate:
    -   file type
    -   file size
    -   corrupted images
3.  Send image to FastAPI.
4.  Store uploaded image safely.
5.  Return analysis ID.
6.  Display uploaded image.
7.  Add basic OpenCV processing:
    -   read image
    -   resize
    -   normalize where appropriate
    -   convert formats if required
8.  Create reusable image-processing service.

## API

Example:

``` text
POST /api/v1/analyze/image
```

Response should include:

``` json
{
  "analysis_id": "...",
  "status": "received"
}
```

## Acceptance

A user can upload a food image and see that the backend successfully
received and processed it.

## Git checkpoint

``` text
feat: add image upload and preprocessing pipeline
```

------------------------------------------------------------------------

# PHASE 2 --- Build Food Detection with YOLO

## Goal

Introduce the main Computer Vision model.

## Important

Do not immediately claim the model detects every possible food.

Start with a clearly defined supported food-class dataset.

## Tasks

1.  Understand:
    -   object detection
    -   bounding boxes
    -   confidence scores
    -   IoU
    -   NMS
    -   precision
    -   recall
    -   mAP
2.  Select an appropriate food image dataset.
3.  Define supported classes.
4.  Prepare dataset.
5.  Perform data labeling/conversion if needed.
6.  Create train/validation/test split.
7.  Train or fine-tune YOLO.
8.  Evaluate the model.
9.  Save model weights.
10. Build inference service.
11. Return:
    -   class name
    -   confidence
    -   bounding box
12. Draw bounding boxes for visualization.

## Example result

``` json
{
  "detections": [
    {
      "label": "rice",
      "confidence": 0.94,
      "bbox": [120, 80, 450, 390]
    }
  ]
}
```

## Acceptance

Food image:

``` text
Rice + Dal + Roti
```

should produce detected objects where supported by the trained classes.

If confidence is low, UI must say that the result is uncertain instead
of pretending it is correct.

## Git checkpoint

``` text
feat: add YOLO food detection
```

------------------------------------------------------------------------

# PHASE 3 --- Food Nutrition Engine

## Goal

Convert detected food classes into useful nutrition information.

## Important distinction

YOLO detects:

``` text
WHAT is present?
```

It does not directly know exact calories.

## Tasks

1.  Create a nutrition data layer.
2.  Store nutrition information per food item.
3.  Define standard serving assumptions.
4.  Allow serving-size adjustment.
5.  Calculate:
    -   calories
    -   protein
    -   carbohydrates
    -   fat
6.  Show estimates with clear assumptions.
7.  Return ranges where uncertainty is significant.

## Example

``` text
Detected:
Dal

Serving:
1 bowl

Estimated:
Calories: 180–240 kcal
Protein: 9–12 g
Carbs: 25–32 g
Fat: 4–7 g
```

## UI

Show:

``` text
Estimated nutrition
```

not:

``` text
Exact nutrition
```

## Acceptance

Detected food items produce consistent nutrition estimates based on a
documented serving assumption.

## Git checkpoint

``` text
feat: add nutrition estimation engine
```

------------------------------------------------------------------------

# PHASE 4 --- Food Photo Results UI

## Goal

Turn model output into a useful product experience.

## UI should show

-   Original image
-   Detected bounding boxes
-   Food item names
-   Confidence
-   Serving-size selector
-   Estimated calories
-   Protein
-   Carbs
-   Fat
-   Disclaimer about estimation

## Add

-   loading state
-   error state
-   empty result state
-   low-confidence warning

## Acceptance

A user can understand the result without looking at logs or raw JSON.

## Git checkpoint

``` text
feat: add food analysis results UI
```

------------------------------------------------------------------------

# PHASE 5 --- Packaged Food OCR

## Goal

Use OCR for real-world packaged food.

This phase directly demonstrates the OCR requirement.

## Workflow

``` text
Package Photo
    ↓
OpenCV preprocessing
    ↓
PaddleOCR
    ↓
Raw text
    ↓
Nutrition parser
    ↓
Structured data
```

## Tasks

1.  Add OCR service abstraction.
2.  Integrate PaddleOCR.
3.  Preprocess difficult images.
4.  Extract:
    -   product name
    -   calories
    -   serving size
    -   protein
    -   carbohydrates
    -   fat
    -   sugar
    -   sodium when available
    -   ingredients
    -   allergen statements when present
5.  Store raw OCR text separately.
6.  Store parsed structured fields separately.
7.  Show OCR confidence/review warnings.
8.  Never invent missing label values.

## Example

``` text
Product: Example Protein Bar

Calories: 210 kcal
Protein: 12 g
Carbs: 22 g
Fat: 8 g
Sugar: 7 g

Source:
Extracted from package label
```

If OCR cannot confidently read a field:

``` text
Not confidently detected
```

Do not hallucinate it.

## Git checkpoint

``` text
feat: add packaged food OCR pipeline
```

------------------------------------------------------------------------

# PHASE 6 --- Menu Upload & Menu OCR

## Goal

Allow users to upload a restaurant/canteen menu.

## Inputs

Support:

1.  Image menu
2.  PDF if practical
3.  Plain text

## Workflow

``` text
Menu Image
   ↓
OCR
   ↓
Raw Text
   ↓
Menu Item Parser
   ↓
Normalized Items
```

Example:

``` text
Today's Menu

Paneer Butter Masala
Dal Tadka
Jeera Rice
Roti
Fried Rice
```

Convert to structured items:

``` json
[
  "Paneer Butter Masala",
  "Dal Tadka",
  "Jeera Rice",
  "Roti",
  "Fried Rice"
]
```

## Acceptance

User can upload a menu and see cleanly extracted menu items.

## Git checkpoint

``` text
feat: add menu extraction workflow
```

------------------------------------------------------------------------

# PHASE 7 --- Nutrition-Aware Menu Recommendation

## Goal

Make FoodLens more than a detector/OCR app.

## User preferences

Allow selection such as:

-   Balanced
-   Higher protein
-   Lower calorie
-   Vegetarian
-   Custom preference

Do not make medical claims.

## Recommendation flow

``` text
Menu Items
   ↓
Nutrition Data
   ↓
User Preference
   ↓
Scoring
   ↓
Ranking
   ↓
Explanation
```

## Example

``` text
Best balanced options:

1. Dal Tadka + Roti
2. Paneer + Roti
3. Rajma Rice
```

For each recommendation explain:

``` text
Why?
Good estimated protein-to-calorie balance
```

## Important

Recommendations should be presented as general food guidance, not
medical advice.

## Git checkpoint

``` text
feat: add nutrition-aware menu recommendations
```

------------------------------------------------------------------------

# PHASE 8 --- "What's the Better Choice?" Feature

## Goal

Create a standout product feature.

User can upload:

``` text
Menu
```

or select several foods.

FoodLens compares them.

## Example

``` text
Compare

Pizza
Burger
Dal Rice
```

Output:

``` text
Recommended:
Dal Rice

Reason:
Estimated lower fat and a more balanced
carbohydrate/protein profile than the other
detected options.
```

## Comparison UI

Show:

-   estimated calories
-   protein
-   carbs
-   fat
-   preference score
-   confidence
-   assumptions

## Acceptance

User can compare at least 2 supported items and receive a transparent
ranking.

## Git checkpoint

``` text
feat: add food comparison engine
```

------------------------------------------------------------------------

# PHASE 9 --- Database & User History

## Goal

Persist useful application data.

## Store

-   analysis ID
-   uploaded image reference
-   analysis type
-   detected foods
-   confidence
-   nutrition estimate
-   OCR output
-   menu items
-   recommendations
-   timestamps
-   status

## Optional user accounts

If added, keep authentication separate from AI services.

## History UI

``` text
My Analyses

Food Photo
Package Scan
Menu Analysis
```

Users can reopen previous results.

## Git checkpoint

``` text
feat: add analysis history and persistence
```

------------------------------------------------------------------------

# PHASE 10 --- API Design & Documentation

## Goal

Make backend professional.

Create versioned APIs such as:

``` text
/api/v1/health

/api/v1/food/analyze
/api/v1/package/analyze
/api/v1/menu/analyze

/api/v1/recommendations
/api/v1/analyses
```

Use:

-   Pydantic schemas
-   validation
-   consistent errors
-   status codes
-   API documentation
-   clear response models

FastAPI should automatically expose Swagger/OpenAPI documentation.

## Acceptance

A developer can understand and test the AI system through API docs.

## Git checkpoint

``` text
refactor: finalize versioned API architecture
```

------------------------------------------------------------------------

# PHASE 11 --- Testing & Evaluation

## Goal

Do not call the project "AI" without measuring it.

## Computer Vision evaluation

Measure:

-   precision
-   recall
-   mAP
-   confusion where applicable
-   inference latency

## OCR evaluation

Create a small manually verified test set and measure field extraction
accuracy.

## Nutrition

Test deterministic calculations separately from model predictions.

## Backend

Test:

-   valid image
-   invalid file
-   oversized file
-   corrupted image
-   no detections
-   low confidence
-   OCR missing fields
-   unsupported food

## Acceptance

The project has a reproducible evaluation section in the README.

## Git checkpoint

``` text
test: add model and API evaluation suite
```

------------------------------------------------------------------------

# PHASE 12 --- Production Hardening

## Goal

Make the application reliable.

Implement:

-   environment variables
-   secure file handling
-   request validation
-   file size limits
-   safe filenames
-   logging
-   error handling
-   rate limiting where appropriate
-   model loading strategy
-   health checks
-   cleanup of temporary files

Never commit: - API keys - passwords - secrets - private credentials -
huge datasets - model artifacts if repository size becomes unreasonable

## Git checkpoint

``` text
chore: harden application for production
```

------------------------------------------------------------------------

# PHASE 13 --- Docker & Deployment

## Goal

Make the project easy to run.

Create:

``` text
Dockerfile
docker-compose.yml
```

Containerize backend and database where practical.

Frontend should have a clear production build.

## Environment

Provide:

``` text
.env.example
```

Never commit the real `.env`.

## README must explain

``` text
1. Clone
2. Configure environment
3. Install dependencies
4. Start services
5. Run migrations
6. Start frontend
7. Open application
```

## Git checkpoint

``` text
chore: containerize and prepare deployment
```

------------------------------------------------------------------------

# PHASE 14 --- Final UI/UX Polish

## Goal

Make the project look like a real product.

## Main navigation

``` text
FoodLens
├── Analyze Food
├── Scan Package
├── Analyze Menu
├── Compare Foods
└── History
```

## Design principles

-   clean
-   responsive
-   simple
-   fast feedback
-   clear confidence indicators
-   clear distinction between detected and estimated information
-   no unnecessary animations
-   accessible controls

## Main landing page

Explain:

``` text
Snap food.
Understand what's in it.
Make a more informed choice.
```

Then show 3 actions:

``` text
📸 Analyze Food
🥫 Scan Package
📋 Analyze Menu
```

## Git checkpoint

``` text
feat: polish FoodLens product experience
```

------------------------------------------------------------------------

# PHASE 15 --- Final Documentation & Portfolio

## Goal

Turn the project into a strong internship portfolio project.

README sections:

1.  Project overview
2.  Problem statement
3.  Features
4.  Architecture
5.  Tech stack
6.  AI pipeline
7.  Dataset
8.  Model training
9.  Model evaluation
10. OCR pipeline
11. Nutrition methodology
12. API documentation
13. Screenshots
14. Demo
15. Setup instructions
16. Limitations
17. Future improvements

## Architecture diagram

Include:

``` text
Frontend
   ↓
FastAPI
   ↓
┌──────────────┬──────────────┬──────────────┐
│              │              │
OpenCV       YOLO           OCR
│              │              │
└──────────────┼──────────────┘
               ↓
        Nutrition Engine
               ↓
      Recommendation Engine
               ↓
            Database
```

## Portfolio description

Use an honest description such as:

> Built FoodLens, an AI-powered food intelligence application that
> combines computer vision, object detection, OCR, nutrition estimation,
> and menu-aware recommendations in an end-to-end web application.

Do not claim unsupported accuracy or capabilities.

## Git checkpoint

``` text
docs: finalize FoodLens documentation
```

------------------------------------------------------------------------

# 5. Final Feature Set

The completed application should support:

## Food Analysis

-   Image upload
-   Food detection
-   Bounding boxes
-   Confidence scores
-   Nutrition estimates
-   Serving-size adjustment

## Packaged Food

-   OCR
-   Nutrition label extraction
-   Ingredient extraction
-   Allergen text detection where available
-   OCR confidence/review indicators

## Menu

-   Menu image upload
-   OCR
-   Item extraction
-   Nutrition lookup/estimation
-   Preference-based ranking

## Comparison

-   Compare multiple foods
-   Compare nutrition estimates
-   Explain recommendation

## Platform

-   FastAPI
-   Database
-   History
-   Responsive frontend
-   API docs
-   Docker/deployment

------------------------------------------------------------------------

# 6. What NOT to Do

1.  Do not build the whole project in one giant generation.
2.  Do not use fake AI outputs.
3.  Do not hardcode "AI detected" results without actually running a
    model.
4.  Do not claim exact calories from a photograph.
5.  Do not invent nutrition values when data is unavailable.
6.  Do not hide low-confidence predictions.
7.  Do not put the entire backend in one file.
8.  Do not commit secrets.
9.  Do not download random datasets without checking their license/usage
    terms.
10. Do not skip model evaluation.
11. Do not move to the next phase until the current phase works.

------------------------------------------------------------------------

# 7. Mandatory Antigravity Workflow

For **every phase**, follow this exact process:

### Step 1 --- Explain

Before coding, explain: - what this phase does - why it is needed -
which technology is being used - how it connects to the previous phase

### Step 2 --- Inspect

Inspect the existing repository.

Do not overwrite working code unnecessarily.

### Step 3 --- Implement

Implement only the current phase.

### Step 4 --- Test

Run relevant: - backend tests - frontend checks - model checks - API
checks

### Step 5 --- Verify

Show: - what was built - files changed - commands run - test results -
known limitations

### Step 6 --- Stop

**Do not automatically continue to the next phase.**

Wait for the user to explicitly request the next phase.

------------------------------------------------------------------------

# 8. Learning Alongside Development

The developer should not only write code.

For every important concept introduced, explain it briefly.

Examples:

## When YOLO is introduced

Explain:

-   object detection
-   classification vs detection
-   bounding boxes
-   confidence
-   IoU
-   NMS
-   precision/recall
-   mAP

## When OCR is introduced

Explain:

-   OCR
-   preprocessing
-   text detection
-   text recognition
-   confidence
-   structured extraction

## When FastAPI is introduced

Explain:

-   API
-   endpoint
-   request
-   response
-   HTTP methods
-   Pydantic validation

## When database is introduced

Explain:

-   table
-   row
-   primary key
-   relationships
-   ORM

The goal is that the user **understands the architecture instead of
blindly copying code**.

------------------------------------------------------------------------

# 9. Recommended Development Order

Follow exactly:

``` text
PHASE 0
Project setup
   ↓
PHASE 1
Image pipeline
   ↓
PHASE 2
YOLO food detection
   ↓
PHASE 3
Nutrition engine
   ↓
PHASE 4
Food results UI
   ↓
PHASE 5
Packaged food OCR
   ↓
PHASE 6
Menu OCR
   ↓
PHASE 7
Menu recommendation
   ↓
PHASE 8
Food comparison
   ↓
PHASE 9
Database/history
   ↓
PHASE 10
Professional API
   ↓
PHASE 11
Testing/evaluation
   ↓
PHASE 12
Production hardening
   ↓
PHASE 13
Docker/deployment
   ↓
PHASE 14
UI polish
   ↓
PHASE 15
Documentation/portfolio
```

------------------------------------------------------------------------

# 10. Definition of Done

FoodLens is complete only when:

-   [ ] Food photo can be uploaded
-   [ ] Supported foods are detected by a real trained/integrated model
-   [ ] Bounding boxes are displayed
-   [ ] Confidence is shown
-   [ ] Nutrition is estimated transparently
-   [ ] Packaged food labels can be OCR-scanned
-   [ ] OCR data is parsed into structured fields
-   [ ] Menu images can be analyzed
-   [ ] Menu items can be ranked based on preferences
-   [ ] Food comparison works
-   [ ] Results are persisted
-   [ ] APIs are documented
-   [ ] Tests exist
-   [ ] Model evaluation is documented
-   [ ] Application is deployable
-   [ ] README is complete
-   [ ] No secrets are committed
-   [ ] Limitations are explicitly documented

------------------------------------------------------------------------

# START CONDITION

When beginning this project with Antigravity:

**Start with PHASE 0 only.**

Do not implement Phase 1 or any AI functionality until Phase 0 has been
completed and verified.

After Phase 0 is verified, wait for the user to say:

> `Start Phase 1`

Then implement only Phase 1.
