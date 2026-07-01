<div align="center">

# 🏥 Personalised Disease Risk Prediction Using Deep Learning

**A multi-disease personalised risk prediction platform powered by deep learning and the CDC BRFSS dataset.**

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.x-000000?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Latest-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Classifier-006400)](https://xgboost.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](#license)

<br/>

<img src="https://img.shields.io/badge/Status-Active-brightgreen" alt="Status" />
<img src="https://img.shields.io/badge/Dataset-BRFSS%202014-orange" alt="Dataset" />
<img src="https://img.shields.io/badge/Records-400%2C000%2B-informational" alt="Records" />

---

*Predict individual risk for **Diabetes · Heart Disease · Asthma · Kidney Disease · Arthritis** using real-time ML inference with personalized lifestyle recommendations.*

</div>

<br/>

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#-key-features)
- [Supported Diseases & Models](#-supported-diseases--models)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the Server](#running-the-server)
- [API Reference](#-api-reference)
- [Pipeline Architecture](#-pipeline-architecture)
  - [Data Ingestion](#1-data-ingestion--pre-processing)
  - [Feature Engineering](#2-feature-translation--engineering)
- [Testing & Verification](#-testing--verification)
- [Contributing a New Disease Model](#-contributing-a-new-disease-model)
- [Documentation Index](#-documentation-index)
- [Tech Stack](#-tech-stack)
- [License](#license)

---

## Overview

This platform provides a **unified web dashboard** for multi-disease health risk prediction. It integrates heterogeneous ML classifiers — each trained on the CDC's [Behavioral Risk Factor Surveillance System (BRFSS)](https://www.cdc.gov/brfss/) 2014 dataset containing **400,000+ patient records** — into a single Flask application with a dynamic frontend.

Healthcare professionals and patients can input clinical and lifestyle parameters, receive a **real-time probability score**, a color-coded **risk meter**, and **tailored medical recommendations** based on the predicted risk level.

---

## ✨ Key Features

| Feature | Description |
|:--------|:------------|
| **Multi-Disease Dashboard** | Single UI supporting 5+ diseases with dynamic form fields per disease tab |
| **Heterogeneous ML Models** | Logistic Regression, Random Forest, Gradient Boosting, and XGBoost classifiers |
| **Real-time Prediction Engine** | Decoupled pipeline handling pre-processing, scaling, feature engineering, and inference |
| **Intelligent Risk Scoring** | Probability-based classification (Low / Medium / High) with dynamic visual gauge |
| **Lifestyle Recommendations** | Personalized clinical guidelines generated per risk level |
| **REST API** | JSON endpoints for programmatic access and integration |
| **Extensible Architecture** | Plug-in new disease models via configuration — no core code changes needed |

---

## 🩺 Supported Diseases & Models

| Disease | Algorithm | Features | Key Predictors | Threshold | Scaler |
|:--------|:----------|:--------:|:---------------|:---------:|:------:|
| **Diabetes** | Logistic Regression | 10 | BMI, Age Group, General Health, Smoker Status, Exercise, CVD History | 0.50 | StandardScaler |
| **Heart Disease** | Random Forest | 19 | Age, Sex, General Health, Physical/Mental Days, Exercise, BMI, Smoking, Alcohol | Dynamic | StandardScaler |
| **Asthma** | Gradient Boosting | 10 | Age Group, Sex, BMI, Smoking, Sleep Duration, COPD History | 0.50 | StandardScaler |
| **Kidney Disease** | Random Forest | 19 | Age, General Health, Physical Health Days, Diabetes, Education, Marital Status | Dynamic | StandardScaler |
| **Arthritis** | XGBoost Classifier | 14 | Age Group, Sex, BMI, Smoking, Alcohol, Exercise, Sleep, Joint Pain (Days) | ~0.59 | StandardScaler |

> **⚠️ Note:** The **Stroke** tab UI and API endpoint are configured, but the model file (`stroke_model.pkl`) is pending upload. Predictions for Stroke will return a descriptive API error until the model is provided.

---

## 📁 Project Structure

```
personalised-disease-risk-prediction/
│
├── app.py                     # Flask server & API endpoint configuration
├── predict.py                 # Centralized prediction engine & feature engineering
├── requirements.txt           # Python dependencies
├── .gitignore                 # Git ignore rules
│
├── models/                    # Trained model assets (pickle files)
│   ├── diabetes/              # Logistic Regression — 4 files
│   ├── heartDisease/          # Random Forest — 6 files
│   ├── asthma/                # Gradient Boosting — 4 files
│   ├── kidneyDisease/         # Random Forest — 5 files
│   ├── arthritis/             # XGBoost — 5 files
│   └── stroke/                # Pending model upload
│
├── templates/
│   └── index.html             # Frontend dashboard (HTML + CSS + JS)
│
├── test_cases.json            # Programmatic test profiles (Low/Med/High per disease)
├── test_cases.md              # Human-readable manual testing guide
│
└── docs/                      # Integration & collaboration guides
    ├── README_INTEGRATION_DOCS.md
    ├── INTEGRATION_GUIDE.md
    ├── INTEGRATION_SYSTEM_OVERVIEW.md
    ├── MODEL_PROVIDER_CHECKLIST.md
    ├── ADMIN_INTEGRATION_GUIDE.md
    ├── EXAMPLE_MODEL_README.md
    ├── GOOGLE_COLAB_TRAINING_PROMPT.md
    └── PROVIDER_PROMPT.py
```

> **Note:** The documentation files currently live in the project root. The `docs/` structure above is the recommended layout for new clones.

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.9+** installed on your system
- **pip** (Python package manager)
- **Git** (for cloning the repository)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/personalised-disease-risk-prediction.git
cd personalised-disease-risk-prediction

# 2. Create and activate a virtual environment
python -m venv .venv

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Windows (CMD)
.venv\Scripts\activate.bat

# macOS / Linux
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Running the Server

```bash
python app.py
```

The dashboard will be available at **http://127.0.0.1:5000**

> **💡 Tip:** The server runs with `use_reloader=False` to prevent Flask from spawning sub-processes that would use the system Python instead of the virtual environment — this avoids `ModuleNotFoundError` for packages like `xgboost` that are only installed inside `.venv`.

---

## 📡 API Reference

### `POST /predict/<disease>`

Accepts a JSON body with patient parameters and returns a risk prediction.

**Supported `<disease>` values:** `diabetes`, `heart_disease`, `asthma`, `kidney_disease`, `arthritis`, `stroke`

#### Example Request

```bash
curl -X POST http://127.0.0.1:5000/predict/arthritis \
  -H "Content-Type: application/json" \
  -d '{
    "age": 21,
    "sex": "Male",
    "bmi": 21.2,
    "general_health": "Poor",
    "physical_health_days": 7,
    "sleep_duration": 8,
    "exercise": "No",
    "smoking": "Yes",
    "smoker_status": "Former Smoker",
    "alcohol_consumption": "Yes",
    "diabetes": "No",
    "heart_disease": "No",
    "diff_walking": "Yes",
    "income": "$50k to $75k",
    "education": "Some college"
  }'
```

<details>
<summary><strong>Example Response</strong> (click to expand)</summary>

```json
{
  "status": "success",
  "prediction": "Low Risk",
  "risk_percentage": 17.63,
  "risk_level": "Low Risk",
  "accuracy": 0.8062,
  "model_name": "XGBoost Classifier",
  "dataset": "BRFSS",
  "recommendations": [
    "Maintain a regular low-impact exercise routine (brisk walking, swimming, cycling).",
    "Keep body weight within a healthy range to reduce stress on joints.",
    "Incorporate anti-inflammatory foods like omega-3 rich fish, nuts, and leafy greens.",
    "Stay active and avoid long periods of sitting or standing in one position."
  ],
  "risk_factors": []
}
```

</details>

#### PowerShell Example

```powershell
Invoke-RestMethod -Method Post `
  -Uri "http://127.0.0.1:5000/predict/diabetes" `
  -ContentType "application/json" `
  -Body '{"age_group": "50-54", "sex": "Male", "bmi": 28.5, "general_health": "Fair", "smoking": "Yes", "exercise": "No", "heart_attack": "No", "heart_disease": "No", "stroke": "No", "kidney_disease": "No"}'
```

---

## ⚙️ Pipeline Architecture

### 1. Data Ingestion & Pre-processing

The Flask backend ([app.py](app.py)) receives form/JSON requests and normalizes inputs before passing them to the prediction engine:

| Transformation | Details |
|:--------------|:--------|
| **Binary fields** | `Yes → 1`, `No → 0` |
| **Sex/Gender** | `Male → 1`, `Female → 0` |
| **Categorical fields** | Age group, health, smoker status, income, education → integer codes |
| **Raw age fallback** | If a model needs continuous `_AGE80` but only an age group is provided, the midpoint is calculated (e.g., `55-59 → 57`) |

### 2. Feature Translation & Engineering

The prediction module ([predict.py](predict.py)) applies disease-specific feature engineering via `_engineer_features()`:

<details>
<summary><strong>Heart Disease & Kidney Disease</strong></summary>

- `AGE_GROUP` — 1–13 categories
- `BMI_CATEGORY` — 1: Underweight, 2: Normal, 3: Overweight, 4: Obese
- `HEALTH_SCORE` = `PHYSHLTH + MENTHLTH + GENHLTH × 2`
- `LIFESTYLE_RISK` = `SMOKE100 + (1 − EXERANY2) + DRNKANY5`

</details>

<details>
<summary><strong>Asthma</strong></summary>

- `_AGE_G` — 6 age categories (`18-24`=1 through `65+`=6)

</details>

<details>
<summary><strong>Arthritis</strong></summary>

- `Age` — 6 categories matching `_AGE_G`
- `Gender` — Male=1.0, Female=2.0
- `BMI` — Scaled continuous BMI × 100.0
- `JointPain` — 0 days → BRFSS code `88.0` (None); 1–30 days preserved
- `Diabetes` — Yes=1.0, No=3.0
- `AlcoholConsumption` / `PhysicalActivity` / `DiffWalking` — Yes=1.0, No=2.0

</details>

---

## 🧪 Testing & Verification

The project ships with pre-verified test profiles (**Low**, **Medium**, **High** risk) for every disease:

| Resource | Description |
|:---------|:------------|
| [test_cases.md](test_cases.md) | Human-readable manual testing guide with step-by-step instructions |
| [test_cases.json](test_cases.json) | Machine-readable JSON profiles for automated/API testing |

### Quick API Test

```powershell
# Test low-risk arthritis profile
Invoke-RestMethod -Method Post `
  -Uri "http://127.0.0.1:5000/predict/arthritis" `
  -ContentType "application/json" `
  -Body '{"age": 21, "sex": "Male", "bmi": 21.2, "general_health": "Poor", "physical_health_days": 7, "sleep_duration": 8, "exercise": "No", "smoking": "Yes", "smoker_status": "Former Smoker", "alcohol_consumption": "Yes", "diabetes": "No", "heart_disease": "No", "diff_walking": "Yes", "income": "$50k to $75k", "education": "Some college"}'
```

---

## 🤝 Contributing a New Disease Model

This system is designed for **collaborative model integration**. If you're training and contributing a new disease model:

1. **Start here →** [README_INTEGRATION_DOCS.md](README_INTEGRATION_DOCS.md) — Navigation portal for all integration docs
2. **Model Providers →** [MODEL_PROVIDER_CHECKLIST.md](MODEL_PROVIDER_CHECKLIST.md) — Export, name, and package your `.pkl` files
3. **Training in Colab →** [GOOGLE_COLAB_TRAINING_PROMPT.md](GOOGLE_COLAB_TRAINING_PROMPT.md) — Copy-paste prompts for Google Colab
4. **System Admins →** [ADMIN_INTEGRATION_GUIDE.md](ADMIN_INTEGRATION_GUIDE.md) — Wire up `app.py`, `index.html`, and test

### What You Need to Deliver

```
your_disease.zip
├── README.md                    # Feature mappings, thresholds, recommendations
└── models/<disease_id>/
    ├── <disease>_model.pkl      # Trained classifier
    ├── <disease>_scaler.pkl     # Fitted StandardScaler
    ├── feature_names.pkl        # Feature list in exact training order
    └── model_meta.pkl           # {"accuracy": X, "model_name": "...", "dataset": "BRFSS 2014"}
```

---

## 📚 Documentation Index

| Document | Audience | Description |
|:---------|:---------|:------------|
| [README_INTEGRATION_DOCS.md](README_INTEGRATION_DOCS.md) | Everyone | Quick navigation portal for all integration documents |
| [INTEGRATION_SYSTEM_OVERVIEW.md](INTEGRATION_SYSTEM_OVERVIEW.md) | Everyone | High-level system overview (5 min read) |
| [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md) | Technical | Comprehensive specs — directory structure, feature mappings, validation |
| [MODEL_PROVIDER_CHECKLIST.md](MODEL_PROVIDER_CHECKLIST.md) | Model Providers | Printable checklist for exporting and delivering trained models |
| [ADMIN_INTEGRATION_GUIDE.md](ADMIN_INTEGRATION_GUIDE.md) | Admins | Step-by-step code changes for adding a new disease to the system |
| [EXAMPLE_MODEL_README.md](EXAMPLE_MODEL_README.md) | Model Providers | Template example for documenting feature mappings |
| [GOOGLE_COLAB_TRAINING_PROMPT.md](GOOGLE_COLAB_TRAINING_PROMPT.md) | Model Providers | Copy-paste prompts and Colab workflow guide |
| [PROVIDER_PROMPT.py](PROVIDER_PROMPT.py) | Model Providers | Shareable prompt script for collaborators |

---

## 🛠 Tech Stack

| Layer | Technology |
|:------|:-----------|
| **Backend** | Python 3.9+, Flask |
| **ML Framework** | scikit-learn, XGBoost |
| **Data Processing** | NumPy, Pandas, SciPy |
| **Model Serialization** | Joblib |
| **Frontend** | HTML5, CSS3, Vanilla JavaScript |
| **Dataset** | CDC BRFSS 2014 (400K+ records) |

---

## License

This project is developed as part of **Personalised Disease Risk Prediction Using Deep Learning**. See [LICENSE](LICENSE) for details.

---

<div align="center">

**Personalised Disease Risk Prediction Using Deep Learning — Built with ❤️ using Flask & scikit-learn**

*If you find this useful, give it a ⭐!*

</div>
