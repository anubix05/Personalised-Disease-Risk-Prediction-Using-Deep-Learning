"""
DISEASE MODEL INTEGRATION GUIDE
================================
Instructions for integrating trained disease prediction models into the system.

This document specifies the exact format, structure, and requirements for adding
new diseases to the Personalised Disease Risk Prediction Using Deep Learning platform.

"""

# ═══════════════════════════════════════════════════════════════════════════════
# 1. DIRECTORY STRUCTURE & MODEL FILES
# ═══════════════════════════════════════════════════════════════════════════════

"""
REQUIRED STRUCTURE:

models/
├── diabetes/                          [EXISTING - DO NOT MODIFY]
│   ├── diabetes_model.pkl
│   ├── diabetes_scaler.pkl
│   ├── feature_names.pkl
│   └── model_meta.pkl
│
├── hypertension/                      [NEW - ADD YOUR MODELS HERE]
│   ├── hypertension_model.pkl
│   ├── hypertension_scaler.pkl
│   ├── feature_names.pkl
│   └── model_meta.pkl
│
├── heart_disease/                     [NEW - ADD YOUR MODELS HERE]
│   ├── heart_disease_model.pkl
│   ├── heart_disease_scaler.pkl
│   ├── feature_names.pkl
│   └── model_meta.pkl
│
└── stroke/                            [NEW - ADD YOUR MODELS HERE]
    ├── stroke_model.pkl
    ├── stroke_scaler.pkl
    ├── feature_names.pkl
    └── model_meta.pkl

NAMING CONVENTION:
- All files must use joblib (not pickle)
- Disease directory name must match disease ID (lowercase, underscores)
- Files must be named exactly as shown above (no variations)
"""


# ═══════════════════════════════════════════════════════════════════════════════
# 2. MODEL FILES SPECIFICATION
# ═══════════════════════════════════════════════════════════════════════════════

"""
REQUIRED MODEL FILES (all saved with joblib.dump()):

A) <disease>_model.pkl
   - Scikit-learn trained classifier (LogisticRegression, RandomForest, etc.)
   - Must support .predict() and .predict_proba() methods
   - predict_proba() must return 2D array with class 0 and 1 probabilities
   - Example:
     model = LogisticRegression()
     model.fit(X_train, y_train)
     joblib.dump(model, 'hypertension_model.pkl')

B) <disease>_scaler.pkl
   - StandardScaler or similar fitted to training data
   - Must have .transform() method matching model input
   - Example:
     from sklearn.preprocessing import StandardScaler
     scaler = StandardScaler()
     scaler.fit(X_train)
     joblib.dump(scaler, 'hypertension_scaler.pkl')

C) feature_names.pkl
   - Python list of feature names in exact order used during training
   - MUST match the order that data was passed to scaler and model
   - Example:
     features = [
         "_BMI5", "_AGEG5YR", "SEX", "GENHLTH", "SMOKE100",
         "EXERANY2", "CVDINFR4", "CVDCRHD4", "CVDSTRK3", "CHCKIDNY"
     ]
     joblib.dump(features, 'feature_names.pkl')

D) model_meta.pkl
   - Python dict with metadata
   - Required keys: "accuracy", "model_name", "dataset"
   - Example:
     meta = {
         "accuracy": 0.85,               # float between 0 and 1
         "model_name": "Logistic Regression",
         "dataset": "BRFSS 2014",
         "trained_date": "2026-06-23",   # optional
         "version": "1.0"                # optional
     }
     joblib.dump(meta, 'model_meta.pkl')
"""


# ═══════════════════════════════════════════════════════════════════════════════
# 3. FEATURE MAPPING SPECIFICATION
# ═══════════════════════════════════════════════════════════════════════════════

"""
FEATURE MAPPING REQUIREMENTS:

Each disease must define:
1. Human-readable input field names (used in HTML form)
2. BRFSS field names (used in model)
3. Mapping functions for each field type
4. Valid value ranges

FEATURE TYPES & MAPPINGS:

TYPE 1: CONTINUOUS (e.g., BMI, Age)
  Format: Float
  HTML:   <input type="number" step="0.1" min="X" max="Y">
  API:    Pass as float directly
  Example: "bmi": 27.5

TYPE 2: ORDINAL CATEGORIES (e.g., Age Groups)
  Format: Categorical with numeric mapping
  HTML:   <select> with options
  API:    Must map dropdown value to numeric code
  Example Mapping:
    {
        "18-24": 1,
        "25-29": 2,
        ...
        "80+": 13
    }

TYPE 3: BINARY (Yes/No)
  Format: Boolean (1 = Yes, 0 = No)
  HTML:   <select> with "Yes" and "No" options
  API:    Map "Yes" → 1, "No" → 0
  Example:
    "smoking": "Yes"  →  SMOKE100: 1

TYPE 4: ORDINAL SCALE (e.g., Health Rating 1-5)
  Format: Numeric scale with mapping
  HTML:   <select> with descriptive options
  API:    Must map option text to numeric code
  Example Mapping:
    {
        "excellent": 1,
        "very_good": 2,
        "good": 3,
        "fair": 4,
        "poor": 5
    }

TYPE 5: CATEGORICAL SINGLE SELECT (e.g., Sex, Race)
  Format: 1-2 categorical values with mapping
  HTML:   <select> with options
  API:    Must map option to numeric code
  Example Mapping:
    {
        "male": 1,
        "female": 0
    }
"""


# ═══════════════════════════════════════════════════════════════════════════════
# 4. PROVIDING MODEL FOR HYPERTENSION (TEMPLATE)
# ═══════════════════════════════════════════════════════════════════════════════

"""
STEP 1: GATHER YOUR MODEL FILES
================================
From your trained model on your machine, export:
  □ hypertension_model.pkl       (fitted classifier)
  □ hypertension_scaler.pkl      (fitted StandardScaler)
  □ feature_names.pkl            (list of feature names)
  □ model_meta.pkl               (metadata dict)

STEP 2: DEFINE YOUR FEATURES & MAPPINGS
=========================================
You must provide a complete list of all input features your model uses.
For each feature, specify:

EXAMPLE FOR HYPERTENSION:

[
  {
    "id": "bmi",
    "display_name": "Body Mass Index (BMI)",
    "field_type": "continuous",
    "brfss_name": "_BMI5",
    "html_input": {"type": "number", "step": 0.1, "min": 10, "max": 60},
    "placeholder": "e.g., 27.5"
  },
  {
    "id": "age_group",
    "display_name": "Age Group",
    "field_type": "ordinal_category",
    "brfss_name": "_AGEG5YR",
    "mapping": {
      "18-24": 1,
      "25-29": 2,
      "30-34": 3,
      "35-39": 4,
      "40-44": 5,
      "45-49": 6,
      "50-54": 7,
      "55-59": 8,
      "60-64": 9,
      "65-69": 10,
      "70-74": 11,
      "75-79": 12,
      "80+": 13
    }
  },
  {
    "id": "sex",
    "display_name": "Sex",
    "field_type": "binary_category",
    "brfss_name": "SEX",
    "mapping": {"Male": 1, "Female": 0}
  },
  {
    "id": "general_health",
    "display_name": "General Health",
    "field_type": "ordinal_scale",
    "brfss_name": "GENHLTH",
    "mapping": {
      "Excellent": 1,
      "Very Good": 2,
      "Good": 3,
      "Fair": 4,
      "Poor": 5
    }
  },
  {
    "id": "smoking",
    "display_name": "Smoking History",
    "field_type": "binary_yes_no",
    "brfss_name": "SMOKE100",
    "mapping": {"Yes": 1, "No": 0}
  }
  ... [ADD MORE FIELDS AS NEEDED]
]

STEP 3: PROVIDE IN THIS FORMAT
===============================
Send the model provider (data scientist) the following information:

├── Disease ID:            hypertension
├── Display Name:          Hypertension Risk
├── Model Accuracy:        0.82
├── Dataset Used:          BRFSS 2014 / Custom
├── Feature Count:         10
├── Risk Thresholds:
│   ├── Low Risk:         0-30%
│   ├── Medium Risk:      31-70%
│   └── High Risk:        71-100%
├── Model Files:
│   ├── hypertension_model.pkl
│   ├── hypertension_scaler.pkl
│   ├── feature_names.pkl (ordered list)
│   └── model_meta.pkl
└── Feature Mappings (see example above)
"""


# ═══════════════════════════════════════════════════════════════════════════════
# 5. DETAILED FEATURE SPECIFICATIONS FOR EACH DISEASE
# ═══════════════════════════════════════════════════════════════════════════════

"""
═══════════════════════════════════════════════════════════════════════════════
DISEASE: HYPERTENSION
═══════════════════════════════════════════════════════════════════════════════

Model Provider Checklist:
  □ Disease ID: hypertension
  □ API Route: /predict/hypertension
  □ Model files saved to: models/hypertension/
  
Features to Collect (EXACT FORMAT REQUIRED):

1. Body Mass Index (BMI)
   └─ Type: Continuous (float)
   └─ BRFSS Name: _BMI5
   └─ Example Value: 27.5
   └─ Range: 10-60
   └─ Step: 0.1

2. Age Group
   └─ Type: Ordinal Category
   └─ BRFSS Name: _AGEG5YR
   └─ Options: 18-24, 25-29, 30-34, 35-39, 40-44, 45-49, 50-54, 55-59, 60-64, 65-69, 70-74, 75-79, 80+
   └─ Mapping: 1-13
   └─ Example Value: "50-54" → 7

3. Sex
   └─ Type: Binary Category
   └─ BRFSS Name: SEX
   └─ Options: Male, Female
   └─ Mapping: Male=1, Female=0

4. General Health
   └─ Type: Ordinal Scale
   └─ BRFSS Name: GENHLTH
   └─ Options: Excellent, Very Good, Good, Fair, Poor
   └─ Mapping: 1-5 (1=Excellent, 5=Poor)

5. Smoking History
   └─ Type: Binary Yes/No
   └─ BRFSS Name: SMOKE100
   └─ Options: Yes, No
   └─ Mapping: Yes=1, No=0
   └─ Description: "Have you smoked 100+ cigarettes in your lifetime?"

6. Regular Exercise
   └─ Type: Binary Yes/No
   └─ BRFSS Name: EXERANY2
   └─ Options: Yes, No
   └─ Mapping: Yes=1, No=0

7. Heart Attack History
   └─ Type: Binary Yes/No
   └─ BRFSS Name: CVDINFR4
   └─ Options: Yes, No
   └─ Mapping: Yes=1, No=0

8. Coronary Heart Disease
   └─ Type: Binary Yes/No
   └─ BRFSS Name: CVDCRHD4
   └─ Options: Yes, No
   └─ Mapping: Yes=1, No=0

9. Stroke History
   └─ Type: Binary Yes/No
   └─ BRFSS Name: CVDSTRK3
   └─ Options: Yes, No
   └─ Mapping: Yes=1, No=0

10. Kidney Disease
    └─ Type: Binary Yes/No
    └─ BRFSS Name: CHCKIDNY
    └─ Options: Yes, No
    └─ Mapping: Yes=1, No=0

Feature Order (CRITICAL - MUST MATCH TRAINING):
  ["_BMI5", "_AGEG5YR", "SEX", "GENHLTH", "SMOKE100", "EXERANY2", "CVDINFR4", "CVDCRHD4", "CVDSTRK3", "CHCKIDNY"]

Risk Thresholds (define these):
  □ Low Risk:     0-30%
  □ Medium Risk:  31-70%
  □ High Risk:    71-100%

Model Metadata:
  accuracy: [YOUR MODEL ACCURACY]
  model_name: [e.g., "Logistic Regression", "Random Forest"]
  dataset: [e.g., "BRFSS 2014"]


═══════════════════════════════════════════════════════════════════════════════
DISEASE: HEART DISEASE
═══════════════════════════════════════════════════════════════════════════════

Model Provider Checklist:
  □ Disease ID: heart_disease
  □ API Route: /predict/heart_disease
  □ Model files saved to: models/heart_disease/

[PROVIDE SAME STRUCTURE AS ABOVE FOR HEART DISEASE]
[Define which features are used by your model]
[Specify any custom feature mappings]


═══════════════════════════════════════════════════════════════════════════════
DISEASE: STROKE
═══════════════════════════════════════════════════════════════════════════════

Model Provider Checklist:
  □ Disease ID: stroke
  □ API Route: /predict/stroke
  □ Model files saved to: models/stroke/

[PROVIDE SAME STRUCTURE AS ABOVE FOR STROKE]
[Define which features are used by your model]
[Specify any custom feature mappings]


═══════════════════════════════════════════════════════════════════════════════
DISEASE: [YOUR CUSTOM DISEASE]
═══════════════════════════════════════════════════════════════════════════════

Model Provider Checklist:
  □ Disease ID: [lowercase_with_underscores]
  □ API Route: /predict/[disease_id]
  □ Model files saved to: models/[disease_id]/

[COPY STRUCTURE FROM TEMPLATES ABOVE]
"""


# ═══════════════════════════════════════════════════════════════════════════════
# 6. CODE CHANGES REQUIRED (For System Administrator)
# ═══════════════════════════════════════════════════════════════════════════════

"""
WHEN RECEIVING A NEW DISEASE MODEL, MAKE THESE CODE CHANGES:

CHANGE 1: app.py - Add to DISEASE_CONFIGS dictionary
───────────────────────────────────────────────────────

Location: Near top of app.py after Flask imports

Add this configuration for each disease:

DISEASE_CONFIGS = {
    "diabetes": {
        "display_name": "Diabetes",
        "model_dir": "models/diabetes",
        "features": ["_BMI5", "_AGEG5YR", "SEX", "GENHLTH", "SMOKE100", "EXERANY2", "CVDINFR4", "CVDCRHD4", "CVDSTRK3", "CHCKIDNY"],
        "field_mappings": {
            "age_group": {
                "18-24": 1, "25-29": 2, ... "80+": 13
            },
            "general_health": {
                "Excellent": 1, "Very Good": 2, "Good": 3, "Fair": 4, "Poor": 5
            },
            # ... [other field mappings]
        },
        "risk_thresholds": {"low": 30, "high": 70},
    },
    "hypertension": {
        "display_name": "Hypertension",
        "model_dir": "models/hypertension",
        "features": [...],
        "field_mappings": {...},
        "risk_thresholds": {...},
    },
    # [ADD MORE DISEASES HERE]
}

CHANGE 2: app.py - Add Dynamic Route Generator
───────────────────────────────────────────────

After DISEASE_CONFIGS definition, add:

@app.route("/predict/<disease>", methods=["POST"])
def predict_disease(disease):
    if disease not in DISEASE_CONFIGS:
        return jsonify({"status": "error", "message": "Unknown disease"}), 404
    
    config = DISEASE_CONFIGS[disease]
    # [Route logic - see full implementation below]

CHANGE 3: index.html - Add Disease Tab
────────────────────────────────────────

In the disease-nav section, add:

<button class="disease-btn inactive" data-disease="hypertension" title="Coming Soon">
    Hypertension (Coming Soon)
</button>

CHANGE 4: index.html - Add Form Fields
───────────────────────────────────────

When disease becomes active, update form fields to match features.
This can be dynamic via JavaScript or separate HTML for each disease.

CHANGE 5: database (if using one)
──────────────────────────────────

Add disease entry to diseases table:

INSERT INTO diseases (disease_id, display_name, model_accuracy, dataset, created_at)
VALUES ('hypertension', 'Hypertension Risk', 0.82, 'BRFSS 2014', NOW());
"""


# ═══════════════════════════════════════════════════════════════════════════════
# 7. PYTHON SCRIPT TO VALIDATE MODEL FILES
# ═══════════════════════════════════════════════════════════════════════════════

"""
VALIDATION SCRIPT - Run this to verify model files are correct:

import joblib
import os

DISEASE = "hypertension"  # Change for each disease
MODEL_DIR = f"models/{DISEASE}"

def validate_model_files(disease_dir):
    print(f"Validating {disease_dir}...")
    
    # Check all files exist
    required_files = [
        "diabetes_model.pkl",
        "diabetes_scaler.pkl",
        "feature_names.pkl",
        "model_meta.pkl"
    ]
    
    for file in required_files:
        file_path = os.path.join(disease_dir, file)
        if not os.path.exists(file_path):
            print(f"  ✗ MISSING: {file}")
            return False
        print(f"  ✓ Found: {file}")
    
    # Validate structure
    try:
        model = joblib.load(os.path.join(disease_dir, f"{DISEASE}_model.pkl"))
        scaler = joblib.load(os.path.join(disease_dir, f"{DISEASE}_scaler.pkl"))
        features = joblib.load(os.path.join(disease_dir, "feature_names.pkl"))
        meta = joblib.load(os.path.join(disease_dir, "model_meta.pkl"))
        
        # Validate types
        assert isinstance(features, list), "feature_names must be a list"
        assert isinstance(meta, dict), "model_meta must be a dict"
        assert "accuracy" in meta, "model_meta must contain 'accuracy'"
        assert "model_name" in meta, "model_meta must contain 'model_name'"
        
        # Validate model has required methods
        assert hasattr(model, "predict"), "model must have predict() method"
        assert hasattr(model, "predict_proba"), "model must have predict_proba() method"
        
        print(f"  ✓ Model validation passed")
        print(f"    - Features: {len(features)}")
        print(f"    - Accuracy: {meta.get('accuracy')}")
        print(f"    - Model: {meta.get('model_name')}")
        return True
    
    except Exception as e:
        print(f"  ✗ Validation failed: {str(e)}")
        return False

if __name__ == "__main__":
    validate_model_files(MODEL_DIR)
"""


# ═══════════════════════════════════════════════════════════════════════════════
# 8. CHECKLIST FOR COLLABORATORS
# ═══════════════════════════════════════════════════════════════════════════════

"""
COLLABORATOR CHECKLIST - BEFORE SENDING MODEL
═══════════════════════════════════════════════

For [DISEASE NAME]:

PRE-SUBMISSION:
  □ Model is trained on clean, validated data
  □ Model has been tested on holdout test set
  □ Model accuracy is documented
  □ Feature names are finalized and ordered

FILES TO DELIVER:
  □ [disease]_model.pkl (saved with joblib.dump())
  □ [disease]_scaler.pkl (saved with joblib.dump())
  □ feature_names.pkl (Python list, correct order)
  □ model_meta.pkl (dict with accuracy, model_name, dataset)

DOCUMENTATION TO PROVIDE:
  □ Disease ID (lowercase_with_underscores)
  □ Display name for UI
  □ Complete list of input features
  □ Mapping for each feature (continuous/categorical/binary)
  □ Risk thresholds (Low, Medium, High percentages)
  □ Model type (LogisticRegression, RandomForest, etc.)
  □ Dataset used for training
  □ Model accuracy on test set
  □ Any known limitations or edge cases
  □ Recommendations thresholds (when to show each recommendation level)

VALIDATION:
  □ Run validation script successfully
  □ Features match model training features exactly
  □ Scaler output shape matches model input shape
  □ predict_proba() returns valid probabilities (0-1)

DELIVERY FORMAT:
  □ Compress as: [disease].zip containing:
     models/[disease]/
     ├── [disease]_model.pkl
     ├── [disease]_scaler.pkl
     ├── feature_names.pkl
     └── model_meta.pkl
  
  □ Include README with:
     - Feature specifications (see template)
     - Risk thresholds
     - Recommendations for each risk level

EXAMPLE ZIP STRUCTURE:
  hypertension.zip
  ├── README.md (with feature specs and thresholds)
  └── models/hypertension/
      ├── hypertension_model.pkl
      ├── hypertension_scaler.pkl
      ├── feature_names.pkl
      └── model_meta.pkl
"""


# ═══════════════════════════════════════════════════════════════════════════════
# 9. EXAMPLE: COMPLETE HYPERTENSION MODEL SUBMISSION
# ═══════════════════════════════════════════════════════════════════════════════

"""
FILE: README.md (Include with submission)
───────────────────────────────────────────

# Hypertension Risk Prediction Model

## Model Information
- **Disease ID**: hypertension
- **Display Name**: Hypertension Risk Prediction
- **Model Type**: Logistic Regression
- **Dataset**: BRFSS 2014
- **Accuracy**: 0.84
- **Precision**: 0.82
- **Recall**: 0.81

## Input Features

| Feature ID | Display Name | Type | BRFSS Name | Details |
|-----------|--------------|------|-----------|---------|
| bmi | Body Mass Index | Continuous | _BMI5 | Range: 10-60, Step: 0.1 |
| age_group | Age Group | Category | _AGEG5YR | 13 options (18-24 to 80+) → Codes 1-13 |
| sex | Sex | Binary | SEX | Male=1, Female=0 |
| general_health | General Health | Ordinal | GENHLTH | Excellent→1, Very Good→2, Good→3, Fair→4, Poor→5 |
| smoking | Smoking History | Binary | SMOKE100 | Yes=1, No=0 |
| exercise | Regular Exercise | Binary | EXERANY2 | Yes=1, No=0 |
| heart_attack | Heart Attack History | Binary | CVDINFR4 | Yes=1, No=0 |
| heart_disease | Heart Disease | Binary | CVDCRHD4 | Yes=1, No=0 |
| stroke | Stroke History | Binary | CVDSTRK3 | Yes=1, No=0 |
| kidney_disease | Kidney Disease | Binary | CHCKIDNY | Yes=1, No=0 |

## Risk Thresholds
- **Low Risk**: 0-30%
- **Medium Risk**: 31-70%
- **High Risk**: 71-100%

## Recommendations

### Low Risk (0-30%)
- Maintain current lifestyle
- Annual blood pressure check
- Reduce sodium intake
- Exercise 150 minutes/week

### Medium Risk (31-70%)
- Consult healthcare provider
- Home blood pressure monitoring
- Reduce sodium and alcohol
- Increase physical activity to 30 min/day

### High Risk (71-100%)
- Schedule immediate medical consultation
- Begin home blood pressure monitoring
- Strict low-sodium diet (<2300 mg/day)
- Discuss antihypertensive medication options
- Daily exercise under medical supervision

## Feature Order (Critical)
["_BMI5", "_AGEG5YR", "SEX", "GENHLTH", "SMOKE100", "EXERANY2", "CVDINFR4", "CVDCRHD4", "CVDSTRK3", "CHCKIDNY"]

## Notes
- Model trained on 50,000 BRFSS 2014 records
- Scaler is StandardScaler fitted to training data
- All features must be provided; no missing values accepted
- Tested with edge cases (BMI 10-60, all age groups)
"""


# ═══════════════════════════════════════════════════════════════════════════════
# 10. QUICK START FOR NEW DISEASE
# ═══════════════════════════════════════════════════════════════════════════════

"""
QUICK INTEGRATION STEPS:

1. RECEIVE MODEL ZIP FROM COLLABORATOR
   └─ Extract to: models/[disease]/
   └─ Verify all 4 pkl files present

2. VALIDATE MODEL
   └─ Run validation script
   └─ Confirm features list is correct
   └─ Test with sample input

3. UPDATE app.py
   └─ Add disease to DISEASE_CONFIGS dict
   └─ Copy feature mappings from collaborator's README

4. UPDATE index.html
   └─ Add disease button to navigation
   └─ Update form fields (or make dynamic)
   └─ Add route handler in JavaScript

5. TEST
   └─ Submit test form
   └─ Verify /predict/[disease] route works
   └─ Check risk levels displayed correctly

6. DEPLOY
   └─ Commit to version control
   └─ Update documentation
   └─ Notify stakeholders

ESTIMATED TIME: 30 minutes per disease
"""
