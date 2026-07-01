"""
MODEL PROVIDER CHECKLIST (Print & Share with Collaborators)
════════════════════════════════════════════════════════════════════════════════

For Model Provider: [Your Name]
Disease: [DISEASE NAME]
Date: [YYYY-MM-DD]

════════════════════════════════════════════════════════════════════════════════
QUICK START: IF TRAINING IN GOOGLE COLAB
════════════════════════════════════════════════════════════════════════════════

If you don't have a trained model yet and need to train in Google Colab:

1. See: GOOGLE_COLAB_TRAINING_PROMPT.md
2. Copy one of the training prompts
3. Paste into Claude or ChatGPT
4. Claude will generate complete Colab training code
5. Run code in Colab notebook
6. Download 4 pkl files
7. Then continue with PART 1 below

Estimated time: 30-60 minutes to train & export

════════════════════════════════════════════════════════════════════════════════
PART 1: MODEL FILES
════════════════════════════════════════════════════════════════════════════════

□ Step 1: Create model files directory
  mkdir -p models/[disease_id]
  (Example: models/hypertension)

□ Step 2: Export trained model
  joblib.dump(trained_model, 'models/[disease_id]/[disease_id]_model.pkl')
  Verify:
    - Model has .predict() method
    - Model has .predict_proba() method
    - predict_proba() returns [[prob_0, prob_1], ...]

□ Step 3: Export scaler
  joblib.dump(scaler, 'models/[disease_id]/[disease_id]_scaler.pkl')
  Verify:
    - Scaler is fitted to training data
    - Scaler has .transform() method
    - Output features count matches model input

□ Step 4: Export feature names
  feature_list = ["feature1", "feature2", ...]  # In EXACT training order
  joblib.dump(feature_list, 'models/[disease_id]/feature_names.pkl')
  Verify:
    - It's a Python list
    - Order matches your training data columns exactly
    - Length matches model n_features

□ Step 5: Export metadata
  meta = {
      "accuracy": 0.XX,
      "model_name": "Algorithm name",
      "dataset": "BRFSS 2014"
  }
  joblib.dump(meta, 'models/[disease_id]/model_meta.pkl')
  Verify:
    - accuracy is a float between 0 and 1
    - model_name describes the algorithm
    - dataset name is clear

════════════════════════════════════════════════════════════════════════════════
PART 2: FEATURE DOCUMENTATION
════════════════════════════════════════════════════════════════════════════════

For EACH feature in your model, fill out this table:

Feature #1
─────────
  Feature ID:          [user_field_name]         (used in form)
  Display Name:        [What shows on form]
  BRFSS Code:          [_XXXXX or code]          (what model expects)
  Data Type:           ○ Continuous  ○ Category  ○ Binary  ○ Scale
  Input Example:       [e.g., "27.5" or "Male"]
  Valid Range/Options: [e.g., "10-60" or "Male, Female"]
  
  If Continuous:
    □ Min value: ___
    □ Max value: ___
    □ Step: ___
    □ Units: ___
  
  If Categorical/Scale/Binary:
    Mapping:
    ├─ Option 1: "Display Text"  →  [Code: 1]
    ├─ Option 2: "Display Text"  →  [Code: 2]
    ├─ Option 3: "Display Text"  →  [Code: 3]
    └─ ... (ALL options)

[Repeat for each feature]

════════════════════════════════════════════════════════════════════════════════
PART 3: RISK THRESHOLDS & RECOMMENDATIONS
════════════════════════════════════════════════════════════════════════════════

Risk Thresholds (define probability ranges):
─────────────────────────────────────────────

□ LOW RISK
  Probability Range: _____% to _____% 
  (Example: 0% to 30%)
  Color: 🟢 Green
  
□ MEDIUM RISK
  Probability Range: _____% to _____% 
  (Example: 31% to 70%)
  Color: 🟡 Yellow
  
□ HIGH RISK
  Probability Range: _____% to _____% 
  (Example: 71% to 100%)
  Color: 🔴 Red

Recommendations (provide for each risk level):
──────────────────────────────────────────────

LOW RISK RECOMMENDATIONS (for 0-30% probability):
  □ Recommendation 1: _________________________________
  □ Recommendation 2: _________________________________
  □ Recommendation 3: _________________________________
  □ Recommendation 4: _________________________________

MEDIUM RISK RECOMMENDATIONS (for 31-70% probability):
  □ Recommendation 1: _________________________________
  □ Recommendation 2: _________________________________
  □ Recommendation 3: _________________________________
  □ Recommendation 4: _________________________________
  □ Recommendation 5: _________________________________

HIGH RISK RECOMMENDATIONS (for 71-100% probability):
  □ Recommendation 1: _________________________________
  □ Recommendation 2: _________________________________
  □ Recommendation 3: _________________________________
  □ Recommendation 4: _________________________________
  □ Recommendation 5: _________________________________

════════════════════════════════════════════════════════════════════════════════
PART 4: MODEL METADATA
════════════════════════════════════════════════════════════════════════════════

Disease ID:          [disease_id]              (lowercase_with_underscores)
Display Name:        [Friendly name]           (Example: "Hypertension Risk")

Model Information:
  □ Model Type:      [Algorithm name]          (LogisticRegression, RandomForest, etc.)
  □ Accuracy:        _____%                    (on test set)
  □ Precision:       _____%                    (optional)
  □ Recall:          _____%                    (optional)
  □ F1-Score:        _____%                    (optional)

Training Information:
  □ Dataset:         [BRFSS 2014 / Custom]
  □ Training Date:   [YYYY-MM-DD]
  □ Training Size:   [Number of records]
  □ Feature Count:   [Number of features]
  □ Class Balance:   [%Negative / %Positive]

Special Notes:
  □ Known Limitations: ____________________________
  □ Edge Cases:        ____________________________
  □ Improvements:      ____________________________

════════════════════════════════════════════════════════════════════════════════
PART 5: VALIDATION TESTS
════════════════════════════════════════════════════════════════════════════════

Run this Python code before sending:

```python
import joblib
import numpy as np

DISEASE = "[disease_id]"
MODEL_DIR = f"models/{DISEASE}"

# Load all files
print("Loading files...")
model = joblib.load(f'{MODEL_DIR}/{DISEASE}_model.pkl')
scaler = joblib.load(f'{MODEL_DIR}/{DISEASE}_scaler.pkl')
features = joblib.load(f'{MODEL_DIR}/feature_names.pkl')
meta = joblib.load(f'{MODEL_DIR}/model_meta.pkl')

print("Checking structure...")
assert isinstance(features, list), "feature_names must be list"
assert isinstance(meta, dict), "model_meta must be dict"
assert "accuracy" in meta, "accuracy missing"

print("Testing prediction...")
sample = np.random.randn(1, len(features))
scaled = scaler.transform(sample)
pred = model.predict(scaled)
prob = model.predict_proba(scaled)

print(f"✓ All tests passed!")
print(f"  Features: {len(features)}")
print(f"  Accuracy: {meta['accuracy']}")
print(f"  Sample prediction: {pred}, probability: {prob}")
```

□ Validation Script Runs Successfully
  Output:
  ___________________________________________________________________
  
  ___________________________________________________________________

════════════════════════════════════════════════════════════════════════════════
PART 6: DELIVERY PACKAGE
════════════════════════════════════════════════════════════════════════════════

Create this folder structure:

[disease_id].zip
├── README.md
│   └─ Contains all information from PARTS 2-4 above
│   └─ Feature table with all mappings
│   └─ Risk thresholds
│   └─ Recommendations
│   └─ Model metadata
│   └─ Special notes
│
└── models/[disease_id]/
    ├── [disease_id]_model.pkl          [✓ Check]
    ├── [disease_id]_scaler.pkl         [✓ Check]
    ├── feature_names.pkl               [✓ Check]
    └── model_meta.pkl                  [✓ Check]

Example structure for Hypertension:

hypertension.zip
├── README.md
└── models/hypertension/
    ├── hypertension_model.pkl
    ├── hypertension_scaler.pkl
    ├── feature_names.pkl
    └── model_meta.pkl

════════════════════════════════════════════════════════════════════════════════
PART 7: FINAL SUBMISSION CHECKLIST
════════════════════════════════════════════════════════════════════════════════

FILES & FORMAT:
  □ All 4 .pkl files exist in models/[disease_id]/
  □ All files loadable without errors
  □ ZIP file created correctly
  □ README.md complete with all details

FEATURES:
  □ feature_names.pkl contains list of all features
  □ Feature order matches training data exactly
  □ All features mapped to BRFSS codes or custom names
  □ All categorical options mapped to numbers
  □ No features missing mappings

MODEL QUALITY:
  □ predict_proba() returns [[prob_0, prob_1], ...]
  □ predict() returns 0 or 1
  □ Accuracy is reasonable (>0.70)
  □ No model errors on test data

DOCUMENTATION:
  □ README.md includes feature table
  □ All feature mappings complete
  □ Risk thresholds defined
  □ Recommendations provided for each risk level
  □ Model metadata accurate

VALIDATION:
  □ Python validation script runs
  □ All 4 files load successfully
  □ Test prediction works

════════════════════════════════════════════════════════════════════════════════
PART 8: SUBMISSION
════════════════════════════════════════════════════════════════════════════════

BEFORE SENDING:
  □ All checkboxes above are checked
  □ Validation script passed
  □ README is complete and clear
  □ ZIP file is organized correctly

SEND VIA:
  Email to: [INTEGRATION_CONTACT]
  Subject:  [Disease] Model - Ready for Integration
  Attach:   [disease_id].zip
  Include:  Brief summary of model performance

EXAMPLE EMAIL:
──────────────
Subject: Hypertension Model - Ready for Integration

Hi [Contact],

I've prepared the Hypertension risk prediction model for system integration.

Model Details:
- Accuracy: 84%
- Features: 10 BRFSS fields
- Ready for: Immediate integration
- Validation: All tests passed ✓

Attached: hypertension.zip

Please let me know if you need any clarifications or adjustments.

Thank you!
"""
