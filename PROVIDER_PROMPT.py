"""
DISEASE MODEL PROVIDER PROMPT
==============================
Share this prompt with collaborators who are providing trained models for integration.

"""

PROMPT_FOR_HYPERTENSION = """
═══════════════════════════════════════════════════════════════════════════════
HYPERTENSION MODEL PROVIDER INSTRUCTIONS
═══════════════════════════════════════════════════════════════════════════════

PROJECT: Personalised Disease Risk Prediction Using Deep Learning - Integration

WHAT YOU NEED TO PROVIDE:
─────────────────────────

You have trained a Hypertension risk prediction model. We need to integrate it 
into a centralized healthcare prediction platform. 

DELIVERABLES (4 FILES):

1. hypertension_model.pkl
   - Scikit-learn classifier (LogisticRegression, RandomForest, SVM, etc.)
   - Must support .predict() → returns 0 or 1
   - Must support .predict_proba() → returns [[prob_0, prob_1], ...]
   - Saved using: joblib.dump(model, 'hypertension_model.pkl')

2. hypertension_scaler.pkl
   - Fitted StandardScaler or preprocessing object used in training
   - Used to scale new predictions before model inference
   - Saved using: joblib.dump(scaler, 'hypertension_scaler.pkl')

3. feature_names.pkl
   - Python list of feature names in EXACT order used during training
   - Example: ["_BMI5", "_AGEG5YR", "SEX", "GENHLTH", "SMOKE100", "EXERANY2", "CVDINFR4", "CVDCRHD4", "CVDSTRK3", "CHCKIDNY"]
   - Saved using: joblib.dump(features_list, 'feature_names.pkl')
   - ⚠️ CRITICAL: Order must exactly match training data columns

4. model_meta.pkl
   - Python dict with: {"accuracy": 0.xx, "model_name": "...", "dataset": "...", "disease_id": "hypertension"}
   - Example: {
       "accuracy": 0.84,
       "model_name": "Logistic Regression",
       "dataset": "BRFSS 2014",
       "disease_id": "hypertension",
       "trained_date": "2026-06-20"
     }
   - Saved using: joblib.dump(meta_dict, 'model_meta.pkl')

REQUIRED DOCUMENTATION:
─────────────────────────

Provide a README file with:

1. FEATURES TABLE (exactly as your model expects them):
   ┌─ Feature ID: smoking
   ├─ Display Name: Smoking History
   ├─ Data Type: Binary Yes/No
   ├─ BRFSS Field Code: SMOKE100
   ├─ Input Mapping: Yes → 1, No → 0
   └─ User-Facing Option: "Have you smoked 100+ cigarettes in your lifetime?"

   [Repeat for ALL features used in your model]

2. FEATURES CHECKLIST:
   If using the BRFSS 2014 dataset, inspect the file columns to identify features.
   Suggested features to extract from 2014.csv:
   - _BMI5 (Body Mass Index)
   - _AGEG5YR (Age Group)
   - SEX (Sex: Male=1, Female=0)
   - GENHLTH (General Health: Excellent→1 to Poor→5)
   - SMOKE100 (Smoking History: Yes→1, No→0)
   - EXERANY2 (Exercise: Yes→1, No→0)
   - CVDINFR4 (Heart Attack: Yes→1, No→0)
   - CVDCRHD4 (Heart Disease: Yes→1, No→0)
   - CVDSTRK3 (Stroke: Yes→1, No→0)
   - CHCKIDNY (Kidney Disease: Yes→1, No→0)
   
   If your model uses DIFFERENT features, list them all with mappings.

3. FEATURE MAPPINGS (for non-continuous features):
   
   Age Group Dropdown:
   ├─ "18-24" → 1
   ├─ "25-29" → 2
   ├─ "30-34" → 3
   ├─ ...
   └─ "80+" → 13

   General Health Dropdown:
   ├─ "Excellent" → 1
   ├─ "Very Good" → 2
   ├─ "Good" → 3
   ├─ "Fair" → 4
   └─ "Poor" → 5

   [Map every single feature and every single option value]

4. RISK THRESHOLDS:
   ├─ Low Risk: 0-30%
   ├─ Medium Risk: 31-70%
   └─ High Risk: 71-100%
   
   [Adjust these percentages if different for Hypertension]

5. RECOMMENDATIONS FOR EACH RISK LEVEL:
   
   Low Risk (0-30%):
   ├─ [Recommendation 1]
   ├─ [Recommendation 2]
   ├─ [Recommendation 3]
   └─ [Recommendation 4]
   
   Medium Risk (31-70%):
   ├─ [Recommendation 1]
   ├─ [Recommendation 2]
   ├─ [Recommendation 3]
   ├─ [Recommendation 4]
   └─ [Recommendation 5]
   
   High Risk (71-100%):
   ├─ [Recommendation 1]
   ├─ [Recommendation 2]
   ├─ [Recommendation 3]
   ├─ [Recommendation 4]
   └─ [Recommendation 5]

6. MODEL METADATA:
   - Model Accuracy: [X.X%]
   - Model Type: [Logistic Regression / Random Forest / etc.]
   - Dataset Used: [BRFSS 2014 / Custom / etc.]
   - Training Date: [YYYY-MM-DD]
   - Data Size: [Number of training records]
   - Class Balance: [%positive / %negative]
   - Feature Selection Method: [e.g., "All BRFSS features" or "Selected via Chi-square"]

DELIVERY FORMAT:
─────────────────

Create a ZIP file: hypertension.zip

Contents:
├── README.md
│   └─ [Your documentation with all info above]
└── models/hypertension/
    ├── hypertension_model.pkl
    ├── hypertension_scaler.pkl
    ├── feature_names.pkl
    └── model_meta.pkl

VALIDATION (BEFORE SENDING):
─────────────────────────────

Run this Python code to verify your files are correct:

```python
import joblib
import numpy as np

# Load all files
model = joblib.load('hypertension_model.pkl')
scaler = joblib.load('hypertension_scaler.pkl')
features = joblib.load('feature_names.pkl')
meta = joblib.load('model_meta.pkl')

# Verify structure
print("✓ Features loaded:", len(features))
print("✓ Model has predict:", hasattr(model, 'predict'))
print("✓ Model has predict_proba:", hasattr(model, 'predict_proba'))
print("✓ Accuracy:", meta.get('accuracy'))

# Test prediction
sample = np.random.randn(1, len(features))
scaled = scaler.transform(sample)
pred = model.predict(scaled)
prob = model.predict_proba(scaled)
print("✓ Prediction successful:", pred, prob)
```

INTEGRATION CHECKLIST:
──────────────────────

Before submitting, verify:
  □ All 4 .pkl files created and loadable
  □ feature_names.pkl list order matches your training columns exactly
  □ model.predict() returns 0 or 1
  □ model.predict_proba() returns [[prob_0, prob_1], ...]
  □ scaler.transform() output shape matches expected features
  □ All feature mappings are complete (no missing options)
  □ Risk thresholds are defined
  □ Recommendations provided for each risk level
  □ Model accuracy is accurate
  □ README with all documentation is complete
  □ ZIP file is organized correctly

QUESTIONS?
───────────

If you're unsure about:
- Feature order → Check your training script. Features must be in exact order.
- Mappings → Check all unique values in each feature. Every option must map to a number.
- Pickle format → Use joblib, not pickle: `joblib.dump(object, 'file.pkl')`
- Scaling → Include whatever preprocessing you used (StandardScaler, MinMaxScaler, etc.)

EXAMPLE DELIVERY:
──────────────────

Subject: Hypertension Model Ready for Integration

Attached: hypertension.zip

Metadata:
- Disease: Hypertension Risk Prediction
- Accuracy: 84%
- Features: 10 (listed in README)
- Ready for: Immediate integration

Please let me know if you need any clarification or have questions about the format.
"""

# ═══════════════════════════════════════════════════════════════════════════════

TRAINING_IN_GOOGLE_COLAB_REFERENCE = """
═══════════════════════════════════════════════════════════════════════════════
TRAINING YOUR MODEL IN GOOGLE COLAB
═══════════════════════════════════════════════════════════════════════════════

If you're training your model in Google Colab:

1. Go to: https://colab.research.google.com
2. Create new notebook
3. Upload your `2014.csv` file to Colab.
4. Copy this prompt and paste to Claude (feed the model details or file path):

───────────────────────────────────────────────────────────────────────────────
[See GOOGLE_COLAB_TRAINING_PROMPT.md for complete Colab training prompts]
───────────────────────────────────────────────────────────────────────────────

Claude will generate complete code that:
  ✓ Inspects the provided `2014.csv` dataset dynamically to identify target and feature columns
  ✓ Cleans the target variable (converting to binary classification) and feature columns
  ✓ Trains a classifier
  ✓ Exports 4 pkl files automatically
  ✓ Includes download instructions

Then follow: GOOGLE_COLAB_TRAINING_PROMPT.md → COMPLETE_COLAB_GUIDE

After training, you'll have all 4 files needed for delivery.
"""

# ═══════════════════════════════════════════════════════════════════════════════

PROMPT_FOR_HEART_DISEASE = """
═══════════════════════════════════════════════════════════════════════════════
HEART DISEASE MODEL PROVIDER INSTRUCTIONS
═══════════════════════════════════════════════════════════════════════════════

[SAME AS HYPERTENSION ABOVE - Replace all "hypertension" with "heart_disease"]
[Replace all "Hypertension" with "Heart Disease"]
[Adjust disease-specific thresholds and recommendations]

MINIMAL TEMPLATE:

DELIVERABLES (4 FILES):
1. heart_disease_model.pkl
2. heart_disease_scaler.pkl
3. feature_names.pkl
4. model_meta.pkl

DELIVERY PACKAGE:
├── README.md (with all feature mappings, thresholds, recommendations)
└── models/heart_disease/
    ├── heart_disease_model.pkl
    ├── heart_disease_scaler.pkl
    ├── feature_names.pkl
    └── model_meta.pkl

NOTE: If training in Google Colab, see GOOGLE_COLAB_TRAINING_PROMPT.md
"""

# ═══════════════════════════════════════════════════════════════════════════════

PROMPT_FOR_STROKE = """
═══════════════════════════════════════════════════════════════════════════════
STROKE MODEL PROVIDER INSTRUCTIONS
═══════════════════════════════════════════════════════════════════════════════

[SAME AS HYPERTENSION ABOVE - Replace all "hypertension" with "stroke"]
[Replace all "Hypertension" with "Stroke"]
[Adjust disease-specific thresholds and recommendations]

MINIMAL TEMPLATE:

DELIVERABLES (4 FILES):
1. stroke_model.pkl
2. stroke_scaler.pkl
3. feature_names.pkl
4. model_meta.pkl

DELIVERY PACKAGE:
├── README.md (with all feature mappings, thresholds, recommendations)
└── models/stroke/
    ├── stroke_model.pkl
    ├── stroke_scaler.pkl
    ├── feature_names.pkl
    └── model_meta.pkl
"""

# ═══════════════════════════════════════════════════════════════════════════════

QUICK_REFERENCE_TABLE = """
QUICK REFERENCE: WHAT TO SEND FOR EACH DISEASE
═══════════════════════════════════════════════════════════════════════════════

Disease: [DISEASE_NAME]
├── Disease ID: [disease_id]  (lowercase, underscores)
├── Files (4 required):
│   ├── [disease_id]_model.pkl
│   ├── [disease_id]_scaler.pkl
│   ├── feature_names.pkl
│   └── model_meta.pkl
├── Directory: models/[disease_id]/
├── Package: [disease_id].zip
└── Contains:
    ├── models/[disease_id]/
    │   └─ [4 pkl files]
    └── README.md
        ├─ Feature table (all features, all mappings)
        ├─ Risk thresholds (Low, Medium, High %)
        ├─ Recommendations (for each risk level)
        ├─ Model accuracy
        └─ Any special notes

EXAMPLE FOR HYPERTENSION:
├── Disease ID: hypertension
├── Files: hypertension_model.pkl, hypertension_scaler.pkl, feature_names.pkl, model_meta.pkl
├── Directory: models/hypertension/
├── Package: hypertension.zip
└── ZIP Contents:
    ├── models/hypertension/[4 files]
    └── README.md [with features, thresholds, recommendations]
"""

# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("DISEASE MODEL PROVIDER PROMPTS")
    print("=" * 80)
    print("\nUse these prompts to instruct collaborators on what to provide.")
    print("\nOptions:")
    print("1. Share PROMPT_FOR_HYPERTENSION with Hypertension model trainer")
    print("2. Share PROMPT_FOR_HEART_DISEASE with Heart Disease model trainer")
    print("3. Share PROMPT_FOR_STROKE with Stroke model trainer")
    print("4. Reference QUICK_REFERENCE_TABLE for quick overview")
