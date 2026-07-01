"""
GOOGLE COLAB TRAINING PROMPT FOR CLAUDE
════════════════════════════════════════════════════════════════════════════════

Share this prompt with collaborators who need to train models in Google Colab.
They can copy-paste this directly to Claude to generate complete training code.

"""

# ═══════════════════════════════════════════════════════════════════════════════
# TEMPLATE: HYPERTENSION MODEL TRAINING IN GOOGLE COLAB
# ═══════════════════════════════════════════════════════════════════════════════

PROMPT_HYPERTENSION_TRAINING = """
Generate complete Google Colab training code for a Hypertension risk prediction model
using the BRFSS 2014 dataset.

REQUIREMENTS:
─────────────

Dataset:
  - Use the provided BRFSS 2014 dataset (2014.csv).
  - Target: Inspect the 2014.csv file to locate the target variable representing Hypertension (e.g., _RFHYPE5, BPHIGH4, or the appropriate BRFSS field). Preprocess and clean the target variable into a binary indicator (1 = Hypertension present, 0 = Hypertension absent). Be sure to handle BRFSS coding conventions (e.g., mapping 1/2 or other codes, and removing missing/refused values like 7/9).
  - Features: Inspect the 2014.csv file to select 8-12 relevant features for predicting Hypertension (such as _BMI5, _AGEG5YR, SEX, GENHLTH, SMOKE100, EXERANY2, CVDINFR4, CVDCRHD4, CVDSTRK3, CHCKIDNY, or other clinical/lifestyle indicators present in the dataset).
  - Data size: 50,000+ records if possible

Model Requirements:
  - Algorithm: Logistic Regression (or Random Forest/XGBoost)
  - Split: 80% train, 20% test
  - Scaling: StandardScaler for continuous features (fit only on training set)
  - Cross-validation: 5-fold CV for robustness
  - Target accuracy: >70%

Output Files (MUST CREATE):
  1. hypertension_model.pkl       (trained classifier)
  2. hypertension_scaler.pkl      (fitted StandardScaler)
  3. feature_names.pkl            (list of selected features in exact training order)
  4. model_meta.pkl               ({"accuracy": X, "model_name": "...", "dataset": "BRFSS 2014", "disease_id": "hypertension"})

Code Should:
  ✓ Load/prepare data from the uploaded 2014.csv
  ✓ Inspect column headers and types
  ✓ Handle missing/refused values
  ✓ Train model with cross-validation
  ✓ Evaluate on test set
  ✓ Save all 4 pkl files to /content/ (Colab root)
  ✓ Print model accuracy and metrics
  ✓ Include comments explaining each step

Include:
  ✓ Libraries: pandas, numpy, scikit-learn, joblib
  ✓ Data loading & preprocessing
  ✓ Feature engineering if needed
  ✓ Model training with cross-validation
  ✓ Model evaluation (accuracy, precision, recall, F1)
  ✓ Model serialization (joblib.dump for all 4 files)
  ✓ Instructions to download files

After Code, Provide:
  ✓ Instructions on how to download .pkl files from Colab
  ✓ Validation Python code to test the saved model
  ✓ Instructions on how to package for delivery

Important:
  - Feature order MUST be consistent between features list and training data
  - Use joblib (not pickle) to save models
  - Scaler must be fit on training data only (not test data)
  - Model should output probabilities for class 1 (hypertension present)
"""

# ═══════════════════════════════════════════════════════════════════════════════

PROMPT_HEART_DISEASE_TRAINING = """
Generate complete Google Colab training code for a Heart Disease risk prediction model
using the BRFSS 2014 dataset.

REQUIREMENTS:
─────────────

Dataset:
  - Use the provided BRFSS 2014 dataset (2014.csv).
  - Target: Inspect the 2014.csv file to locate the target variable representing Heart Disease (e.g., _MICHD, CVDCRHD4, CVDINFR4, or the appropriate BRFSS field). Preprocess and clean the target variable into a binary indicator (1 = Heart Disease present, 0 = Heart Disease absent). Be sure to handle BRFSS coding conventions (e.g., mapping 1/2 or other codes, and removing missing/refused values like 7/9).
  - Features: Inspect the 2014.csv file to select 8-12 relevant features for predicting Heart Disease (such as _BMI5, _AGEG5YR, SEX, GENHLTH, SMOKE100, EXERANY2, CVDINFR4, CVDCRHD4, CVDSTRK3, CHCKIDNY, or other clinical/lifestyle indicators present in the dataset).
  - Data size: 50,000+ records if possible

Model Requirements:
  - Algorithm: Logistic Regression (or Random Forest/XGBoost)
  - Split: 80% train, 20% test
  - Scaling: StandardScaler for continuous features (fit only on training set)
  - Cross-validation: 5-fold CV for robustness
  - Target accuracy: >70%

Output Files (MUST CREATE):
  1. heart_disease_model.pkl       (trained classifier)
  2. heart_disease_scaler.pkl      (fitted StandardScaler)
  3. feature_names.pkl             (list of selected features in exact training order)
  4. model_meta.pkl                ({"accuracy": X, "model_name": "...", "dataset": "BRFSS 2014", "disease_id": "heart_disease"})

[FOLLOW THE REST OF THE HYPERTENSION REQUIREMENTS ABOVE, REPLACING HYPERTENSION WITH HEART DISEASE]
"""

# ═══════════════════════════════════════════════════════════════════════════════

PROMPT_STROKE_TRAINING = """
Generate complete Google Colab training code for a Stroke risk prediction model
using the BRFSS 2014 dataset.

REQUIREMENTS:
─────────────

Dataset:
  - Use the provided BRFSS 2014 dataset (2014.csv).
  - Target: Inspect the 2014.csv file to locate the target variable representing Stroke (e.g., CVDSTRK3, or the appropriate BRFSS field). Preprocess and clean the target variable into a binary indicator (1 = Stroke present, 0 = Stroke absent). Be sure to handle BRFSS coding conventions (e.g., mapping 1/2 or other codes, and removing missing/refused values like 7/9).
  - Features: Inspect the 2014.csv file to select 8-12 relevant features for predicting Stroke (such as _BMI5, _AGEG5YR, SEX, GENHLTH, SMOKE100, EXERANY2, CVDINFR4, CVDCRHD4, CVDSTRK3, CHCKIDNY, or other clinical/lifestyle indicators present in the dataset).
  - Data size: 50,000+ records if possible

Model Requirements:
  - Algorithm: Logistic Regression (or Random Forest/XGBoost)
  - Split: 80% train, 20% test
  - Scaling: StandardScaler for continuous features (fit only on training set)
  - Cross-validation: 5-fold CV for robustness
  - Target accuracy: >70%

Output Files (MUST CREATE):
  1. stroke_model.pkl       (trained classifier)
  2. stroke_scaler.pkl      (fitted StandardScaler)
  3. feature_names.pkl      (list of selected features in exact training order)
  4. model_meta.pkl         ({"accuracy": X, "model_name": "...", "dataset": "BRFSS 2014", "disease_id": "stroke"})

[FOLLOW THE REST OF THE HYPERTENSION REQUIREMENTS ABOVE, REPLACING HYPERTENSION WITH STROKE]
"""

# ═══════════════════════════════════════════════════════════════════════════════
# COMPLETE STEP-BY-STEP GUIDE FOR GOOGLE COLAB TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

COMPLETE_COLAB_GUIDE = """
GOOGLE COLAB MODEL TRAINING - COMPLETE STEP-BY-STEP GUIDE
══════════════════════════════════════════════════════════════════════════════

STEP 1: CREATE GOOGLE COLAB NOTEBOOK & UPLOAD DATA
───────────────────────────────────────────────────

1. Go to: https://colab.research.google.com
2. Click "New Notebook"
3. Name it: "[Disease] Model Training"
4. Upload `2014.csv` to the notebook runtime:
   - Click the folder icon in the left sidebar of Google Colab.
   - Click the "Upload to session storage" button.
   - Select your copy of `2014.csv` and wait for it to upload.

STEP 2: GET TRAINING CODE FROM CLAUDE
──────────────────────────────────────

Provide Claude with your `2014.csv` (or describe its schema/first few rows if file is too large to upload directly to Claude) and copy-paste this prompt:

---BEGIN PROMPT TO GIVE CLAUDE---

I have provided the `2014.csv` BRFSS dataset. Please write complete Python code to train a [DISEASE] risk prediction model in Google Colab.

REQUIREMENTS:
  - Dataset: Use the provided `2014.csv` dataset.
  - Target variable detection: Inspect the column names and values of the `2014.csv` dataset to find the target variable for [DISEASE] (e.g. `_RFHYPE5` or `BPHIGH4` for Hypertension, `_MICHD` or `CVDCRHD4` for Heart Disease, `CVDSTRK3` for Stroke, or another appropriate variable in the dataset). Preprocess it into a binary target (1 = disease present, 0 = disease absent), mapping the BRFSS values appropriately and filtering out missing, refused, or unknown response codes (e.g., 7, 9).
  - Feature selection: Inspect `2014.csv` to select 8-12 features relevant for predicting [DISEASE] (such as `_BMI5`, `_AGEG5YR`, `SEX`, `GENHLTH`, `SMOKE100`, `EXERANY2`, or other clinical/lifestyle columns available).
  - Algorithm: Logistic Regression or Random Forest/XGBoost.
  - Accuracy target: >70%

MUST OUTPUT 4 FILES (all to /content/):
  1. [disease]_model.pkl       (the trained model classifier)
  2. [disease]_scaler.pkl      (StandardScaler or preprocessor fit on training data only)
  3. feature_names.pkl        (Python list of selected features in the exact order they were fed to the model)
  4. model_meta.pkl           (Python dict with: {"accuracy": X, "model_name": "...", "dataset": "BRFSS 2014", "disease_id": "[disease]"})

MUST INCLUDE:
  ✓ Code to inspect the dataset column names/types
  ✓ Data loading & preprocessing (imputing/dropping missing values, mapping target and feature codes)
  ✓ Feature standardization (StandardScaler fit on training data only)
  ✓ 5-fold cross-validation
  ✓ Model evaluation metrics (Accuracy, Precision, Recall, F1)
  ✓ File serialization with joblib
  ✓ Comments explaining each step
  ✓ Instructions to download files from Colab

IMPORTANT:
  - Use joblib.dump() to save (not pickle)
  - Feature order must be consistent
  - model_meta.pkl must be dict with "accuracy", "model_name", "dataset", "disease_id" keys

---END PROMPT---

3. Claude will generate complete training code
4. Copy the Python code

STEP 3: ADD CODE TO COLAB NOTEBOOK
───────────────────────────────────

1. Go back to Colab notebook
2. Click in first cell
3. Paste the code Claude gave you
4. Run cell (Ctrl+Enter or click ▶ button)
5. Wait for execution (may take 2-5 minutes)

STEP 4: MONITOR EXECUTION
──────────────────────────

While code runs:
  ✓ Watch for error messages
  ✓ Monitor model training progress
  ✓ See accuracy metrics printed
  ✓ Check for "Files saved successfully"

Expected output should show:
  ├─ Columns inspected: [List of selected features]
  ├─ Data loaded: X records
  ├─ Target variable cleaned and mapped: [disease] (1 = Yes, 0 = No)
  ├─ Model trained with 5-fold CV
  ├─ Test accuracy: ~[75-85%]
  ├─ Files saved to /content/
  └─ Download instructions provided

STEP 5: DOWNLOAD FILES FROM COLAB
──────────────────────────────────

In Colab:
  1. Left sidebar → Files icon
  2. Find: [disease]_model.pkl
  3. Right-click → Download
  4. Repeat for all 4 files:
     ├─ [disease]_model.pkl
     ├─ [disease]_scaler.pkl
     ├─ feature_names.pkl
     └─ model_meta.pkl

Save to your computer in a folder

STEP 6: VALIDATE FILES LOCALLY (OPTIONAL)
──────────────────────────────────────────

Run this Python code on your computer:

  import joblib
  import numpy as np
  
  DISEASE = "[disease]"
  
  # Load files
  model = joblib.load(f'{DISEASE}_model.pkl')
  scaler = joblib.load(f'{DISEASE}_scaler.pkl')
  features = joblib.load('feature_names.pkl')
  meta = joblib.load('model_meta.pkl')
  
  print(f"✓ All files loaded")
  print(f"  Features: {features}")
  print(f"  Accuracy: {meta['accuracy']}")
  
  # Test prediction
  sample = np.random.randn(1, len(features))
  scaled = scaler.transform(sample)
  pred = model.predict(scaled)
  prob = model.predict_proba(scaled)
  print(f"✓ Test prediction works: {pred}, {prob}")

STEP 7: WRITE README.MD
───────────────────────

Create a README.md file with:

  # [Disease] Risk Prediction Model
  
  ## Model Info
  - Accuracy: [accuracy from model_meta.pkl]
  - Model Type: [Logistic Regression / Random Forest]
  - Dataset: BRFSS 2014
  - Features: [number] BRFSS fields
  
  ## Features & Mappings
  [Document each of the selected features]
  [Include all user-friendly → BRFSS code mappings]
  [List all categorical options and their numeric codes]
  
  ## Risk Thresholds
  - Low Risk: 0-30%
  - Medium Risk: 31-70%
  - High Risk: 71-100%
  
  ## Recommendations
  [List recommendations for each risk level]
  
  (See EXAMPLE_MODEL_README.md for complete format)

STEP 8: PACKAGE FOR DELIVERY
────────────────────────────

Create folder structure:

  [disease].zip
  ├── README.md (your documentation)
  └── models/[disease]/
      ├── [disease]_model.pkl
      ├── [disease]_scaler.pkl
      ├── feature_names.pkl
      └── model_meta.pkl

Zip and send to system administrator

TROUBLESHOOTING
───────────────

If code fails to run:

  ✗ "Module not found" error
    → Add: !pip install [module_name] as first cell
  
  ✗ "Data not found" error
    → Upload 2014.csv to Colab first (Files → Upload)
  
  ✗ "Memory error"
    → Reduce dataset size (e.g. read first 100,000 rows: pd.read_csv('2014.csv', nrows=100000)) or use Colab GPU
    → Click Runtime → Change runtime type → GPU
  
  ✗ Files not saved
    → Check file paths use /content/
    → Run: !ls /content/ to list files
  
  ✗ Accuracy too low (<70%)
    → Try different algorithm
    → Add/remove features
    → Adjust hyperparameters
    → Ask Claude for help: "Improve model accuracy"

WHAT CLAUDE WILL GENERATE
──────────────────────────

Claude's code will include:

  Cell 1: Install libraries
    !pip install pandas numpy scikit-learn joblib
  
  Cell 2: Load & prepare data
    - Load BRFSS dataset (2014.csv)
    - Inspect column headers and values
    - Extract and clean target variable and features
    - Handle missing values
  
  Cell 3: Train model
    - StandardScaler.fit()
    - Model.fit() with cross-validation
    - Evaluate metrics
  
  Cell 4: Save files
    - joblib.dump(model, ..._model.pkl)
    - joblib.dump(scaler, ..._scaler.pkl)
    - joblib.dump(features, feature_names.pkl)
    - joblib.dump(meta, model_meta.pkl)
  
  Cell 5: Download instructions
    - Show how to download files
    - Provide validation code

TIME ESTIMATE
─────────────

Total time: 30-60 minutes

  ├─ Create Colab notebook & upload 2014.csv: 5 min
  ├─ Get code from Claude: 5 min
  ├─ Paste & run in Colab: 10 min
  ├─ Model training: 5-10 min
  ├─ Download files: 5 min
  ├─ Validate locally: 5 min
  ├─ Write README: 10 min
  └─ Package & deliver: 5 min

NEXT STEPS AFTER TRAINING
──────────────────────────

1. Send [disease].zip to system administrator
2. Include what accuracy/model type you used
3. Admin will integrate into system (follows ADMIN_INTEGRATION_GUIDE.md)
4. Your model goes live in production!
"""

# ═══════════════════════════════════════════════════════════════════════════════
# MINIMAL PROMPT (IF COLLABORATOR IS RUSHED)
# ═══════════════════════════════════════════════════════════════════════════════

MINIMAL_COLAB_PROMPT = """
I need to train a [DISEASE] prediction model in Google Colab. Please write code that:

1. Loads the provided BRFSS 2014 dataset (`2014.csv`).
2. Inspects `2014.csv` to dynamically find the target variable for [DISEASE] (e.g. `_RFHYPE5` for Hypertension, `_MICHD` for Heart Disease, `CVDSTRK3` for Stroke, etc.) and selects 8-12 relevant predictor features (such as `_BMI5`, `_AGEG5YR`, `SEX`, `GENHLTH`, `SMOKE100`, `EXERANY2`, etc.).
3. Preprocesses the target to binary 0/1, handles missing/invalid values, and trains a classifier (Logistic Regression or Random Forest/XGBoost).
4. Saves exactly these 4 files to /content/:
   - [disease]_model.pkl (the trained model)
   - [disease]_scaler.pkl (StandardScaler)
   - feature_names.pkl (list: ["feature1", "feature2", ...])
   - model_meta.pkl (dict: {"accuracy": X, "model_name": "...", "dataset": "BRFSS 2014", "disease_id": "[disease]"})

Use joblib to save. Include comments. Target 70%+ accuracy.
"""

# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 80)
    print("GOOGLE COLAB TRAINING PROMPT LIBRARY")
    print("=" * 80)
    print("\nUse these prompts to generate training code in Google Colab:\n")
    print("\n1. For HYPERTENSION - Copy this prompt to Claude:")
    print("-" * 80)
    print(PROMPT_HYPERTENSION_TRAINING)
    print("\n2. For complete step-by-step guide:")
    print("-" * 80)
    print(COMPLETE_COLAB_GUIDE)
    print("\n3. For quick/minimal prompt:")
    print("-" * 80)
    print(MINIMAL_COLAB_PROMPT)
