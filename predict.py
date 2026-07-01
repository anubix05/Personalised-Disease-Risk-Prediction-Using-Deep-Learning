"""
predict.py — Centralized Risk Prediction Module
Dataset  : BRFSS 2014
Part of  : Personalised Disease Risk Prediction Using Deep Learning
"""

import numpy as np
import joblib
import os
import json
import sys

# Workaround for scikit-learn version mismatches when loading models trained in other versions
try:
    import sklearn._loss._loss as sklearn_loss
    sys.modules['_loss'] = sklearn_loss
except ImportError:
    pass

# Configure Keras to use JAX backend (lightweight, no TensorFlow needed)
os.environ.setdefault('KERAS_BACKEND', 'jax')

try:
    import keras

    # Register the custom focal_loss used by the stroke model during training
    # The model config references 'StrokeModel>focal_loss', so package must match
    @keras.saving.register_keras_serializable(package='StrokeModel', name='focal_loss')
    def focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
        """Focal loss for imbalanced binary classification (stroke)."""
        import keras.ops as ops
        y_pred = ops.clip(y_pred, 1e-7, 1.0 - 1e-7)
        bce = -(y_true * ops.log(y_pred) + (1 - y_true) * ops.log(1 - y_pred))
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        focal_weight = alpha * ops.power(1 - p_t, gamma)
        return ops.mean(focal_weight * bce)
except ImportError:
    # Keras not installed — stroke model won't be available but other models still work
    pass

BASE_DIR = os.path.dirname(__file__)

# Global cache for loaded model assets
_assets_cache = {}

def _load_assets(disease: str):
    """Load and cache model assets for a specific disease."""
    if disease in _assets_cache:
        return _assets_cache[disease]
    
    # Resolve directories based on disease
    if disease == "diabetes":
        model_dir = os.path.join(BASE_DIR, "models", "diabetes")
        model_file = "diabetes_model.pkl"
        scaler_file = "diabetes_scaler.pkl"
        feats_file = "feature_names.pkl"
        meta_file = "model_meta.pkl"
    elif disease == "heart_disease":
        model_dir = os.path.join(BASE_DIR, "models", "heartDisease")
        model_file = "heart_disease_rf.pkl"
        scaler_file = "heart_disease_scaler.pkl"
        feats_file = "feature_columns.pkl"
        meta_file = "model_meta.pkl" 
    elif disease == "stroke":
        model_dir = os.path.join(BASE_DIR, "models", "stroke")
        model_file = "stroke_model.pkl"
        scaler_file = "stroke_scaler.pkl"
        feats_file = "stroke_features.pkl"
        meta_file = "model_meta.pkl"
    elif disease == "asthma":
        model_dir = os.path.join(BASE_DIR, "models", "asthma")
        model_file = "asthma_model.pkl"
        scaler_file = "asthma_scaler.pkl"
        feats_file = "feature_names.pkl"
        meta_file = "model_meta.pkl"
    elif disease == "kidney_disease":
        model_dir = os.path.join(BASE_DIR, "models", "kidneyDisease")
        model_file = "kidney_model.pkl"
        scaler_file = "kidney_scaler.pkl"
        feats_file = "kidney_feature_columns.pkl"
        meta_file = "kidney_metrics.pkl"
    elif disease == "arthritis":
        model_dir = os.path.join(BASE_DIR, "models", "arthritis")
        model_file = "arthritis_xgb.pkl"
        scaler_file = "arthritis_scaler.pkl"
        feats_file = "arthritis_features.pkl"
        meta_file = "model_meta.pkl"
    else:
        raise ValueError(f"Unknown disease: {disease}")

    # Check files exist
    model_path = os.path.join(model_dir, model_file)
    scaler_path = os.path.join(model_dir, scaler_file)
    feats_path = os.path.join(model_dir, feats_file)
    meta_path = os.path.join(model_dir, meta_file)

    # Specific check for missing stroke model file
    if disease == "stroke":
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                "Stroke classifier model file (stroke_model.pkl) is missing from the server. "
                "Please ensure your collaborator provides the trained model file and uploads it to 'models/stroke/'."
            )
        # Check if keras is available
        try:
            import keras
        except ImportError:
            raise ImportError(
                "The stroke prediction model requires the 'keras' package, which is not installed in the currently running Python environment. "
                "Please run the application using the project's virtual environment: `.venv\\Scripts\\python app.py`"
            )

    # General check for model file
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_file} not found at {model_path}")

    # Load assets
    model = joblib.load(model_path)
    
    scaler = None
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
    
    if not os.path.exists(feats_path):
        raise FileNotFoundError(f"Feature columns file not found at {feats_path}")
    feats = joblib.load(feats_path)
    
    # Load metadata or metrics
    meta = {}
    if os.path.exists(meta_path):
        meta = joblib.load(meta_path)
    elif disease == "heart_disease":
        metrics_path = os.path.join(model_dir, "model_metrics.pkl")
        if os.path.exists(metrics_path):
            meta = joblib.load(metrics_path)

    # Handle decimal vs percentage accuracy representation for all diseases
    if "accuracy" in meta:
        raw_acc = meta["accuracy"]
        if isinstance(raw_acc, (int, float, np.float64)):
            meta["accuracy"] = float(raw_acc) / 100.0 if raw_acc > 1.0 else float(raw_acc)

    # Custom threshold lookup
    threshold = 0.5
    if disease == "heart_disease":
        thresh_path = os.path.join(model_dir, "best_threshold.pkl")
        if os.path.exists(thresh_path):
            threshold = float(joblib.load(thresh_path))
    elif disease == "stroke":
        thresh_path = os.path.join(model_dir, "stroke_threshold.pkl")
        if os.path.exists(thresh_path):
            threshold = float(joblib.load(thresh_path))
    elif disease == "kidney_disease":
        thresh_path = os.path.join(model_dir, "kidney_threshold.pkl")
        if os.path.exists(thresh_path):
            threshold = float(joblib.load(thresh_path))
    elif disease == "arthritis":
        thresh_path = os.path.join(model_dir, "arthritis_threshold.pkl")
        if os.path.exists(thresh_path):
            threshold = float(joblib.load(thresh_path))

    assets = {
        "model": model,
        "scaler": scaler,
        "features": feats,
        "meta": meta,
        "threshold": threshold
    }
    
    _assets_cache[disease] = assets
    return assets

def _get_risk_level(prob: float, disease: str, threshold: float = 0.5) -> str:
    """Determine risk level (Low, Medium, High) based on probability and disease thresholds."""
    # Convert threshold to percentage scale
    thresh_pct = threshold * 100
    
    # Custom threshold calculation
    if disease == "stroke":
        # Stroke model output peaks around 78-79% even with maximum risk factors.
        # Let's adjust risk ranges: Low < 45%, Medium 45% - 75%, High > 75%
        if prob < 45.0:
            return "Low Risk"
        elif prob <= 75.0:
            return "Medium Risk"
        return "High Risk"

    if prob < (thresh_pct * 0.6):
        return "Low Risk"
    elif prob <= (thresh_pct * 1.2):
        return "Medium Risk"
    return "High Risk"

def _get_recommendations(risk_level: str, disease: str) -> list:
    """Return personalized medical recommendations based on risk level and disease."""
    recs = {
        "diabetes": {
            "Low Risk": [
                "Maintain balanced diet with vegetables and whole grains.",
                "150 min/week of moderate aerobic exercise.",
                "Annual fasting blood glucose check.",
                "Stay well-hydrated (8+ glasses of water daily)."
            ],
            "Medium Risk": [
                "Consult your doctor for a glucose tolerance test.",
                "Reduce refined carbohydrates and sugary foods.",
                "Target 5-7% body weight reduction.",
                "30 min moderate exercise 5x per week.",
                "Monitor blood glucose every 3-6 months."
            ],
            "High Risk": [
                "Schedule physician consultation immediately.",
                "Request HbA1c and fasting glucose tests urgently.",
                "Adopt low-GI, high-fibre diet immediately.",
                "Discuss preventive medication options with your doctor.",
                "Begin gentle daily exercise under medical guidance."
            ]
        },
        "heart_disease": {
            "Low Risk": [
                "Maintain a heart-healthy diet low in saturated fats and sodium.",
                "Aim for at least 150 minutes of moderate cardiovascular exercise per week.",
                "Continue annual check-ups and monitor cholesterol and blood pressure.",
                "Keep stress levels managed and avoid tobacco products."
            ],
            "Medium Risk": [
                "Consult a physician or cardiologist for a routine cardiovascular workup.",
                "Limit intake of processed foods, red meat, and sodium (< 2,300 mg/day).",
                "Increase daily steps and incorporate brisk walking or cycling.",
                "Monitor blood pressure and resting heart rate weekly at home."
            ],
            "High Risk": [
                "Schedule an urgent medical consultation with a cardiologist.",
                "Request a comprehensive cardiac assessment (ECG, stress test, lipid panel).",
                "Adopt a strict DASH or Mediterranean style diet.",
                "Strictly avoid smoking, second-hand smoke, and limit alcohol completely.",
                "Discuss blood pressure or cholesterol-lowering medications with your doctor."
            ]
        },
        "stroke": {
            "Low Risk": [
                "Maintain active lifestyle and monitor blood pressure annually.",
                "Include healthy fats (olive oil, avocados) and limit sodium.",
                "Ensure consistent high-quality sleep of 7-8 hours per night."
            ],
            "Medium Risk": [
                "Schedule a check-up to evaluate blood pressure and arterial health.",
                "Monitor sodium intake strictly and increase dietary potassium.",
                "Discuss weight management and daily exercise targets with your physician."
            ],
            "High Risk": [
                "Consult your doctor immediately to check for stroke indicators (hypertension, AFib).",
                "Learn the FAST stroke warning signs (Face, Arm, Speech, Time).",
                "Strictly control hypertension and manage diabetes if present.",
                "Avoid high-stress situations and engage in low-impact daily movement."
            ]
        },
        "asthma": {
            "Low Risk": [
                "Avoid known triggers such as pollen, dust mites, or pet dander.",
                "Maintain active lifestyle but carry a rescue inhaler if prescribed.",
                "Ensure home ventilation is good and air filters are clean."
            ],
            "Medium Risk": [
                "Schedule a review with your physician for an asthma action plan.",
                "Monitor your lung function or peak flow regularly if recommended.",
                "Keep a log of symptoms and use of quick-relief medications."
            ],
            "High Risk": [
                "Consult a pulmonologist or primary care physician immediately.",
                "Verify correct inhaler technique and adherence to daily controller medication.",
                "Avoid strenuous outdoor activities during high pollen or poor air quality days.",
                "Seek immediate emergency care if you experience severe shortness of breath."
            ]
        },
        "kidney_disease": {
            "Low Risk": [
                "Maintain adequate daily hydration (around 2 liters of water).",
                "Maintain a balanced diet, limiting excessive sodium and high-protein intake.",
                "Monitor blood pressure and blood sugar annually, as hypertension and diabetes are key risk factors."
            ],
            "Medium Risk": [
                "Consult a healthcare professional to check kidney function (e.g., eGFR and urine albumin-to-creatinine ratio).",
                "Manage blood pressure closely (aim for < 130/80 mmHg) and monitor blood sugar levels.",
                "Limit the use of over-the-counter NSAIDs (like ibuprofen or naproxen) which can impact kidney function.",
                "Restrict dietary sodium to under 2,300 mg per day."
            ],
            "High Risk": [
                "Schedule an appointment with a nephrologist or your primary care physician immediately for a comprehensive kidney health evaluation.",
                "Strictly control blood pressure and manage diabetes under medical supervision.",
                "Review all prescription and over-the-counter medications to avoid nephrotoxic drugs.",
                "Work with a renal dietitian to restrict sodium, phosphorus, and potassium if necessary."
            ]
        },
        "arthritis": {
            "Low Risk": [
                "Maintain a regular low-impact exercise routine (brisk walking, swimming, cycling).",
                "Keep body weight within a healthy range to reduce stress on joints.",
                "Incorporate anti-inflammatory foods like omega-3 rich fish, nuts, and leafy greens.",
                "Stay active and avoid long periods of sitting or standing in one position."
            ],
            "Medium Risk": [
                "Consult a doctor or physical therapist for joint-friendly strength training exercises.",
                "Perform daily range-of-motion stretching exercises to maintain joint flexibility.",
                "Use hot/cold therapy to manage minor joint stiffness or aches.",
                "Monitor joint symptoms and track any stiffness or pain, especially in the morning.",
                "Avoid high-impact activities (e.g., running on hard surfaces) that stress knee and hip joints."
            ],
            "High Risk": [
                "Schedule a consultation with a rheumatologist or primary care physician for a comprehensive joint assessment.",
                "Request diagnostic evaluation (e.g., X-rays or joint fluid analysis) if experiencing chronic swelling or pain.",
                "Discuss medical management options, including anti-inflammatory medications or joint supplements, with your physician.",
                "Adopt a tailored physical therapy program to support joint stability and range of motion.",
                "Verify joint protection techniques (e.g., using assistive devices, ergonomic workspaces) to minimize daily joint strain."
            ]
        }
    }
    return recs.get(disease, {}).get(risk_level, [])

def _engineer_features(disease: str, input_dict: dict) -> dict:
    """Preprocess and engineer custom features from raw inputs if needed."""
    output_dict = input_dict.copy()
    
    if disease in ["heart_disease", "kidney_disease"]:
        # Extract raw inputs
        age = float(input_dict.get('_AGE80', 45))
        bmi = float(input_dict.get('_BMI5', 25.0))
        genhlth = float(input_dict.get('GENHLTH', 3))
        physhlth = float(input_dict.get('PHYSHLTH', 0))
        menthlth = float(input_dict.get('MENTHLTH', 0))
        exer = float(input_dict.get('EXERANY2', 1))
        smoke100 = float(input_dict.get('SMOKE100', 0))
        drnkany = float(input_dict.get('DRNKANY5', 0))
        
        # 1. AGE_GROUP (1-13)
        bins_age = [0, 24, 29, 34, 39, 44, 49, 54, 59, 64, 69, 74, 79, 150]
        age_group = 1
        for idx in range(len(bins_age) - 1):
            if bins_age[idx] < age <= bins_age[idx+1]:
                age_group = idx + 1
                break
        
        # 2. BMI_CATEGORY (1-4)
        bmi_cat = 2
        if bmi < 18.5:
            bmi_cat = 1
        elif bmi < 25.0:
            bmi_cat = 2
        elif bmi < 30.0:
            bmi_cat = 3
        else:
            bmi_cat = 4
            
        # 3. HEALTH_SCORE = PHYSHLTH + MENTHLTH + GENHLTH * 2
        health_score = physhlth + menthlth + genhlth * 2
        
        # 4. LIFESTYLE_RISK = SMOKE100 + (1 - EXERANY2) + DRNKANY5
        sedentary = 1.0 - exer
        lifestyle_risk = smoke100 + sedentary + drnkany
        
        output_dict['AGE_GROUP'] = age_group
        output_dict['BMI_CATEGORY'] = bmi_cat
        output_dict['HEALTH_SCORE'] = health_score
        output_dict['LIFESTYLE_RISK'] = lifestyle_risk
        
    elif disease == "asthma":
        # Extract raw age
        age = float(input_dict.get('_AGE80', 45))
        
        # Six categories for _AGE_G:
        # 1: 18 to 24
        # 2: 25 to 34
        # 3: 35 to 44
        # 4: 45 to 54
        # 5: 55 to 64
        # 6: 65 or older
        if age <= 24:
            age_g = 1
        elif age <= 34:
            age_g = 2
        elif age <= 44:
            age_g = 3
        elif age <= 54:
            age_g = 4
        elif age <= 64:
            age_g = 5
        else:
            age_g = 6
            
        output_dict['_AGE_G'] = age_g
        
    elif disease == "stroke":
        # Extract raw inputs (expecting standard BRFSS format keys from app.py)
        age = float(input_dict.get('_AGE80', 45))
        sex = float(input_dict.get('SEX', 1))  # Male=1, Female=0 in app.py
        bmi = float(input_dict.get('_BMI5', 25.0))
        smoker_status = float(input_dict.get('_SMOKER3', 4))
        alcohol = float(input_dict.get('DRNKANY5', 0))  # Yes=1, No=0 in app.py
        exercise = float(input_dict.get('EXERANY2', 1))  # Yes=1, No=0 in app.py
        sleep = float(input_dict.get('SLEPTIM1', 7))
        diabetes = float(input_dict.get('DIABETE3', 0))  # Yes=1, No=0 in app.py
        heart_disease = float(input_dict.get('CVDCRHD4', 0))  # Yes=1, No=0 in app.py
        afib = float(input_dict.get('AFib', 0))  # Yes=1, No=0 in app.py
        diff_walking = float(input_dict.get('DiffWalking', 0))  # Yes=1, No=0 in app.py
        genhlth = float(input_dict.get('GENHLTH', 3))
        income = float(input_dict.get('INCOME2', 5))
        education = float(input_dict.get('EDUCA', 4))

        # 1. Age (1-6 categories matching _AGE_G)
        if age <= 24:
            age_g = 1
        elif age <= 34:
            age_g = 2
        elif age <= 44:
            age_g = 3
        elif age <= 54:
            age_g = 4
        elif age <= 64:
            age_g = 5
        else:
            age_g = 6

        # 2. Gender (Male=1, Female=2)
        gender_val = 2.0 if sex == 0.0 else 1.0

        # 3. BMI (BMI * 100)
        bmi_val = bmi * 100.0

        # 4. Smoking (1-4 smoker status)
        smoking_val = smoker_status

        # 5. AlcoholConsumption (Yes=1, No=2)
        alcohol_val = 1.0 if alcohol == 1.0 else 2.0

        # 6. PhysicalActivity (Yes=1, No=2)
        exercise_val = 1.0 if exercise == 1.0 else 2.0

        # 7. SleepDuration (continuous)
        sleep_val = sleep

        # 8. Diabetes (Yes=1, No=3)
        diabetes_val = 1.0 if diabetes == 1.0 else 3.0

        # 9. HeartDisease (Yes=1, No=2)
        heart_disease_val = 1.0 if heart_disease == 1.0 else 2.0

        # 10. AFib (Yes=1, No=2)
        afib_val = 1.0 if afib == 1.0 else 2.0

        # 11. DiffWalking (Yes=1, No=2)
        diff_walking_val = 1.0 if diff_walking == 1.0 else 2.0

        # 12. GeneralHealth (1-5)
        genhlth_val = genhlth

        # 13. Income (1-8)
        income_val = income

        # 14. Education (1-6)
        education_val = education

        output_dict['Age'] = age_g
        output_dict['Gender'] = gender_val
        output_dict['BMI'] = bmi_val
        output_dict['Smoking'] = smoking_val
        output_dict['AlcoholConsumption'] = alcohol_val
        output_dict['PhysicalActivity'] = exercise_val
        output_dict['SleepDuration'] = sleep_val
        output_dict['Diabetes'] = diabetes_val
        output_dict['HeartDisease'] = heart_disease_val
        output_dict['AFib'] = afib_val
        output_dict['DiffWalking'] = diff_walking_val
        output_dict['GeneralHealth'] = genhlth_val
        output_dict['Income'] = income_val
        output_dict['Education'] = education_val
        
    elif disease == "arthritis":
        # Extract raw inputs (expecting standard BRFSS format keys from app.py)
        age = float(input_dict.get('_AGE80', 45))
        sex = float(input_dict.get('SEX', 1))  # Male=1, Female=0 in app.py
        bmi = float(input_dict.get('_BMI5', 25.0))
        smoker_status = float(input_dict.get('_SMOKER3', 4))
        alcohol = float(input_dict.get('DRNKANY5', 0))  # Yes=1, No=0 in app.py
        exercise = float(input_dict.get('EXERANY2', 1))  # Yes=1, No=0 in app.py
        sleep = float(input_dict.get('SLEPTIM1', 7))
        diabetes = float(input_dict.get('DIABETE3', 0))  # Yes=1, No=0 in app.py
        heart_disease = float(input_dict.get('CVDCRHD4', 0))  # Yes=1, No=0 in app.py
        physhlth = float(input_dict.get('PHYSHLTH', 0))
        diff_walking = float(input_dict.get('DiffWalking', 0))  # Yes=1, No=0 in app.py
        genhlth = float(input_dict.get('GENHLTH', 3))
        income = float(input_dict.get('INCOME2', 5))
        education = float(input_dict.get('EDUCA', 4))

        # 1. Age (1-6 categories matching _AGE_G)
        if age <= 24:
            age_g = 1
        elif age <= 34:
            age_g = 2
        elif age <= 44:
            age_g = 3
        elif age <= 54:
            age_g = 4
        elif age <= 64:
            age_g = 5
        else:
            age_g = 6

        # 2. Gender (Male=1, Female=2)
        gender_val = 2.0 if sex == 0.0 else 1.0

        # 3. BMI (BMI * 100)
        bmi_val = bmi * 100.0

        # 4. Smoking (1-4 smoker status)
        smoking_val = smoker_status

        # 5. AlcoholConsumption (Yes=1, No=2)
        alcohol_val = 1.0 if alcohol == 1.0 else 2.0

        # 6. PhysicalActivity (Yes=1, No=2)
        exercise_val = 1.0 if exercise == 1.0 else 2.0

        # 7. SleepDuration (continuous)
        sleep_val = sleep

        # 8. Diabetes (Yes=1, No=3)
        diabetes_val = 1.0 if diabetes == 1.0 else 3.0

        # 9. HeartDisease (Yes=1, No=2)
        heart_disease_val = 1.0 if heart_disease == 1.0 else 2.0

        # 10. JointPain (0 days mapped to 88.0, 1-30 mapped as is)
        joint_pain_val = 88.0 if physhlth == 0.0 else physhlth

        # 11. DiffWalking (Yes=1, No=2)
        diff_walking_val = 1.0 if diff_walking == 1.0 else 2.0

        # 12. GeneralHealth (1-5)
        genhlth_val = genhlth

        # 13. Income (1-8)
        income_val = income

        # 14. Education (1-6)
        education_val = education

        output_dict['Age'] = age_g
        output_dict['Gender'] = gender_val
        output_dict['BMI'] = bmi_val
        output_dict['Smoking'] = smoking_val
        output_dict['AlcoholConsumption'] = alcohol_val
        output_dict['PhysicalActivity'] = exercise_val
        output_dict['SleepDuration'] = sleep_val
        output_dict['Diabetes'] = diabetes_val
        output_dict['HeartDisease'] = heart_disease_val
        output_dict['JointPain'] = joint_pain_val
        output_dict['DiffWalking'] = diff_walking_val
        output_dict['GeneralHealth'] = genhlth_val
        output_dict['Income'] = income_val
        output_dict['Education'] = education_val
        
    return output_dict

def predict_full(disease: str, input_dict: dict) -> dict:
    """
    Main prediction function supporting multiple diseases.

    Args:
        disease: String identifier ('diabetes', 'heart_disease', 'stroke', 'asthma')
        input_dict: Patient values as a dict. Keys must match expected features.

    Returns:
        JSON-serialisable dict.
    """
    # Load model and scaler assets
    assets = _load_assets(disease)
    model = assets["model"]
    scaler = assets["scaler"]
    features = assets["features"]
    meta = assets["meta"]
    threshold = assets["threshold"]

    # Preprocess and calculate custom engineered features if needed
    cleaned_input = _engineer_features(disease, input_dict)

    # Order the inputs according to the model's feature column expectations
    ordered = [float(cleaned_input.get(f, 0)) for f in features]
    arr = np.array(ordered).reshape(1, -1)

    # Apply StandardScaler if present
    if scaler is not None:
        scaled = scaler.transform(arr)
    else:
        scaled = arr

    # Run predictions — handle both sklearn (predict_proba) and Keras (predict) models
    if hasattr(model, 'predict_proba'):
        # Scikit-learn style model
        prob = round(float(model.predict_proba(scaled)[0][1] * 100), 2)
    else:
        # Keras model — predict() returns sigmoid probabilities directly
        raw_pred = model.predict(scaled, verbose=0)
        prob = round(float(raw_pred[0][0] * 100), 2)
    
    # Classify based on the Best Threshold
    pred = 1 if (prob / 100.0) >= threshold else 0
    level = _get_risk_level(prob, disease, threshold)

    # For UI display
    display_prediction = "High Risk" if pred == 1 else "Low Risk"

    return {
        "prediction": display_prediction,
        "risk_percentage": prob,
        "risk_level": level,
        "accuracy": meta.get("accuracy", 0.75),
        "model_name": meta.get("model_name", "Model"),
        "dataset": meta.get("dataset", "BRFSS 2014"),
        "recommendations": _get_recommendations(level, disease),
        "risk_factors": [],
        "status": "success",
    }

# Standalone test
if __name__ == "__main__":
    print("Testing Diabetes Model:")
    sample_dia = {
        "_BMI5": 27.5, "_AGEG5YR": 7, "SEX": 1,
        "GENHLTH": 3, "SMOKE100": 1, "EXERANY2": 1,
        "CVDINFR4": 0, "CVDCRHD4": 0, "CVDSTRK3": 0, "CHCKIDNY": 0,
    }
    print(json.dumps(predict_full("diabetes", sample_dia), indent=2))
    
    print("\nTesting Heart Disease Model:")
    sample_hd = {
        "_AGE80": 65, "SEX": 1, "GENHLTH": 3, "PHYSHLTH": 5, "MENTHLTH": 2, 
        "EXERANY2": 0, "SLEPTIM1": 6, "_BMI5": 28.5, "SMOKE100": 1, "_SMOKER3": 1, 
        "DRNKANY5": 1, "DIABETE3": 3, "INCOME2": 5, "EDUCA": 4, "MARITAL": 1
    }
    print(json.dumps(predict_full("heart_disease", sample_hd), indent=2))

    print("\nTesting Asthma Model:")
    sample_asthma = {
        "_AGE80": 30, "SEX": 0, "_BMI5": 22.0, "SMOKE100": 0, "EXERANY2": 1,
        "SLEPTIM1": 8, "GENHLTH": 2, "PHYSHLTH": 1, "MENTHLTH": 3, "CHCCOPD1": 0
    }
    print(json.dumps(predict_full("asthma", sample_asthma), indent=2))

    print("\nTesting Kidney Disease Model:")
    sample_kidney = {
        "_AGE80": 55, "SEX": 1, "GENHLTH": 2, "PHYSHLTH": 2, "MENTHLTH": 0,
        "EXERANY2": 1, "SLEPTIM1": 7, "_BMI5": 26.5, "SMOKE100": 0, "_SMOKER3": 4,
        "DRNKANY5": 0, "DIABETE3": 3, "INCOME2": 6, "EDUCA": 5, "MARITAL": 1
    }
    print(json.dumps(predict_full("kidney_disease", sample_kidney), indent=2))

    print("\nTesting Stroke Model:")
    sample_stroke = {
        "_AGE80": 60, "SEX": 1, "GENHLTH": 4, "_BMI5": 30.0, "SMOKE100": 1, "_SMOKER3": 1,
        "EXERANY2": 0, "SLEPTIM1": 5, "DRNKANY5": 1, "DIABETE3": 1,
        "CVDCRHD4": 1, "AFib": 1, "DiffWalking": 1, "INCOME2": 3, "EDUCA": 3
    }
    print(json.dumps(predict_full("stroke", sample_stroke), indent=2))
