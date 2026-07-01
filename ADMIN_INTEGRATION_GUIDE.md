"""
SYSTEM ADMINISTRATOR: CODE INTEGRATION GUIDE
══════════════════════════════════════════════════════════════════════════════

After receiving a disease model, follow these exact steps to integrate it.
This document shows ALL code changes required with complete examples.

"""


# ═══════════════════════════════════════════════════════════════════════════════
# PART 1: PREPARE MODEL FILES
# ═══════════════════════════════════════════════════════════════════════════════

"""
STEP 1.1: Receive model package

Collaborator sends: [disease_id].zip

Extract to:
  models/[disease_id]/
  ├── [disease_id]_model.pkl
  ├── [disease_id]_scaler.pkl
  ├── feature_names.pkl
  └── model_meta.pkl

Example for Hypertension:
  models/hypertension/
  ├── hypertension_model.pkl
  ├── hypertension_scaler.pkl
  ├── feature_names.pkl
  └── model_meta.pkl

STEP 1.2: Run validation script

  import joblib
  import numpy as np
  
  DISEASE = "hypertension"
  MODEL_DIR = f"models/{DISEASE}"
  
  model = joblib.load(f'{MODEL_DIR}/{DISEASE}_model.pkl')
  scaler = joblib.load(f'{MODEL_DIR}/{DISEASE}_scaler.pkl')
  features = joblib.load(f'{MODEL_DIR}/feature_names.pkl')
  meta = joblib.load(f'{MODEL_DIR}/model_meta.pkl')
  
  # Test
  sample = np.random.randn(1, len(features))
  scaled = scaler.transform(sample)
  pred = model.predict(scaled)
  prob = model.predict_proba(scaled)
  
  print(f"✓ All files loaded successfully")
  print(f"  Accuracy: {meta['accuracy']}")
  print(f"  Features: {len(features)}")
"""


# ═══════════════════════════════════════════════════════════════════════════════
# PART 2: UPDATE app.py
# ═══════════════════════════════════════════════════════════════════════════════

"""
STEP 2.1: Add disease to DISEASE_CONFIGS dictionary

Location: Near top of app.py (after Flask imports)

CURRENT CODE (Diabetes only):

  DISEASE_CONFIGS = {
      "diabetes": {
          "display_name": "Diabetes",
          "model_dir": "models/diabetes",
          "features": ["_BMI5", "_AGEG5YR", "SEX", "GENHLTH", "SMOKE100", "EXERANY2", "CVDINFR4", "CVDCRHD4", "CVDSTRK3", "CHCKIDNY"],
          "field_mappings": {
              "age_group": {
                  "18-24": 1, "25-29": 2, "30-34": 3, "35-39": 4, "40-44": 5, "45-49": 6,
                  "50-54": 7, "55-59": 8, "60-64": 9, "65-69": 10, "70-74": 11, "75-79": 12, "80+": 13
              },
              "general_health": {
                  "Excellent": 1, "Very Good": 2, "Good": 3, "Fair": 4, "Poor": 5
              },
              "sex": {"Male": 1, "Female": 0},
          },
          "risk_thresholds": {"low": 30, "high": 70},
      },
  }

UPDATED CODE (After adding Hypertension):

  DISEASE_CONFIGS = {
      "diabetes": {
          "display_name": "Diabetes",
          "model_dir": "models/diabetes",
          "features": ["_BMI5", "_AGEG5YR", "SEX", "GENHLTH", "SMOKE100", "EXERANY2", "CVDINFR4", "CVDCRHD4", "CVDSTRK3", "CHCKIDNY"],
          "field_mappings": {
              "age_group": {
                  "18-24": 1, "25-29": 2, "30-34": 3, "35-39": 4, "40-44": 5, "45-49": 6,
                  "50-54": 7, "55-59": 8, "60-64": 9, "65-69": 10, "70-74": 11, "75-79": 12, "80+": 13
              },
              "general_health": {
                  "Excellent": 1, "Very Good": 2, "Good": 3, "Fair": 4, "Poor": 5
              },
              "sex": {"Male": 1, "Female": 0},
          },
          "risk_thresholds": {"low": 30, "high": 70},
      },
      
      "hypertension": {
          "display_name": "Hypertension",
          "model_dir": "models/hypertension",
          "features": ["_BMI5", "_AGEG5YR", "SEX", "GENHLTH", "SMOKE100", "EXERANY2", "CVDINFR4", "CVDCRHD4", "CVDSTRK3", "CHCKIDNY"],
          "field_mappings": {
              "age_group": {
                  "18-24": 1, "25-29": 2, "30-34": 3, "35-39": 4, "40-44": 5, "45-49": 6,
                  "50-54": 7, "55-59": 8, "60-64": 9, "65-69": 10, "70-74": 11, "75-79": 12, "80+": 13
              },
              "general_health": {
                  "Excellent": 1, "Very Good": 2, "Good": 3, "Fair": 4, "Poor": 5
              },
              "sex": {"Male": 1, "Female": 0},
          },
          "risk_thresholds": {"low": 30, "high": 70},
      },
  }

NOTE: Copy field_mappings from the README provided by collaborator.
      They may be different for each disease!

STEP 2.2: Create generic prediction route (if not exists)

Add this route to app.py:

  @app.route("/predict/<disease>", methods=["POST"])
  def predict_disease(disease):
      '''Generic prediction route for any disease'''
      
      if disease not in DISEASE_CONFIGS:
          return jsonify({
              "status": "error",
              "message": f"Unknown disease: {disease}"
          }), 404
      
      config = DISEASE_CONFIGS[disease]
      
      try:
          data = request.get_json()
          
          # Validate required fields
          required_fields = config["field_mappings"].keys()
          for field in required_fields:
              if field not in data:
                  return jsonify({
                      "status": "error",
                      "message": f"Missing required field: {field}"
                  }), 400
          
          # Map user inputs to model features
          model_input = {}
          
          # BMI (continuous)
          try:
              model_input["_BMI5"] = float(data["bmi"])
          except ValueError:
              return jsonify({
                  "status": "error",
                  "message": "BMI must be a valid number"
              }), 400
          
          # Age group (categorical)
          age_mapping = config["field_mappings"]["age_group"]
          if data["age_group"] not in age_mapping:
              return jsonify({
                  "status": "error",
                  "message": f"Invalid age group: {data['age_group']}"
              }), 400
          model_input["_AGEG5YR"] = age_mapping[data["age_group"]]
          
          # Sex (binary categorical)
          sex_mapping = config["field_mappings"]["sex"]
          if data["sex"] not in sex_mapping:
              return jsonify({
                  "status": "error",
                  "message": "Invalid sex"
              }), 400
          model_input["SEX"] = sex_mapping[data["sex"]]
          
          # General health (ordinal)
          health_mapping = config["field_mappings"]["general_health"]
          health_key = data["general_health"].lower()
          if health_key not in health_mapping:
              return jsonify({
                  "status": "error",
                  "message": f"Invalid health rating"
              }), 400
          model_input["GENHLTH"] = health_mapping[health_key]
          
          # Yes/No fields
          yes_no_fields = {
              "smoking": "SMOKE100",
              "exercise": "EXERANY2",
              "heart_attack": "CVDINFR4",
              "heart_disease": "CVDCRHD4",
              "stroke": "CVDSTRK3",
              "kidney_disease": "CHCKIDNY"
          }
          
          for user_field, brfss_field in yes_no_fields.items():
              val = data[user_field].lower()
              if val == "yes":
                  model_input[brfss_field] = 1
              elif val == "no":
                  model_input[brfss_field] = 0
              else:
                  return jsonify({
                      "status": "error",
                      "message": f"{user_field} must be 'Yes' or 'No'"
                  }), 400
          
          # Load model dynamically
          from predict import predict_full
          result = predict_full(model_input)
          
          return jsonify(result), 200
      
      except Exception as e:
          return jsonify({
              "status": "error",
              "message": f"Prediction failed: {str(e)}"
          }), 500

STEP 2.3: Update predict.py to support multiple diseases (OPTIONAL)

If you want a centralized predict function:

  from predict import predict_by_disease
  
  result = predict_by_disease("hypertension", model_input)

Or keep separate predict functions:

  from predict_diabetes import predict_full as predict_diabetes
  from predict_hypertension import predict_full as predict_hypertension
"""


# ═══════════════════════════════════════════════════════════════════════════════
# PART 3: UPDATE index.html
# ═══════════════════════════════════════════════════════════════════════════════

"""
STEP 3.1: Activate disease button in navigation

Location: <div class="disease-nav"> section in index.html

CURRENT CODE:

  <div class="disease-nav">
      <button class="disease-btn active" data-disease="diabetes">Diabetes</button>
      <button class="disease-btn inactive" title="Coming Soon">Hypertension (Coming Soon)</button>
      <button class="disease-btn inactive" title="Coming Soon">Heart Disease (Coming Soon)</button>
      <button class="disease-btn inactive" title="Coming Soon">Stroke (Coming Soon)</button>
  </div>

UPDATED CODE (Hypertension active):

  <div class="disease-nav">
      <button class="disease-btn active" data-disease="diabetes">Diabetes</button>
      <button class="disease-btn active" data-disease="hypertension">Hypertension</button>
      <button class="disease-btn inactive" title="Coming Soon">Heart Disease (Coming Soon)</button>
      <button class="disease-btn inactive" title="Coming Soon">Stroke (Coming Soon)</button>
  </div>

STEP 3.2: Make form dynamic or create separate forms

OPTION A: Single form (update dynamically)

Add this JavaScript to update form fields when disease is switched:

  const DISEASE_FORMS = {
      "diabetes": [
          { id: "bmi", label: "Body Mass Index (BMI)" },
          { id: "ageGroup", label: "Age Group" },
          // ... all fields
      ],
      "hypertension": [
          { id: "bmi", label: "Body Mass Index (BMI)" },
          { id: "ageGroup", label: "Age Group" },
          // ... all fields
      ],
  };
  
  document.querySelectorAll(".disease-btn.active").forEach(btn => {
      btn.addEventListener("click", function() {
          const disease = this.getAttribute("data-disease");
          updateFormFields(disease);
      });
  });
  
  function updateFormFields(disease) {
      const formFields = DISEASE_FORMS[disease];
      // Update form based on disease
  }

OPTION B: Separate forms (simpler approach)

Create separate form HTML for each disease:

  <div id="diabetes-form" class="disease-form active">
      <!-- Diabetes form fields -->
  </div>
  
  <div id="hypertension-form" class="disease-form" style="display: none;">
      <!-- Hypertension form fields -->
  </div>
  
  Then toggle with JavaScript:
  
  document.querySelectorAll(".disease-btn").forEach(btn => {
      btn.addEventListener("click", function() {
          const disease = this.getAttribute("data-disease");
          // Hide all forms
          document.querySelectorAll(".disease-form").forEach(f => 
              f.style.display = "none"
          );
          // Show selected form
          document.getElementById(disease + "-form").style.display = "block";
      });
  });

STEP 3.3: Update API endpoint in JavaScript

Location: predictionForm submission handler

CURRENT CODE:

  const response = await fetch("/predict/diabetes", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(formData),
  });

UPDATED CODE (Dynamic):

  const disease = document.querySelector(".disease-btn.active").getAttribute("data-disease");
  
  const response = await fetch(`/predict/${disease}`, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(formData),
  });
"""


# ═══════════════════════════════════════════════════════════════════════════════
# PART 4: TESTING THE INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════════

"""
STEP 4.1: Test Flask API endpoint

Use curl or Postman:

  curl -X POST http://localhost:5000/predict/hypertension \
    -H "Content-Type: application/json" \
    -d '{
      "bmi": 27.5,
      "age_group": "50-54",
      "sex": "Male",
      "general_health": "Good",
      "smoking": "Yes",
      "exercise": "No",
      "heart_attack": "No",
      "heart_disease": "No",
      "stroke": "No",
      "kidney_disease": "No"
    }'

Expected response:

  {
    "prediction": "High Risk",
    "risk_percentage": 62.35,
    "risk_level": "Medium Risk",
    "accuracy": 0.84,
    "model_name": "Logistic Regression",
    "dataset": "BRFSS 2014",
    "recommendations": [...],
    "risk_factors": [],
    "status": "success"
  }

STEP 4.2: Test web UI

1. Start Flask server: python app.py
2. Open browser: http://localhost:5000
3. Select disease from navigation
4. Fill out form
5. Click "Get Prediction"
6. Verify results display correctly

STEP 4.3: Verify all UI elements

  □ Disease button highlights when clicked
  □ Form fields populate correctly
  □ Loading spinner appears during prediction
  □ Results panel displays
  □ Risk percentage shows with correct color
  □ Risk bar fills correctly
  □ Recommendations display
  □ Error messages show on failure
"""


# ═══════════════════════════════════════════════════════════════════════════════
# PART 5: DEPLOYMENT CHECKLIST
# ═══════════════════════════════════════════════════════════════════════════════

"""
Before deploying new disease:

CODE CHANGES:
  □ app.py: Added disease to DISEASE_CONFIGS
  □ app.py: Routes handle new disease
  □ index.html: Disease button is active (not greyed out)
  □ index.html: Form fields match new disease
  □ index.html: API endpoint updated to use dynamic disease

MODEL FILES:
  □ [disease]_model.pkl in models/[disease]/
  □ [disease]_scaler.pkl in models/[disease]/
  □ feature_names.pkl in models/[disease]/
  □ model_meta.pkl in models/[disease]/
  □ All files loadable without errors

TESTING:
  □ Validation script runs successfully
  □ API endpoint responds to POST request
  □ Web form submits successfully
  □ Results display with correct risk level
  □ Recommendations appear
  □ Error handling works (try invalid input)
  □ All features map correctly

DOCUMENTATION:
  □ README saved with model package
  □ Feature mappings documented
  □ Risk thresholds defined
  □ Recommendations provided

VERSION CONTROL:
  □ Changes committed with descriptive message
  □ Example: "Add Hypertension model integration"
  □ Include which features were added
  □ Reference model accuracy
"""


# ═══════════════════════════════════════════════════════════════════════════════
# QUICK REFERENCE: FILES TO MODIFY
# ═══════════════════════════════════════════════════════════════════════════════

"""
When integrating a new disease, modify these files:

FILE 1: app.py
   Location: DISEASE_CONFIGS dictionary (near top)
   Change: Add new disease entry with field mappings
   
FILE 2: index.html
   Location: <div class="disease-nav">
   Change: Activate disease button (remove "inactive" class)
   
FILE 3: index.html
   Location: Form section (predictionForm)
   Change: Update form fields to match disease features
   
FILE 4: index.html
   Location: JavaScript fetch() call
   Change: Make endpoint dynamic (use disease variable)

TOTAL FILES: 1 Python file + 1 HTML file
ESTIMATED TIME: 30 minutes per disease
"""
