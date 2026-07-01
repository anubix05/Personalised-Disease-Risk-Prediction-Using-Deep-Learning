"""
Flask backend for Personalised Disease Risk Prediction Using Deep Learning
"""

from flask import Flask, render_template, request, jsonify
from predict import predict_full
import json

app = Flask(__name__)

# ─── Disease Configurations and Mappings ───
DISEASE_CONFIGS = {
    "diabetes": {
        "display_name": "Diabetes",
        "model_dir": "models/diabetes",
        "field_mappings": {
            "age_group": {
                "18-24": 1, "25-29": 2, "30-34": 3, "35-39": 4, "40-44": 5, "45-49": 6,
                "50-54": 7, "55-59": 8, "60-64": 9, "65-69": 10, "70-74": 11, "75-79": 12, "80+": 13
            },
            "general_health": {
                "Excellent": 1, "Very Good": 2, "Good": 3, "Fair": 4, "Poor": 5
            },
            "sex": {"Male": 1, "Female": 0},
        }
    },
    "heart_disease": {
        "display_name": "Heart Disease",
        "model_dir": "models/heartDisease",
        "field_mappings": {
            "age_group": {
                "18-24": 1, "25-29": 2, "30-34": 3, "35-39": 4, "40-44": 5, "45-49": 6,
                "50-54": 7, "55-59": 8, "60-64": 9, "65-69": 10, "70-74": 11, "75-79": 12, "80+": 13
            },
            "general_health": {
                "Excellent": 1, "Very Good": 2, "Good": 3, "Fair": 4, "Poor": 5
            },
            "sex": {"Male": 1, "Female": 0},
            "smoker_status": {
                "Everyday Smoker": 1, "Someday Smoker": 2, "Former Smoker": 3, "Never Smoked": 4
            },
            "income": {
                "Under $10k": 1, "$10k to $15k": 2, "$15k to $20k": 3, "$20k to $25k": 4,
                "$25k to $35k": 5, "$35k to $50k": 6, "$50k to $75k": 7, "$75k or more": 8
            },
            "education": {
                "Never attended school": 1, "Elementary school": 2, "Some high school": 3,
                "High school graduate": 4, "Some college": 5, "College graduate": 6
            },
            "marital_status": {
                "Married": 1, "Divorced": 2, "Widowed": 3, "Separated": 4, "Never married": 5,
                "A member of an unmarried couple": 6
            }
        }
    },
    "stroke": {
        "display_name": "Stroke",
        "model_dir": "models/stroke",
        "field_mappings": {
            "age_group": {
                "18-24": 1, "25-29": 2, "30-34": 3, "35-39": 4, "40-44": 5, "45-49": 6,
                "50-54": 7, "55-59": 8, "60-64": 9, "65-69": 10, "70-74": 11, "75-79": 12, "80+": 13
            },
            "general_health": {
                "Excellent": 1, "Very Good": 2, "Good": 3, "Fair": 4, "Poor": 5
            },
            "sex": {"Male": 1, "Female": 0},
            "smoker_status": {
                "Everyday Smoker": 1, "Someday Smoker": 2, "Former Smoker": 3, "Never Smoked": 4
            },
            "income": {
                "Under $10k": 1, "$10k to $15k": 2, "$15k to $20k": 3, "$20k to $25k": 4,
                "$25k to $35k": 5, "$35k to $50k": 6, "$50k to $75k": 7, "$75k or more": 8
            },
            "education": {
                "Never attended school": 1, "Elementary school": 2, "Some high school": 3,
                "High school graduate": 4, "Some college": 5, "College graduate": 6
            }
        }
    },
    "asthma": {
        "display_name": "Asthma",
        "model_dir": "models/asthma",
        "field_mappings": {
            "general_health": {
                "Excellent": 1, "Very Good": 2, "Good": 3, "Fair": 4, "Poor": 5
            },
            "sex": {"Male": 1, "Female": 0}
        }
    },
    "kidney_disease": {
        "display_name": "Kidney Disease",
        "model_dir": "models/kidneyDisease",
        "field_mappings": {
            "age_group": {
                "18-24": 1, "25-29": 2, "30-34": 3, "35-39": 4, "40-44": 5, "45-49": 6,
                "50-54": 7, "55-59": 8, "60-64": 9, "65-69": 10, "70-74": 11, "75-79": 12, "80+": 13
            },
            "general_health": {
                "Excellent": 1, "Very Good": 2, "Good": 3, "Fair": 4, "Poor": 5
            },
            "sex": {"Male": 1, "Female": 0},
            "smoker_status": {
                "Everyday Smoker": 1, "Someday Smoker": 2, "Former Smoker": 3, "Never Smoked": 4
            },
            "income": {
                "Under $10k": 1, "$10k to $15k": 2, "$15k to $20k": 3, "$20k to $25k": 4,
                "$25k to $35k": 5, "$35k to $50k": 6, "$50k to $75k": 7, "$75k or more": 8
            },
            "education": {
                "Never attended school": 1, "Elementary school": 2, "Some high school": 3,
                "High school graduate": 4, "Some college": 5, "College graduate": 6
            },
            "marital_status": {
                "Married": 1, "Divorced": 2, "Widowed": 3, "Separated": 4, "Never married": 5,
                "A member of an unmarried couple": 6
            }
        }
    },
    "arthritis": {
        "display_name": "Arthritis",
        "model_dir": "models/arthritis",
        "field_mappings": {
            "age_group": {
                "18-24": 1, "25-29": 2, "30-34": 3, "35-39": 4, "40-44": 5, "45-49": 6,
                "50-54": 7, "55-59": 8, "60-64": 9, "65-69": 10, "70-74": 11, "75-79": 12, "80+": 13
            },
            "general_health": {
                "Excellent": 1, "Very Good": 2, "Good": 3, "Fair": 4, "Poor": 5
            },
            "sex": {"Male": 1, "Female": 0},
            "smoker_status": {
                "Everyday Smoker": 1, "Someday Smoker": 2, "Former Smoker": 3, "Never Smoked": 4
            },
            "income": {
                "Under $10k": 1, "$10k to $15k": 2, "$15k to $20k": 3, "$20k to $25k": 4,
                "$25k to $35k": 5, "$35k to $50k": 6, "$50k to $75k": 7, "$75k or more": 8
            },
            "education": {
                "Never attended school": 1, "Elementary school": 2, "Some high school": 3,
                "High school graduate": 4, "Some college": 5, "College graduate": 6
            }
        }
    }
}

@app.route("/", methods=["GET"])
def home():
    """Serve the main dashboard."""
    return render_template("index.html")

@app.route("/predict/<disease>", methods=["POST"])
def predict_disease(disease):
    """Dynamic prediction route for any disease."""
    if disease not in DISEASE_CONFIGS:
        return jsonify({
            "status": "error",
            "message": f"Unknown disease: {disease}"
        }), 404
        
    config = DISEASE_CONFIGS[disease]
    
    try:
        data = request.get_json()
        
        # Build raw model input dictionary
        model_input = {}
        
        # 1. BMI: passed as float
        if "bmi" in data and data["bmi"]:
            try:
                model_input["_BMI5"] = float(data["bmi"])
            except ValueError:
                return jsonify({"status": "error", "message": "BMI must be a valid number"}), 400
                
        # 2. Age Group: mapped to _AGEG5YR
        if "age_group" in data and "age_group" in config["field_mappings"] and data["age_group"]:
            age_val = data["age_group"]
            age_mapping = config["field_mappings"]["age_group"]
            if age_val not in age_mapping:
                return jsonify({"status": "error", "message": f"Invalid age group: {age_val}"}), 400
            model_input["_AGEG5YR"] = age_mapping[age_val]
            
        # 3. Raw Age: _AGE80
        if "age" in data and data["age"]:
            try:
                model_input["_AGE80"] = float(data["age"])
            except ValueError:
                return jsonify({"status": "error", "message": "Age must be a valid number"}), 400
        elif "age_group" in data and disease in ["heart_disease", "stroke", "kidney_disease", "arthritis"] and data["age_group"]:
            # Fallback: estimate raw age using age group midpoint
            age_midpoints = {
                "18-24": 21, "25-29": 27, "30-34": 32, "35-39": 37, "40-44": 42, "45-49": 47,
                "50-54": 52, "55-59": 57, "60-64": 62, "65-69": 67, "70-74": 72, "75-79": 77, "80+": 82
            }
            model_input["_AGE80"] = float(age_midpoints.get(data["age_group"], 45))

        # 4. Sex: mapped to SEX (Male=1, Female=0)
        if "sex" in data and data["sex"]:
            sex_val = data["sex"]
            sex_mapping = config["field_mappings"]["sex"]
            if sex_val not in sex_mapping:
                return jsonify({"status": "error", "message": "Sex must be 'Male' or 'Female'"}), 400
            model_input["SEX"] = sex_mapping[sex_val]

        # 5. General Health: mapped to GENHLTH
        if "general_health" in data and data["general_health"]:
            health_val = data["general_health"]
            health_mapping = config["field_mappings"]["general_health"]
            if health_val not in health_mapping:
                return jsonify({"status": "error", "message": f"Invalid general health: {health_val}"}), 400
            model_input["GENHLTH"] = health_mapping[health_val]

        # 6. Yes/No Fields: Yes=1, No=0
        yes_no_mapping = {
            "smoking": "SMOKE100",
            "exercise": "EXERANY2",
            "heart_attack": "CVDINFR4",
            "heart_disease": "CVDCRHD4",
            "stroke": "CVDSTRK3",
            "kidney_disease": "CHCKIDNY",
            "alcohol_consumption": "DRNKANY5",
            "diabetes": "DIABETE3",
            "afib": "AFib",
            "diff_walking": "DiffWalking",
            "copd": "CHCCOPD1"
        }
        
        for user_field, brfss_field in yes_no_mapping.items():
            if user_field in data and data[user_field]:
                val = data[user_field].lower()
                if val == "yes":
                    model_input[brfss_field] = 1
                elif val == "no":
                    model_input[brfss_field] = 0
                else:
                    return jsonify({"status": "error", "message": f"{user_field} must be 'Yes' or 'No'"}), 400

        # 7. Additional Vitals / Days
        if "physical_health_days" in data and data["physical_health_days"] != "":
            try:
                model_input["PHYSHLTH"] = float(data["physical_health_days"])
            except ValueError:
                return jsonify({"status": "error", "message": "Physical health days must be a number"}), 400
                
        if "mental_health_days" in data and data["mental_health_days"] != "":
            try:
                model_input["MENTHLTH"] = float(data["mental_health_days"])
            except ValueError:
                return jsonify({"status": "error", "message": "Mental health days must be a number"}), 400
                
        if "sleep_duration" in data and data["sleep_duration"] != "":
            try:
                model_input["SLEPTIM1"] = float(data["sleep_duration"])
            except ValueError:
                return jsonify({"status": "error", "message": "Sleep duration must be a number"}), 400

        # 8. Demographic Category Fields
        if "smoker_status" in data and "smoker_status" in config["field_mappings"] and data["smoker_status"]:
            model_input["_SMOKER3"] = config["field_mappings"]["smoker_status"].get(data["smoker_status"], 4)
            
        if "income" in data and "income" in config["field_mappings"] and data["income"]:
            model_input["INCOME2"] = config["field_mappings"]["income"].get(data["income"], 5)
            
        if "education" in data and "education" in config["field_mappings"] and data["education"]:
            model_input["EDUCA"] = config["field_mappings"]["education"].get(data["education"], 4)
            
        if "marital_status" in data and "marital_status" in config["field_mappings"] and data["marital_status"]:
            model_input["MARITAL"] = config["field_mappings"]["marital_status"].get(data["marital_status"], 1)

        # Execute prediction
        result = predict_full(disease, model_input)
        return jsonify(result), 200
        
    except FileNotFoundError as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 404
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Prediction failed: {str(e)}"
        }), 500

@app.route("/predict/diabetes", methods=["POST"])
def predict_diabetes_legacy():
    """Legacy backward compatibility route for diabetes."""
    return predict_disease("diabetes")

if __name__ == "__main__":
    import sys
    print("Python Executable:", sys.executable)
    print("Python Path:", sys.path)
    app.run(debug=True, use_reloader=False, host="127.0.0.1", port=5000)
