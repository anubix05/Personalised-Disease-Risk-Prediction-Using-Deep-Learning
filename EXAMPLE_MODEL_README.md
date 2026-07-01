# Hypertension Model Integration Package

**This is a TEMPLATE example showing exactly what format to use for model delivery.**

---

## Model Information

| Attribute | Value |
|-----------|-------|
| **Disease** | Hypertension Risk Prediction |
| **Disease ID** | `hypertension` |
| **Model Type** | Logistic Regression |
| **Accuracy** | 84% |
| **Dataset** | BRFSS 2014 |
| **Training Date** | 2026-06-20 |
| **Feature Count** | 10 |
| **Status** | Ready for Integration |

---

## Input Features & Mappings

### CRITICAL: Features must be provided in this exact order

```
["_BMI5", "_AGEG5YR", "SEX", "GENHLTH", "SMOKE100", "EXERANY2", "CVDINFR4", "CVDCRHD4", "CVDSTRK3", "CHCKIDNY"]
```

---

## Feature Specifications

### Feature 1: Body Mass Index

| Property | Value |
|----------|-------|
| **Field ID** | `bmi` |
| **Display Name** | Body Mass Index (BMI) |
| **Data Type** | Continuous (float) |
| **BRFSS Code** | `_BMI5` |
| **Input Range** | 10 to 60 |
| **Step** | 0.1 |
| **Example Input** | `27.5` |
| **Model Feature** | Raw float value |

---

### Feature 2: Age Group

| Property | Value |
|----------|-------|
| **Field ID** | `age_group` |
| **Display Name** | Age Group |
| **Data Type** | Categorical (ordinal) |
| **BRFSS Code** | `_AGEG5YR` |
| **Input Type** | HTML Select Dropdown |

**Options & Mappings:**

```
18-24    → 1
25-29    → 2
30-34    → 3
35-39    → 4
40-44    → 5
45-49    → 6
50-54    → 7
55-59    → 8
60-64    → 9
65-69    → 10
70-74    → 11
75-79    → 12
80+      → 13
```

**Example:** User selects "50-54" in dropdown → API sends `"_AGEG5YR": 7`

---

### Feature 3: Sex

| Property | Value |
|----------|-------|
| **Field ID** | `sex` |
| **Display Name** | Sex |
| **Data Type** | Binary categorical |
| **BRFSS Code** | `SEX` |
| **Input Type** | HTML Select Dropdown |

**Options & Mappings:**

```
Male    → 1
Female  → 0
```

**Example:** User selects "Male" → API sends `"SEX": 1`

---

### Feature 4: General Health

| Property | Value |
|----------|-------|
| **Field ID** | `general_health` |
| **Display Name** | General Health |
| **Data Type** | Ordinal scale (1-5) |
| **BRFSS Code** | `GENHLTH` |
| **Input Type** | HTML Select Dropdown |

**Options & Mappings:**

```
Excellent   → 1
Very Good   → 2
Good        → 3
Fair        → 4
Poor        → 5
```

**Example:** User selects "Good" → API sends `"GENHLTH": 3`

---

### Feature 5: Smoking History

| Property | Value |
|----------|-------|
| **Field ID** | `smoking` |
| **Display Name** | Smoking History |
| **Data Type** | Binary Yes/No |
| **BRFSS Code** | `SMOKE100` |
| **Question** | Have you smoked 100+ cigarettes in your lifetime? |
| **Input Type** | HTML Select Dropdown |

**Options & Mappings:**

```
Yes (smoked 100+)    → 1
No (have not)        → 0
```

**Example:** User selects "Yes" → API sends `"SMOKE100": 1`

---

### Feature 6: Regular Exercise

| Property | Value |
|----------|-------|
| **Field ID** | `exercise` |
| **Display Name** | Regular Exercise |
| **Data Type** | Binary Yes/No |
| **BRFSS Code** | `EXERANY2` |
| **Question** | Do you exercise regularly? |
| **Input Type** | HTML Select Dropdown |

**Options & Mappings:**

```
Yes (exercise regularly)    → 1
No (do not exercise)        → 0
```

---

### Feature 7: Heart Attack History

| Property | Value |
|----------|-------|
| **Field ID** | `heart_attack` |
| **Display Name** | Heart Attack History |
| **Data Type** | Binary Yes/No |
| **BRFSS Code** | `CVDINFR4` |
| **Question** | Have you ever been told you had a heart attack? |
| **Input Type** | HTML Select Dropdown |

**Options & Mappings:**

```
Yes (had heart attack)    → 1
No (have not)             → 0
```

---

### Feature 8: Coronary Heart Disease

| Property | Value |
|----------|-------|
| **Field ID** | `heart_disease` |
| **Display Name** | Coronary Heart Disease |
| **Data Type** | Binary Yes/No |
| **BRFSS Code** | `CVDCRHD4` |
| **Question** | Have you been told you have coronary heart disease? |
| **Input Type** | HTML Select Dropdown |

**Options & Mappings:**

```
Yes (been told I have)    → 1
No (have not been told)   → 0
```

---

### Feature 9: Stroke History

| Property | Value |
|----------|-------|
| **Field ID** | `stroke` |
| **Display Name** | Stroke History |
| **Data Type** | Binary Yes/No |
| **BRFSS Code** | `CVDSTRK3` |
| **Question** | Have you ever been told you had a stroke? |
| **Input Type** | HTML Select Dropdown |

**Options & Mappings:**

```
Yes (had stroke)    → 1
No (have not)       → 0
```

---

### Feature 10: Kidney Disease

| Property | Value |
|----------|-------|
| **Field ID** | `kidney_disease` |
| **Display Name** | Kidney Disease |
| **Data Type** | Binary Yes/No |
| **BRFSS Code** | `CHCKIDNY` |
| **Question** | Have you been told you have kidney disease? |
| **Input Type** | HTML Select Dropdown |

**Options & Mappings:**

```
Yes (have kidney disease)    → 1
No (do not have)             → 0
```

---

## Risk Classification

### Risk Thresholds

```
LOW RISK:       0% - 30%     (Green)
MEDIUM RISK:   31% - 70%     (Yellow)
HIGH RISK:     71% - 100%    (Red)
```

---

## Recommendations

### Low Risk (0-30%)

Users at low risk for hypertension should:

- ✓ Maintain current healthy lifestyle habits
- ✓ Check blood pressure annually
- ✓ Keep sodium intake below 2,300 mg/day
- ✓ Exercise 150 minutes per week at moderate intensity

### Medium Risk (31-70%)

Users at medium risk should:

- ✓ Schedule appointment with healthcare provider for blood pressure assessment
- ✓ Begin home blood pressure monitoring (record daily for 1 week)
- ✓ Reduce sodium intake to <2,300 mg/day
- ✓ Increase physical activity to 30 minutes daily
- ✓ Monitor blood pressure every 2-3 months

### High Risk (71-100%)

Users at high risk require immediate attention:

- ⚠️ **Schedule medical appointment immediately**
- ⚠️ Begin daily home blood pressure monitoring
- ⚠️ Adopt strict low-sodium diet (<2,300 mg/day)
- ⚠️ Start gentle daily exercise under medical supervision
- ⚠️ Discuss antihypertensive medication options with physician

---

## Model Training Details

| Metric | Value |
|--------|-------|
| **Training Records** | 50,000 |
| **Test Records** | 10,000 |
| **Positive Cases** | 35% |
| **Negative Cases** | 65% |
| **Features Selected** | All 10 BRFSS fields |
| **Feature Scaling** | StandardScaler |
| **Cross-Validation** | 5-fold |
| **Precision** | 0.82 |
| **Recall** | 0.81 |
| **F1-Score** | 0.815 |

---

## File Manifest

```
hypertension.zip
├── README.md (THIS FILE)
└── models/hypertension/
    ├── hypertension_model.pkl     (Trained Logistic Regression model)
    ├── hypertension_scaler.pkl    (StandardScaler fitted to training data)
    ├── feature_names.pkl          (List of features in order)
    └── model_meta.pkl             (Metadata: accuracy, model name, dataset)
```

---

## Validation

✓ All pickle files validated  
✓ Model can predict on test data  
✓ predict_proba() returns valid probabilities  
✓ Feature order verified  
✓ Scaler output shape matches model input  

---

## Integration Instructions

1. **Extract files** to `models/hypertension/` directory
2. **Update `app.py`**: Add hypertension to `DISEASE_CONFIGS` dict
3. **Update `index.html`**: Add disease button and form fields
4. **Run validation** to confirm model loads
5. **Test prediction** with sample input

---

## Contact & Support

- **Model Trainer**: [Your Name]
- **Training Date**: 2026-06-20
- **Questions**: [Your Email]
- **Version**: 1.0

---

**Ready for production integration** ✓
