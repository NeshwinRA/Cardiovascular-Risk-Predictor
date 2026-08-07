# Cardiovascular Risk Prediction System — Feature Overview

> **Stack:** Python + Flask | HTML/CSS/JS | Logistic Regression + Random Forest | Simple Rule-based XAI | No Database  
> **Model Storage:** `/models` folder | **Frontend:** Multi-page

---

## Phase Overview

| Phase | Title | Status |
|-------|-------|--------|
| P1 | Project Setup | ✅ Completed |
| P2 | Data Preprocessing (train.py) | ✅ Completed |
| P3 | Model Training & Evaluation (train.py) | ✅ Completed |
| P4 | Flask Backend (app.py) | ✅ Completed |
| P5 | Frontend — Input Page (index.html) | ✅ Completed |
| P6 | Frontend — Result Page (result.html) | ✅ Completed |
| P7 | Frontend — Recommendations Page (recommendations.html) | ✅ Completed |
| P8 | Testing & Final Polish | ✅ Completed |

---

## Detailed Feature Status

### P1 — Project Setup
| Step | Description | Status |
|------|-------------|--------|
| P1-S1 | Create project folder structure | ✅ Completed |
| P1-S2 | Create requirements.txt | ✅ Completed |
| P1-S3 | Set up Python virtual environment | ✅ Completed |
| P1-S4 | Verify CSV file is accessible | ✅ Completed |

### P2 — Data Preprocessing
| Step | Description | Status |
|------|-------------|--------|
| P2-S1 | Load framingham.csv | ✅ Completed |
| P2-S2 | Handle missing values | ✅ Completed |
| P2-S3 | Define features (X) and target (y) | ✅ Completed |
| P2-S4 | Handle class imbalance (SMOTE) | ✅ Completed |
| P2-S5 | Scale features using StandardScaler | ✅ Completed |
| P2-S6 | Split data 80/20 | ✅ Completed |

### P3 — Model Training & Evaluation
| Step | Description | Status |
|------|-------------|--------|
| P3-S1 | Train Logistic Regression model | ✅ Completed |
| P3-S2 | Train Random Forest model | ✅ Completed |
| P3-S3 | Evaluate both models on test set | ✅ Completed |
| P3-S4 | Select best model | ✅ Completed |
| P3-S5 | Save scaler and model using joblib | ✅ Completed |

### P4 — Flask Backend
| Step | Description | Status |
|------|-------------|--------|
| P4-S1 | Create app.py and initialize Flask | ✅ Completed |
| P4-S2 | Route: GET / (home/input page) | ✅ Completed |
| P4-S3 | Route: POST /predict | ✅ Completed |
| P4-S4 | Route: GET /recommendations | ✅ Completed |
| P4-S5 | Input validation logic | ✅ Completed |
| P4-S6 | Rule-based XAI logic | ✅ Completed |
| P4-S7 | Disease association logic | ✅ Completed |

### P5 — Frontend: Input Page
| Step | Description | Status |
|------|-------------|--------|
| P5-S1 | Create templates/index.html | ✅ Completed |
| P5-S2 | Create static/css/style.css | ✅ Completed |
| P5-S3 | Add client-side validation (JS) | ✅ Completed |
| P5-S4 | Show loading spinner on predict click | ✅ Completed |

### P6 — Frontend: Result Page
| Step | Description | Status |
|------|-------------|--------|
| P6-S1 | Create templates/result.html | ✅ Completed |
| P6-S2 | Dynamic color coding based on risk level | ✅ Completed |
| P6-S3 | Emergency warning banner | ✅ Completed |
| P6-S4 | Navigation button to recommendations page | ✅ Completed |

### P7 — Frontend: Recommendations Page
| Step | Description | Status |
|------|-------------|--------|
| P7-S1 | Create templates/recommendations.html | ✅ Completed |
| P7-S2 | Disease information section | ✅ Completed |
| P7-S3 | Low risk maintenance plan section | ✅ Completed |
| P7-S4 | Disclaimer section | ✅ Completed |
| P7-S5 | Back/Start Over button | ✅ Completed |

### P8 — Testing & Final Polish
| Step | Description | Status |
|------|-------------|--------|
| P8-S1 | Run train.py and confirm model files generated | ✅ Completed |
| P8-S2 | Test full prediction flow end-to-end | ✅ Completed |
| P8-S3 | Test input validation | ✅ Completed |
| P8-S4 | Test edge cases | ✅ Completed |
| P8-S5 | Final UI review | ✅ Completed |

---

> **Legend:** ⬜ Pending | 🔄 In Progress | ✅ Done
