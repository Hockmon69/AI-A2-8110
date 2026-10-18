# Musanze Harvest Intelligence Service
**An Individual Reproducible Machine Learning System, Multi-Task Pipeline, and Validated REST API**

* **Student Full Name:** Deng Fajwan Changjwok Fajwan
* **Registration Number:** 25/28110 (Clean Digits: `2528110`)
* **Institutional Username:** `ug2528110@ines.ac.rw`
* **Verification Code:** `AI-A2-8110` (Personal Seed: `8110`)
* **Course Module:** SWE 3513 &ndash; Artificial Intelligence
* **Institution:** Institut d'Enseignement Supérieur de Ruhengeri (INES-Ruhengeri), Rwanda
* **Submission Package:** `AI_A2_2528110.zip`
* **Dataset SHA-256 Fingerprint:** `6f94df679bcd2c644aab4eceed2cfb02b09a5ddddbfa920c7dc2c018d59e1d5d`
* **Final Git Commit Hash:** `a6f2d4ae65bc83e1d62e5f4397b97e8669d382eb` *(Updated below upon repository initialization)*
* **Repository URL:** `https://github.com/fajwan/musanze-harvest-intelligence`

---

## 1. Problem Scenario & Professional Purpose
A potato cooperative in Musanze District, Northern Province, Rwanda, receives farm records in fragmented, inconsistent spreadsheets from member smallholders. Its agricultural field officers require an evidence-based, reproducible intelligence tool capable of:
1. **Estimating Harvest Quantity:** Continuous regression prediction of potato harvest volume in metric tons.
2. **Identifying High-Risk Parcels:** Binary classification triage flagging parcels where yield per hectare falls below the regional median.
3. **Agronomic Segmentation:** Unsupervised clustering of agricultural holdings to allocate cooperative agronomic support, input subsidies, and extension training.

Because rural network connectivity can be unreliable, the service operates entirely offline from the command-line, while also providing a high-performance local FastAPI service for integration with future mobile client applications.

> **Dataset Status Declaration:** The underlying farm dataset is **entirely synthetic**, generated deterministically according to the INES SWE 3513 mathematical specifications. It does not reflect real personal identities or private cooperative financial data.

---

## 2. Tested Technical Environment & Dependencies
* **Python Version:** Python 3.11.2 (compatible with Python &ge; 3.10)
* **Tested Operating System:** Ubuntu 22.04 LTS (Linux x86_64)
* **Core Libraries:**
  * Numerical & Data Processing: `numpy>=1.24.0`, `pandas>=2.0.0`, `scipy>=1.10.0`
  * Machine Learning & Modeling: `scikit-learn>=1.3.0`, `joblib>=1.3.0`
  * Visualization: `matplotlib>=3.7.0`, `seaborn>=0.12.0`
  * REST API & Testing: `fastapi>=0.100.0`, `uvicorn>=0.22.0`, `pydantic>=1.10.0,<3.0.0`, `pytest>=7.4.0`, `httpx>=0.23.0`

---

## 3. Exact Setup & Reproduction Commands

### Step A: Clone & Environment Activation
```bash
# Clone repository or extract submission package
unzip AI_A2_2528110.zip
cd AI_A2_2528110

# Create and activate isolated Python virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install locked dependencies
pip install -r requirements.txt
```

### Step B: Single-Command End-to-End Pipeline Execution
To recreate the dataset from seed 8110, audit data quality, perform train-only preprocessing, train all models, and regenerate every artifact:
```bash
python run_all.py --student-id ug2528110@ines.ac.rw --seed 8110
```

### Step C: Command-Line Single-Record Prediction (CLI)
```bash
# Valid inference
python predict.py --json samples/valid.json

# Deliberately invalid inference (exercises range validation error)
python predict.py --json samples/invalid.json
```

### Step D: Automated Test Suite Execution
```bash
pytest -q
# Or for detailed test coverage:
pytest -v
```

### Step E: FastAPI Service Deployment
```bash
# Launch the ASGI uvicorn server locally
uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```
Interactive API documentation is accessible at: `http://127.0.0.1:8000/docs`

---

## 4. API Endpoints & Request/Response Examples

### 1. `GET /health`
Returns service operational readiness and verification code.
```bash
curl -X GET http://127.0.0.1:8000/health
```
**Response (200 OK):**
```json
{
  "status": "healthy",
  "service": "Musanze Harvest Intelligence Service",
  "verification_code": "AI-A2-8110",
  "cooperative": "Musanze Potato Farmers Cooperative",
  "district": "Musanze, Northern Province, Rwanda"
}
```

### 2. `GET /model-info`
Returns task models, held-out test metrics, and dataset verification metadata.
```bash
curl -X GET http://127.0.0.1:8000/model-info
```

### 3. `POST /predict`
Executes multi-task inference on an incoming farm record.
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "rainfall_mm": 950.5,
    "avg_temp_c": 19.2,
    "soil_moisture_pct": 52.4,
    "fertilizer_kg": 120.0,
    "field_size_ha": 2.1,
    "season": "A",
    "cooperative_support": 1
  }'
```
**Response (200 OK):**
```json
{
  "verification_code": "AI-A2-8110",
  "predicted_yield_tons": 12.3637,
  "high_risk_prediction": 0,
  "risk_label": "Low Risk",
  "risk_probability": 0.4476,
  "farm_cluster": 1,
  "input_features": {
    "rainfall_mm": 950.5,
    "avg_temp_c": 19.2,
    "soil_moisture_pct": 52.4,
    "fertilizer_kg": 120.0,
    "field_size_ha": 2.1,
    "season": "A",
    "cooperative_support": 1
  }
}
```

**Invalid Request Error (422 Unprocessable Entity):**
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"rainfall_mm": 2500.0, "season": "Z"}'
```

---

## 5. Directory & Artifact Inventory
```
AI_A2_2528110/
├── README.md                              # Complete technical documentation & guide
├── requirements.txt                       # Version-locked Python dependencies
├── run_all.py                             # Master reproducible pipeline script
├── predict.py                             # Command-line single-record prediction interface
├── app.py                                 # FastAPI REST microservice
├── src/                                   # Reusable Python modules
│   ├── __init__.py                        # Package initialization
│   ├── data_generator.py                  # Seed-based synthetic dataset generator
│   ├── validation.py                      # Data & inference range/type validation
│   ├── preprocessing.py                   # Train-only imputer, scaler, and encoder
│   ├── models_regression.py               # Mean Baseline, OLS, and Ridge Regression
│   ├── models_classification.py           # Majority Baseline, Logistic, and Gaussian NB
│   ├── models_clustering.py               # KMeans and silhouette-based selection
│   ├── evaluation.py                      # MAE, RMSE, R2, F1, ROC-AUC, Silhouette
│   └── pipeline.py                        # Master training orchestrator & plotter
├── data/
│   ├── generate_data.py                   # Standalone data generation runner
│   └── AI_A2_2528110.csv                  # Serialized dataset (276 rows, 4 decimals)
├── models/                                # Serialized trained models & preprocessors
│   ├── preprocessor.joblib                # Fitted StandardScaler and median imputer
│   ├── regression_model.joblib            # Selected Ridge Regression model
│   ├── classification_model.joblib        # Selected Gaussian Naive Bayes classifier
│   └── clustering_model.joblib            # Selected KMeans (K=5) clustering model
├── artifacts/                             # Evaluation evidence and plots
│   ├── data_fingerprint.txt               # Authoritative SHA-256 fingerprint
│   ├── regression_metrics.json            # Regression metrics (MAE, RMSE, R2)
│   ├── classification_metrics.json        # Classification metrics & confusion matrix
│   ├── clustering_report.json             # Silhouette scores for K=2..6 & profiles
│   ├── test_predictions.csv               # 55 test records with errors & clusters
│   └── plots/
│       ├── residuals.png                  # Regression residual distribution
│       ├── confusion_matrix.png           # Classification confusion matrix heatmap
│       └── clusters.png                   # Farm cluster scatter visualization
├── samples/
│   ├── valid.json                         # Valid farm record for inference
│   └── invalid.json                       # Deliberately invalid record for error testing
├── tests/
│   ├── __init__.py                        # Test package initialization
│   └── test_pipeline.py                   # 8 automated unit & integration tests
├── docs/
│   └── AI_A2_2528110_REPORT.pdf           # 7-page comprehensive technical report
└── evidence/
    ├── AI_A2_2528110_DEMO.mp4             # Demonstration video MP4
    ├── AI_A2_2528110_TEST_LOG.pdf         # Formatted automated test execution log
    ├── AI_A2_2528110_DECLARATION.pdf      # Signed academic integrity declaration
    └── AI_USE.md                          # Transparent AI assistance disclosure
```

---

## 6. Known Limitations & Responsible AI Reflections
1. **Simplified Synthetic Physics:** Real Irish potato farming in Musanze experiences extreme volcanic micro-climate variations across Mount Bisoke and Mount Karisimbi elevation contours that are abstracted in synthetic models.
2. **Biological Shock Exclusions:** The model does not include biological variables such as Late Blight (*Phytophthora infestans*) outbreaks or bacterial wilt.
3. **Mandatory Human Oversight:** Predictions are strictly advisory decision support for cooperative officers and must never be used to penalize farmers or deny agricultural subsidies.
