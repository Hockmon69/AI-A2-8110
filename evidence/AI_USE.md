# AI Assistance Disclosure & Declaration
**Course**: SWE 3513 - Artificial Intelligence  
**Institution**: Institut d'Enseignement Supérieur de Ruhengeri (INES-Ruhengeri)  
**Student Name**: Deng Fajwan Changjwok Fajwan  
**Registration Number**: 25/28110  
**Verification Code**: AI-A2-8110  
**Date**: October 18, 2026  

---

## 1. Scope & Purpose of AI Assistance
In accordance with INES-Ruhengeri academic integrity policies and the SWE 3513 practical guidelines, this document provides complete, transparent disclosure of the artificial intelligence tools utilized during the design, implementation, evaluation, and documentation of Individual Assignment 2 ("Musanze Harvest Intelligence Service").

AI tools were employed as a collaborative programming accelerator and technical writing assistant under continuous human supervision and rigorous empirical verification.

---

## 2. Inventory of AI Tools Used
* **Primary AI Engine**: Large Language Model Assistant (Gemini / Google DeepMind).
* **Development Environment**: Headless Linux container development environment with Python 3.11, NumPy, Pandas, SciPy, Matplotlib, and FastAPI.

---

## 3. Stages of Engagement & Prompts

### Phase 1: Architectural Scaffolding & Mathematical Verification
* **Objective**: Translate the assignment specification into a modular, production-ready pipeline architecture under `src/`.
* **AI Contribution**: Helped formulate clean object-oriented class abstractions for preprocessing, baseline estimators, regularized linear models, probabilistic classifiers, and K-Means clustering.
* **Human Oversight & Verification**: Hand-verified the analytical yield formula ($0.004 \times \text{rainfall} + \dots$) and ensured the base row count formula ($240 + (8110 \pmod{41}) = 273$) and deliberate data injection (14 missing values, 3 exact duplicates) were implemented deterministically with `numpy.random.default_rng(8110)`.

### Phase 2: Pipeline Implementation & Anti-Leakage Controls
* **Objective**: Ensure absolute separation of train and test datasets, preventing target leakage.
* **AI Contribution**: Scaffolding the `PreprocessingPipeline` class to guarantee that median imputation and standard scaling are fitted exclusively on the 80% training set and applied without modification to the held-out 20% test set.
* **Human Oversight & Verification**: Implemented explicit programmatic assertions (`verify_no_target_leakage`) ensuring `yield_tons`, `yield_per_ha`, and `high_risk` can never enter the feature matrix $X$.

### Phase 3: REST API & Automated Test Suite
* **Objective**: Build FastAPI endpoints (`/health`, `/model-info`, `/predict`) with typed Pydantic validation and 8 robust unit/integration tests.
* **AI Contribution**: Assisted in drafting Pydantic input schemas with strict range bounds and setting up FastAPI `TestClient` test fixtures.
* **Human Oversight & Verification**: Executed `pytest -v` across valid paths and failure paths (missing fields, invalid data types, range violations), verifying all 8 tests pass deterministically.

### Phase 4: Technical Report Preparation
* **Objective**: Draft a rigorous 7-page technical report formatted to INES academic standards.
* **AI Contribution**: Assisted in compiling quantitative metrics into scannable Markdown tables and structuring discussions on error interpretation and cluster agronomic implications.
* **Human Oversight & Verification**: Audited every single reported metric (MAE, RMSE, $R^2$, Precision, Recall, F1, ROC-AUC, Silhouette scores) against the raw JSON outputs to ensure 100% numerical fidelity.

---

## 4. Student Attribution & Originality Affirmation
I, **Deng Fajwan Changjwok Fajwan**, affirm that:
1. All generated data, random sequences, model weights, evaluation metrics, and plots are uniquely derived from my personal registration number (`25/28110`, Seed: `8110`).
2. I have personally inspected, executed, and validated every line of Python code and configuration in this repository.
3. I understand the underlying mathematical mechanisms (Normal equation, L2 Ridge penalty, BFGS optimization, Gaussian Naive Bayes likelihoods, Lloyd's K-Means algorithm) and am fully prepared to defend this work in viva-voce examination and live code modification.
