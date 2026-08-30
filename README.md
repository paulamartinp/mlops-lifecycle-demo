# 🛒 MLOps End-to-End Lifecycle Demo: Walmart Sales Forecasting

An enterprise-grade, production-ready MLOps implementation for predicting weekly sales across Walmart stores and departments using the **[Walmart Store Sales Forecasting Dataset](https://www.kaggle.com/datasets/aslanahmedov/walmart-sales-forecast)**. This repository showcases a complete machine learning lifecycle: data version control; experiment tracking, automated CI/CD pipelines, model registry, containerized serving, and performance monitoring.

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/badge/package%20manager-uv-purple.svg)](https://github.com/astral-sh/uv)
[![DVC](https://img.shields.io/badge/reproducibility-DVC-orange.svg)](https://dvc.org/)
[![MLflow](https://img.shields.io/badge/tracking-MLflow-blueviolet.svg)](https://mlflow.org/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-green.svg)](https://fastapi.tiangolo.com/)
[![Gradio](https://img.shields.io/badge/UI-Gradio-red.svg)](https://gradio.app/)

---
## 📚 Table of Contents

- [🎯 Project Objectives](#-project-objectives)
- [📁 Repository Structure](#-repository-structure)
- [📊 Data Overview](#-data-overview)
- [🛠 Technologies & Architecture](#-technologies--architecture)
- [⚙ Centralized Configuration](#-centralized-configuration)
- [🚀 Quick Start](#-quick-start)
- [🕹️ How to Run the Project](#️-how-to-run-the-project)
- [🏅 Kaggle Submission](#-kaggle-submission-overview)
- [🧩 Learning Path (Git Branches)](#-step-by-step-learning-path-git-branches)
- [🧭 Overview (Screenshots & Details)](#-overview)
- [🚀 Future Roadmap](#-future-roadmap--improvements)
- [👨‍💻 Author](#-author)
- [📄 License](#-license)


## 🎯 Project Objectives

This project demonstrates a production-ready, end-to-end MLOps implementation designed to showcase industry best practices:

* **Pipeline Reproducibility:** Build fully reproducible machine learning pipelines using [DVC](https://dvc.org/).
* **Experiment Tracking** Centralized metrics, parameters, and model registry via [MLflow](https://mlflow.org/).
* **Model Serving:** Expose model predictions via [FastAPI](https://fastapi.tiangolo.com/) backend.
* **Interactive Frontend:** Provide an intuitive user interface utilizing [Gradio](https://gradio.app/).
* **Production Monitoring:** Detect data and model drift post-deployment using [Evidently AI](https://www.evidentlyai.com/).
* **Containerization:** Package multi-service micro-architectures using [Docker](https://www.docker.com/).
* **CI/CD Automation:** Automate code quality checks, testing, and deployment workflows via [GitHub Actions](https://github.com/features/actions).

![MLOps](docs/images/overview.png)

> **Note**
> This project uses the public Walmart Store Sales Forecasting dataset from Kaggle. All examples, metrics, and artifacts shown in this repository are generated exclusively from this dataset. No proprietary, sensitive, or confidential business information is used or exposed at any stage.

---

## 📁 Repository Structure

```text
mlops-lifecycle-demo/
│
├── api/                  # FastAPI service for model predictions
├── data/                 # Datasets used for training and inference
├── db/                   # Database files
├── docs/                 # Documentation assets (images, guides)
├── logs/                 # Application and execution logs
├── mlruns/               # MLflow tracking directory
├── models/               # Saved model artifacts (e.g., champion.pkl)
├── monitoring/           # Model drift detection and reporting scripts
├── notebooks/            # Jupyter notebooks for EDA and experimentation
├── src/                  # Core source code (pipelines, features, utils)
│   ├── data/             # Data ingestion and preprocessing scripts
│   ├── features/         # Feature engineering scripts
│   ├── models/           # Model training and evaluation scripts
│   ├── pipelines/        # ML pipelines orchestration
│   └── utils/            # Helper utilities
├── ui/                   # Frontend user interface (Streamlit/Gradio)
│
├── .dockerignore         # Docker ignore file
├── .env                  # Local environment variables
├── .env.example          # Example environment configuration
├── .gitignore            # Git ignore file
├── docker-compose.yml    # Docker Compose multi-container setup
├── Dockerfile.api        # Dockerfile for the API service
├── Dockerfile.ui         # Dockerfile for the UI service
├── dvc.yaml              # DVC pipeline definition
└── mlflow.db             # MLflow SQLite backend database
```

---

## 📊 Data Overview

**Goal:** Predict weekly sales across different Walmart stores and departments.  
**Target:** `Weekly_Sales`  
**Source:** Kaggle's [Walmart Store Sales Forecasting](https://www.kaggle.com/datasets/aslanahmedov/walmart-sales-forecast) dataset.  

**Files used:** `train.csv`, `test.csv`, `stores.csv`, `features.csv`

---

## 🛠 Technologies and Architecture


![MLOps Tech Stack](docs/images/tech_stack.jpg)

---

## ⚙ Centralized Configuration

All pipeline settings, feature definitions, and model hyperparameters are centralized in a single configuration file:

[`params.yaml`](./params.yaml)

This file serves as the source of truth for the entire MLOps workflow. Any update to model settings, feature lists, or training parameters is managed here and automatically propagated across the pipeline.

### Example

```yaml
train:
  target: Weekly_Sales
  models:
    - baseline
    - xgboost
    - lightgbm
  feature_set:
    - Store
    - Dept
    - IsHoliday
    ...
```

Any change made in `params.yaml` flows through DVC pipelines, MLflow runs, and the serving stack, ensuring a fully aligned and reproducible lifecycle.

---

## 🚀 Quick Start

### 1. Prerequisites

Before running the project, make sure you have the following tools installed and available.

---

### 1.1) Python and `uv` Package Manager (Required)

This project requires **Python 3.11+** and uses **[uv](https://github.com/astral-sh/uv)** for fast, reproducible dependency management.

* **Python:** Download and install version 3.11 or newer from the [official Python website](https://www.python.org/downloads/). Verify your installation:
  ```bash
  python --version
  ```
* **UV**: Install `uv` via the [official installation guide](https://docs.astral.sh/uv/getting-started/installation/).
* Verify the installation in the terminal:

  ```bash
  uv --version
  ```

---

### 1.2). Docker Desktop (Optional but Recommended)

If you plan to run the FastAPI and Gradio services via Docker Compose, you’ll need Docker Desktop.

> Note: Skip this requirement if you prefer running services locally via terminal commands.

* Install the latest version of [Docker Desktop](https://www.docker.com/products/docker-desktop/).

* Ensure Docker Desktop is **open and running** before executing any Docker commands.


---


### 1.3) Kaggle API Token (Optional)

A Kaggle account and API token are **not required** for the standard setup because all datasets and artifacts are already versioned with DVC and stored in a public S3 remote.

You only need a Kaggle API token if you want to:

- Re-download and reproduce the pipeline from the original Kaggle dataset  
- Rebuild the dataset from scratch using `dvc repro`  
- Recover the project if the DVC remote becomes unavailable  

To generate a Kaggle API token:

1. Sign in to [Kaggle](https://www.kaggle.com/)
2. Open **Account Settings**
3. Scroll to the **API** section and click **Generate New Token**:

   ```text
   Kaggle ➔ Settings ➔ API ➔ Generate New Token
   ```

![Kaggle API Token](docs/images/kaggle_token.jpg)

> ⚠️ **Important:** Kaggle only provides this key at the moment of creation. Save or copy your credentials immediately—if you lose them, you will need to generate a new key.

---

### 2. Setup & Installation

### 2.1) Step 1 — Clone the Repository

```bash
git clone https://github.com/paulamartinp/mlops-lifecycle-demo.git
```
```bash
cd mlops-lifecycle-demo
```
---

### 2.2) Step 2 — Install Dependencies with `uv`
Synchronize the environment from the project root:

```bash
uv sync
```

This command:

- Creates an isolated .venv

- Installs all dependencies from the lockfile

> *(Learn more in the [uv project layout documentation](https://docs.astral.sh/uv/concepts/projects/layout/))*

---

### 2.3) Step 3 — Activate the Virtual Environment
**Windows (PowerShell):**
```powershell
.\.venv\Scripts\Activate
```

**macOS / Linux:**
```bash
source .venv/bin/activate
```

Your terminal prompt will now show the environment name.

### 2.4) Step 4 — Configure Environment Variables (Optional)
> Note:
If you only want to run the project using pre‑built DVC artifacts, this step is optional.

If you plan to rebuild the dataset from Kaggle, configure your credentials:

```bash
cp .env.example .env
```

Open `.env` and fill in your details:

```env
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_api_key
```
> Press Ctrl + S to save your changes
---

---

## 🕹️ How to Run the Project

Run the full automated workflow—from data ingestion and model training to experiment tracking and production serving.

---

### Prerequisites & Quick Start Overview

Choose your workflow depending on your goal:

| Goal | Required Steps |
| :--- | :--- |
| **Just run / evaluate the model** | Step 1 (`dvc pull`) ➔ Step 3 (Serving) |
| **Train models & track experiments** | Step 1 (`dvc pull` or `dvc repro`) + Step 2 (MLflow & Pipeline) ➔ Step 3 |

---

### Step 1 — Sync Data & Artifacts (DVC)
Download the latest dataset and artifacts from the public S3 remote:

```bash
dvc pull
```

This retrieves:

* Raw datasets

* Intermediate artifacts

* Feature stores

* Trained models


> If `dvc pull works`, you **do not need to retrain** unless you want to inspect experiments or modify the pipeline.

> If the S3 bucket becomes unavailable, or anything breaks at this step, regenerate everything locally using Step 2.

---

### Step 2 — Execute the MLOps Pipeline (Optional)
Use this step only if you want to retrain models, log metrics, or inspect experiments.

#### 2.1) Start the MLflow Tracking Server
Run the following command from the project root:

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 127.0.0.1 --port 5000 --workers 1
```
Once the server starts, you should see logs similar to:

```text
2026/08/25 22:28:41 INFO:     Started server process [7896]
2026/08/25 22:28:41 INFO:     Waiting for application startup.
2026/08/25 22:28:41 INFO:     Application startup complete.
2026/08/25 22:28:41 INFO:     Uvicorn running on http://127.0.0.1:5000 (Press CTRL+C to quit)
```
This means the MLflow UI is available at http://127.0.0.1:5000. You can open it in your browser.

> Keep this terminal running. The MLflow server must stay active while you execute the rest of the pipeline.

---

#### 2.2) Run the Pipeline (Choose One)

**Option A — DVC Orchestration (Recommended)**
```bash
dvc repro
```

This executes the complete data and model lifecycle DAG stages:

```text
Data Ingestion ➔ SQLite Load ➔ Feature Engineering ➔ Model Training ➔ MLflow Tracking ➔ Model Registry
```

**Option B — Python Wrapper**
```bash
python -m src.pipelines.training_pipeline
```
> Same workflow, but without DVC’s reproducibility guarantees.

---

### Step 3 — Production Serving (Choose One)

Expose predictions via API + UI.

---

**Option A — Dockerized Deployment (Recommended)**

With Docker Desktop open, run the following command in the project's root directory:

```bash
docker compose up --build
```

**Option B — Local Execution**

* Terminal 1: Run the FastAPI Backend
    ```bash
    uvicorn api.api:app --reload --host 0.0.0.0 --port 8000
    ```

* Terminal 2: Run the Interactive UI
    ```bash
    python -m ui.app
    ```

Once running (via either method), access your services at:

* API Endpoint: http://localhost:8000 (Interactive Swagger docs available at /docs)

* Interactive UI: http://localhost:7860


### Step 4: Production Logging & Drift Monitoring
As predictions are made (either through the API or the UI), they are automatically appended to:

```bash
monitoring/prediction_logs.csv
```

Generate the drift report:
```bash
python -m monitoring.drift
```
Output:
```bash
reports/drift_report.html
```

A complete HTML dashboard with feature stability and drift metrics.

---

## 🏅 Kaggle Submission (Overview)

Although leaderboard ranking is **out of scope**, you can benchmark your model against Kaggle’s official test set at any time.

The submission pipeline automatically uses the current **`@champion`** model from the MLflow Model Registry:

```bash
uv run python -m src.pipelines.inference_pipeline
```

This generates a `submission.csv` file in the project root, fully formatted and ready to upload to Kaggle.

🔗 **Competition Link:** [Walmart Recruiting - Store Sales Forecasting](https://www.kaggle.com/competitions/walmart-recruiting-store-sales-forecasting)

![Kaggle Submission](docs/images/submission.png)

---

## 🧩 Step-by-Step Learning Path (Git Branches)

This project includes a progressive learning path through dedicated feature branches.  
Each branch isolates a core MLOps concept so you can explore the system incrementally.

| Branch | Focus | MLOps Concept |
|--------|--------|----------------|
| `feature/01-data-ingestion` | Data download & extraction | Automated ingestion |
| `feature/02-sql-layer` | SQLite storage | Structured persistence |
| `feature/03-eda` | EDA & quality checks | Data validation |
| `feature/04-data-preparation` | Preprocessing & feature pipelines | Modular transformations |
| `feature/05-model-training` | Model development | Benchmarking & tuning |
| `feature/06-mlflow` | MLflow integration | Experiment tracking |
| `feature/07-dvc` | Data versioning | Reproducible pipelines |
| `feature/08-fast-api` | REST API | Production serving |
| `feature/09-gradio-ui` | Interactive UI | Stakeholder accessibility |
| `feature/10-monitoring` | Drift & observability | Post-deployment monitoring |
| `feature/11-docker` | Containerization | Environment parity |
| `feature/12-ci-cd` | CI/CD automation | Continuous integration & delivery |

> Switch modules with: `git checkout <branch-name>`.

---

## 🧭 Overview (Screenshots & Details)

### Model Training
Unified automated pipeline training **Baseline**, **XGBoost**, and **LightGBM** models.  
Includes feature validation, chronological splits, evaluation, and MLflow logging.

---

### MLflow Experiment Tracking
MLflow logs all training runs, including hyperparameters, metrics (MAE, RMSE), execution time, and artifacts.

![MLflow Experiments](docs/images/mlflow_ui.png)

---

### Model Registry
The best model is promoted to the MLflow Model Registry under the **`@champion`** alias.  
Both API and UI always load this alias for seamless model updates.

---

### FastAPI Service
Production-ready API exposing:
- `GET /health` — service + registry status  
- `POST /predict` — real-time weekly sales prediction  

![API docs](docs/images/api.png)

---

### Gradio UI
Interactive UI for single predictions, automatically using the current `@champion` model.

![Gradio](docs/images/gradio.png)

---

### Monitoring & Drift
Predictions are logged to `monitoring/prediction_logs.csv` and compared against a reference dataset.  
Evidently AI generates an HTML dashboard showing feature and target drift.

![Monitoring](docs/images/drift.png)

---

### Dockerization
Full multi-service Docker setup (API + UI).  
Run the entire stack in containers or mix local execution with Docker services.

![Docker](docs/images/docker.png)

---

### CI/CD
Automated GitHub Actions pipeline:

**CI:**  
- Dependency setup with `uv`  
- Tests via `pytest`  
- DVC pipeline checks  
- Docker build validation  

**CD:**  
- Triggered on pushes to `main`  
- Authenticates with Docker Hub  
- Builds & tags production API/UI images  
- Pushes updated images to Docker Hub  

---

## 🚀 Future Roadmap & Improvements
Planned enhancements:
- Hyperparameter tuning (Optuna)  
- Automated retraining on drift  
- Batch inference orchestration (Prefect / Airflow)  
- Cloud deployment (AWS / GCP / Azure)  
- Kubernetes deployment (KServe / Helm)  
 
---

## 👨‍💻 Author

[![LinkedIn – Paula Martín Palomeque](https://img.shields.io/badge/LinkedIn-Paula%20Mart%C3%ADn%20Palomeque-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/paula-mart%C3%ADn-palomeque/)


---

## 📄 License
MIT License

Copyright (c) 2026 Paula Martín-Palomeque