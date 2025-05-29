# 🧠 Stroke Prediction using Machine Learning

This project leverages machine learning to predict the likelihood of a stroke based on various health-related attributes. The dataset is sourced from Kaggle and is used to train and evaluate the model.

## 📊 Dataset

We use the [Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset) from Kaggle, which includes demographic and medical information relevant to stroke risk.

## 🛠️ Development Setup

### 1. Clone the Repository

```bash
git clone https://github.com/sin22766/stroke-prediction-api.git
cd stroke-prediction-api
```

### 2. Set Up the Environment

We recommend using [Astral uv](https://astral.sh/uv/) to manage dependencies in a clean and efficient virtual environment.

```bash
uv sync
uv pip install -e .
```

> **Note:** Installing the project in *editable* mode (`-e`) allows for easier development and testing.

## 📦 Data Versioning with DVC

We use [DVC (Data Version Control)](https://dvc.org/) to manage datasets and model artifacts. The DVC remote is an S3 bucket, so AWS credentials are required to access it.

### Set up AWS credentials:

```bash
aws configure
```

> 📌 Contact the project maintainer to obtain access credentials for the S3 bucket.

### Pull data with DVC:

```bash
dvc pull
```

Or, if you haven't installed DVC globally, use the bundled binary via `uv`:

```bash
uv run dvc pull
```

## ⚙️ Environment Configuration

Create a `.env` file in the project root with the following content:

```
API_KEY=your_api_key_here
```

## 🚀 Running the App (Locally)

Use the following command to start the FastAPI development server:

```bash
fastapi dev src/app/main.py
```

## 📦 Deployment

This project supports Docker-based deployment. Ensure your `.env` file is correctly set and that AWS credentials are configured to allow DVC access to model artifacts.

### Build the Docker image:

```bash
docker compose build
```

### Run the Docker container:

```bash
docker compose up -d
```
