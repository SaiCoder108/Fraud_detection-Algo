# Credit Card Fraud Detection using XGBoost

A machine learning project that uses **XGBoost** to identify potentially fraudulent credit card transactions based on transaction characteristics.

The project uses Python, Pandas, and XGBoost to preprocess transaction data, train a classification model, and evaluate its performance on unseen transactions.

## Overview

The goal is to develop a functional fraud detection system capable of estimating the likelihood that a transaction is fraudulent.

Currently, the project focuses on building and evaluating the machine learning model. Future development will include an interactive interface for entering transaction details and receiving fraud-risk predictions.

## Dataset

**Source:** [Credit Card Transactions Fraud Detection — Kaggle](https://www.kaggle.com/datasets/kartik2112/fraud-detection)

The dataset contains simulated credit card transactions, including legitimate and fraudulent activity.

Files:
- `fraudTrain.csv` — Training dataset (approximately 1.3 million transactions)
- `fraudTest.csv` — Independent testing dataset

### Features Used

| Feature | Description |
|---|---|
| `amt` | Transaction amount |
| `hour` | Hour of transaction (extracted from timestamp) |
| `category` | Merchant category, converted using one-hot encoding |
| `city_pop` | Population of the cardholder's city |
| `is_fraud` | Target variable: 0 = legitimate, 1 = fraudulent |

## Technologies

- **Python** — Main programming language
- **Pandas** — Data importing, preprocessing, and feature engineering
- **NumPy** — Numerical operations library
- **XGBoost** — Gradient-boosted decision tree classifier
- **Scikit-learn** — Model evaluation and performance metrics

## How It Works

### 1. Data Preprocessing

- Loads training and testing datasets
- Converts transaction timestamps into datetime values
- Extracts transaction hour
- Sorts training transactions chronologically
- Selects relevant features
- Converts merchant categories into numerical features using one-hot encoding

### 2. Model Training

The dataset is divided chronologically into:

- **80% training data** — Used to train the model
- **20% validation data** — Used to evaluate performance during development

The model uses the following baseline hyperparameters:

| Parameter | Value |
|---|---|
| Algorithm | XGBClassifier |
| Estimators | 100 |
| Maximum depth | 4 |
| Learning rate | 0.1 |
| Objective | binary:logistic |

### 3. Model Evaluation

The model is evaluated using:

- **Confusion Matrix** — Counts correct and incorrect classifications
- **Precision** — Proportion of flagged transactions that are actually fraudulent
- **Recall** — Proportion of fraudulent transactions successfully detected
- **F1-Score** — Harmonic mean of precision and recall
- **Average Precision (PR-AUC summary)** — Evaluates fraud-detection performance across classification thresholds

Because fraudulent transactions represent a small percentage of the dataset, accuracy alone is not a reliable measure of performance.

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY
```

Install dependencies:

```bash
pip install pandas numpy xgboost scikit-learn
```

Download the dataset from Kaggle and place `fraudTrain.csv` and `fraudTest.csv` in the project directory.

Run the Python script:

```bash
python main.py
```

*Replace `main.py` with your actual script filename if different.*

## Project Status

**Current stage: Model development and evaluation**

- [x] Dataset importing and preprocessing
- [x] Feature selection and one-hot encoding
- [x] Chronological training/validation split
- [x] XGBoost model training
- [x] Validation and testing evaluation code
- [ ] Model optimization and threshold selection
- [ ] Save trained model for reuse
- [ ] Interactive transaction input interface
- [ ] Real-time prediction functionality

## Future Improvements

- Incorporate additional transaction features, including geographic distance and transaction patterns
- Optimize model hyperparameters and fraud detection thresholds
- Build an interactive application using Streamlit
- Save and load trained models without retraining for every prediction
- Improve handling of class imbalance
- Add input validation and reusable preprocessing functions

## Limitations

The model is trained on simulated transaction data and uses a limited set of features. Its predictions should not be treated as reliable fraud determinations in real financial systems without further validation, monitoring, and testing.

## License

No license has been specified yet.
