# Imports
import pandas as pd # Dataframe
import numpy as np # Numerical operations
import xgboost as xgb # Pattern Recognition ML model
from sklearn.metrics import classification_report, average_precision_score, confusion_matrix


# Step 1: Reading data

df_train = pd.read_csv("fraudTrain.csv") # Reads the csv into a dataframe
df_test = pd.read_csv("fraudTest.csv")
print(df_train.shape) # Outputs (Rows, Columns)

df_train["trans_date_trans_time"] = pd.to_datetime(df_train["trans_date_trans_time"]) # Converts object/teX_traint date and time into datetime propetry in pandas
df_train["hour"] = df_train["trans_date_trans_time"].dt.hour # Gives the time in hour form (i.e. 3:15 am -> 3)
df_train = df_train.sort_values( # Sorts by date
    by="trans_date_trans_time"
).reset_index(drop=True)

# Step 2: Choosing features to train on

X_train = df_train[["amt", "hour", "category", "city_pop"]] # Creates a feature dataframe using columns from df
y_train = df_train["is_fraud"] # Results column

X_train = pd.get_dummies(X_train, columns=["category"], dtype=int) # Converts catagories into a binary yes/no table (One-hot enconding)
split_idx = int(len(X_train) * 0.8)

X_fit = X_train.iloc[:split_idx]  # First 80%
X_val = X_train.iloc[split_idx:]  # Remaining 20%

y_fit = y_train.iloc[:split_idx]  # First 80%
y_val = y_train.iloc[split_idx:]  # Remaining 20%

# Step 3: Set parameters/hyperparameters and train

model = xgb.XGBClassifier(
    n_estimators=100, # How many rounds
    max_depth=4, # How many questions
    learning_rate=0.1, # Contribution of change on model
    objective="binary:logistic", # Evals to two states (valid or fraud)
    n_jobs=-1 # Use all available cpu threads rather than setting a specific quantity
)

model.fit(X_fit, y_fit)
print("Training complete!")


# Step 4: Evaluate model using validation data

y_pred = model.predict(X_val) # Predicts whether each validation transaction is fraudulent
fraud_probs = model.predict_proba(X_val)[:, 1] # Gets the predicted fraud probability for each transaction

print("\nVALIDATION RESULTS")

print("\nConfusion Matrix:")
print(confusion_matrix(y_val, y_pred)) # Shows correct and incorrect predictions

print("\nClassification Report:")
print(classification_report(y_val, y_pred, zero_division=0)) # Shows precision, recall, F1-score and accuracy

print("PR-AUC:", average_precision_score(y_val, fraud_probs)) # Measures fraud detection performance across different thresholds


# Step 5: Prepare and evaluate test data

df_test["trans_date_trans_time"] = pd.to_datetime(df_test["trans_date_trans_time"]) # Converts object/teX_train_traint date and time into datetime propetry in pandas
df_test["hour"] = df_test["trans_date_trans_time"].dt.hour # Gives the time in hour form (i.e. 3:15 am -> 3)
df_test = df_test.sort_values(
    by="trans_date_trans_time"
).reset_index(drop=True)

X_test = df_test[["amt", "hour", "category", "city_pop"]] # Creates a feature dataframe using columns from df
y_test = df_test["is_fraud"] # Results column

X_test = pd.get_dummies(X_test, columns=["category"], dtype=int) # Converts catagories into a binary yes/no table (One-hot enconding)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0) # Ensures test columns match training columns in the same order
print(X_test.shape) # Outputs (Rows, Columns)

test_pred = model.predict(X_test) # Predicts fraud on unseen test transactions
test_probs = model.predict_proba(X_test)[:, 1] # Gets fraud probabilities for test transactions

print("\nTEST RESULTS")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, test_pred))

print("\nClassification Report:")
print(classification_report(y_test, test_pred, zero_division=0))

print("PR-AUC:", average_precision_score(y_test, test_probs))