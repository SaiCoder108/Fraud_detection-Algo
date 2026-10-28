import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

# fake hourly data: demand depends on hour of day and temperature
rng = np.random.default_rng(0)
idx = pd.date_range("2022-01-01", periods=24 * 365 * 2, freq="h")
temp = 10 + 12 * np.sin(2 * np.pi * (idx.dayofyear - 110) / 365) + rng.normal(0, 2, len(idx))
demand = 1000 + 150 * np.sin(2 * np.pi * (idx.hour - 6) / 24) + 8 * np.abs(temp - 18) + rng.normal(0, 20, len(idx))
df = pd.DataFrame({"temp": temp, "demand": demand}, index=idx)

# features
df["hour"] = df.index.hour
df["lag24"] = df["demand"].shift(24)
df["temp_roll6"] = df["temp"].shift(1).rolling(6).mean()
df = df.dropna()

X = df[["hour", "temp", "lag24", "temp_roll6"]] # Inputs
y = df["demand"] # Desired outputs

# chronological split
split = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:] # Splits data set to training data and out-of-sample test
y_train, y_test = y.iloc[:split], y.iloc[split:]

# model
model = XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05) 
# Hyper-parameters at set and fine tuned using eval (train -> test -> eval)

model.fit(X_train, y_train) # Fits models to iputs and desired outputs

# evaluate
preds = model.predict(X_test) # Inputing never-seen inputs
print("MSE:", mean_squared_error(y_test, preds)) # Comparing desired outputs to prediction
# Loss function: Default-> Mean Sqaure Error

print(dict(zip(X.columns, model.feature_importances_.round(3))))
print("train MSE:", mean_squared_error(y_train, model.predict(X_train)))
print("RMSE:", mean_squared_error(y_test, preds) ** 0.5)
print("MAPE %:", ((y_test - preds).abs() / y_test).mean() * 100)