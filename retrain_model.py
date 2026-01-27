import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Load the data
print("Loading data...")
df = pd.read_csv('Final_Clean_Data.csv')

# Based on app.py, these are the inputs:
input_cols = ['DEPT', 'WOB', 'RPM', 'Flow in', 'Torque', 'SPP', 'M.Wt in', 'M.Temp in', 'M.Temp out', 'SSS_H']
# Based on app.py, these are the targets:
target_cols = ['SSL_H', 'VIBXYH', 'VIBZH']

# Select relevant columns
data = df[input_cols + target_cols].copy()

# Handing missing values like in the notebook
data = data.fillna(data.mean())

X = data[input_cols]
y = data[target_cols]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=44)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model (using RandomForestRegressor as suggested by notebook)
print("Training model...")
model = RandomForestRegressor(n_estimators=100, random_state=44)
model.fit(X_train_scaled, y_train)

# Evaluate
score = model.score(X_test_scaled, y_test)
print(f"Model R^2 Score: {score:.4f}")

# Save the model and scaler
print("Saving model and scaler...")
joblib.dump(model, 'models/model.h5')
joblib.dump(scaler, 'models/scaler.h5')
print("Retraining complete. Files saved to models/model.h5 and models/scaler.h5")
