import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv(
    r"C:\Users\prakash\OneDrive\Documents\GitHub\Heart-disease-prediction-ML\data\heart.csv"
)

print(data.head())

# Features & target
X = data.drop("target", axis=1)
y = data["target"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save model & scaler
with open(
    r"C:\Users\prakash\OneDrive\Documents\GitHub\Heart-disease-prediction-ML\model.pkl",
    "wb"
) as f:
    pickle.dump((model, scaler), f)

print("Model saved successfully")
