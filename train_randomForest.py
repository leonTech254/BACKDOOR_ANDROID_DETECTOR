import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset into a pandas DataFrame
data = pd.read_csv("./DATASET/android_traffic.csv")

# Display some basic information about the dataset
print("Shape:", data.shape)
print("Columns:", data.columns)
print(data.head())

# Split the dataset into features (X) and target (y)
X = data.drop("type", axis=1)
y = data["type"]

# Encode categorical features using LabelEncoder
le = LabelEncoder()
X["name"] = le.fit_transform(X["name"])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a Support Vector Machine (SVM) classifier with linear kernel
svm = SVC(kernel='linear', C=1)
svm.fit(X_train_scaled, y_train)

# Make predictions on the testing set and evaluate the model
y_pred = svm.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification report:\n", report)
