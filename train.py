import pandas as pd
import numpy as np
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score

# ======================
# LOAD DATA
# ======================
df = pd.read_csv("dataset.csv")

# ======================
# MERGE SYMPTOMS
# ======================
symptom_cols = df.columns[1:]

df["all_symptoms"] = df[symptom_cols].apply(
    lambda row: " ".join(row.dropna().astype(str).values),
    axis=1
)

X_text = df["all_symptoms"].astype(str)
y = df["Disease"]

# ======================
# VECTORIZATION
# ======================
cv = CountVectorizer()
X = cv.fit_transform(X_text)

# ======================
# SPLIT
# ======================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ======================
# MODEL
# ======================
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    random_state=42
)

model.fit(X_train, y_train)

# ======================
# EVALUATION
# ======================
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# ======================
# SAVE PATH (IMPORTANT FIX)
# ======================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "model.pkl")
vectorizer_path = os.path.join(BASE_DIR, "vectorizer.pkl")

with open(model_path, "wb") as f:
    pickle.dump(model, f)

with open(vectorizer_path, "wb") as f:
    pickle.dump(cv, f)

print("Saved successfully ✔️")
print("Model path:", model_path)