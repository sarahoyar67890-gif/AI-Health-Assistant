import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
#LOAD DATA
df = pd.read_csv("dataset.csv")
#MERGE SYMPTOMS
symptom_cols = df.columns[1:]
df["all_symptoms"] = df[symptom_cols].apply(lambda row: " ".join(row.dropna().astype(str).values),axis=1)
#FEATURES & LABEL
X_text = df["all_symptoms"]
y = df["Disease"]
# VECTORIZATION
cv = CountVectorizer()
X = cv.fit_transform(X_text).toarray()
#TRAIN TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#BEST MODEL
model = RandomForestClassifier(n_estimators=300,max_depth=15,min_samples_split=10,min_samples_leaf=2,random_state=42)
model.fit(X_train, y_train)
#EVALUATION
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)
#SAVE MODEL
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(cv, open("vectorizer.pkl", "wb"))
print("Model saved successfully")