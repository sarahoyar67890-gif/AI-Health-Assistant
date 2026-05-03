import streamlit as st
import numpy as np
import pickle
import os

# ======================
# LOAD MODEL SAFELY
# ======================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "model.pkl")
vectorizer_path = os.path.join(BASE_DIR, "vectorizer.pkl")

with open(model_path, "rb") as f:
    model = pickle.load(f)

with open(vectorizer_path, "rb") as f:
    cv = pickle.load(f)

# ======================
# UI
# ======================
st.set_page_config(page_title="AI Health Assistant", page_icon="🧠")

st.title("🧠 AI Health Assistant")
st.write("Select symptoms to predict disease")

# ======================
# SYMPTOMS
# ======================
symptoms = sorted(cv.get_feature_names_out())

selected = st.multiselect("Select Symptoms", symptoms)

# ======================
# PREDICT FUNCTION
# ======================
def predict(symptoms_list):
    text = " ".join(symptoms_list)

    vector = cv.transform([text])

    pred = model.predict(vector)[0]
    probs = model.predict_proba(vector)[0]

    top3 = np.argsort(probs)[-3:][::-1]

    results = []
    for i in top3:
        results.append((model.classes_[i], round(probs[i] * 100, 2)))

    return pred, results

# ======================
# BUTTON
# ======================
if st.button("Predict Disease"):
    if not selected:
        st.warning("Select symptoms first")
    else:
        pred, results = predict(selected)

        st.success(f"Predicted Disease: {pred}")

        st.subheader("Top Predictions")

        for d, p in results:
            st.write(f"👉 {d} : {p}% confidence")