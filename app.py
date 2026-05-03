import streamlit as st
import numpy as np
import os
import pickle

# ======================
# LOAD MODEL SAFELY
# ======================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, "model.pkl"), "rb") as f:
    model = pickle.load(f)

with open(os.path.join(BASE_DIR, "vectorizer.pkl"), "rb") as f:
    cv = pickle.load(f)

# ======================
# UI
# ======================
st.set_page_config(page_title="AI Health Assistant", page_icon="🧠")

st.title("🧠 AI Health Assistant")

symptoms = sorted(cv.get_feature_names_out())
selected = st.multiselect("Select Symptoms", symptoms)

def predict(symptoms_list):
    text = " ".join(symptoms_list)
    vector = cv.transform([text])

    pred = model.predict(vector)[0]
    probs = model.predict_proba(vector)[0]

    top3 = np.argsort(probs)[-3:][::-1]

    results = []
    for i in top3:
        results.append((model.classes_[i], round(probs[i]*100, 2)))

    return pred, results

if st.button("Predict"):
    if not selected:
        st.warning("Please select symptoms")
    else:
        pred, results = predict(selected)

        st.success(f"Predicted Disease: {pred}")

        for d, p in results:
            st.write(f"👉 {d} : {p}%")