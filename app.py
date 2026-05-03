import streamlit as st
import numpy as np
import pickle

# ======================
# LOAD MODEL
# ======================
model = pickle.load(open("model.pkl", "rb"))
cv = pickle.load(open("vectorizer.pkl", "rb"))

# ======================
# UI CONFIG
# ======================
st.set_page_config(page_title="AI Health Assistant", page_icon="🧠")

st.title("🧠 AI Health Assistant")
st.write("Select symptoms to predict disease")

# ======================
# SYMPTOMS LIST
# ======================
symptoms = sorted(cv.get_feature_names_out())

selected_symptoms = st.multiselect(
    "Select Symptoms",
    symptoms
)

# ======================
# PREDICTION FUNCTION
# ======================
def predict(symptoms_list):
    text = " ".join(symptoms_list)
    vector = cv.transform([text]).toarray()

    prediction = model.predict(vector)[0]
    probs = model.predict_proba(vector)[0]

    # top 3 results
    top3_idx = np.argsort(probs)[-3:][::-1]

    results = []
    for i in top3_idx:
        results.append(
            (model.classes_[i], round(probs[i] * 100, 2))
        )

    return prediction, results

# ======================
# PREDICT BUTTON
# ======================
if st.button("Predict Disease"):
    if len(selected_symptoms) == 0:
        st.warning("Please select at least one symptom")
    else:
        pred, results = predict(selected_symptoms)

        st.success(f"Predicted Disease: {pred}")

        st.subheader("Top Predictions")

        for disease, prob in results:
            st.write(f"👉 {disease} : {prob}% confidence")