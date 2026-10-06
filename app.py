import streamlit as st
import pandas as pd
import joblib

# Cache model artifacts to improve performance
@st.cache_resource
def load_artifacts():
    model = joblib.load('models/Logistic_Regression_heart.pkl')
    scaler = joblib.load('models/scaler.pkl')
    expected_columns = joblib.load('models/columns.pkl')
    return model, scaler, expected_columns

model, scaler, expected_columns = load_artifacts()

st.title("Heart Disease Prediction App By Akshit")
st.markdown("Provide the following details to predict the likelihood of heart disease:")

age = st.slider("Age", min_value=1, max_value=100, value=30)
sex = st.selectbox("Sex", options=["Male", "Female"])
chest_pain = st.selectbox("Chest Pain Type", ['ATA', 'NAP', 'TA', 'ASY'])
resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=120)
cholesterol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=["Yes", "No"])
resting_ecg = st.selectbox("Resting ECG", options=["Normal", "ST", "LVH"])
max_heart_rate = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=220, value=150)
exercise_angina = st.selectbox("Exercise Induced Angina", options=["Yes", "No"])
oldpeak = st.number_input("Oldpeak (ST depression induced by exercise)", min_value=0.0, max_value=10.0, value=1.0)
st_slope = st.selectbox("Slope of the peak exercise ST segment", options=["Up", "Flat", "Down"])

if st.button("Predict"):
    input_data = pd.DataFrame({
        'age': [age],
        'sex': [1 if sex == "Male" else 0],
        'cp': [chest_pain],
        'trestbps': [resting_bp],
        'chol': [cholesterol],
        'fbs': [1 if fasting_bs == "Yes" else 0],
        'restecg': [resting_ecg],
        'thalach': [max_heart_rate],
        'exang': [1 if exercise_angina == "Yes" else 0],
        'oldpeak': [oldpeak],
        'slope': [st_slope]
    })

    # If your training data used one-hot encoding, uncomment the line below:
    input_data = pd.get_dummies(input_data)

    # Ensure the input data has the same columns as the training data
    input_data = input_data.reindex(columns=expected_columns, fill_value=0)

    # Scale the input data
    scaled_input = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(scaled_input)

    if prediction[0] == 1:
        st.error("The model predicts that you are likely to have heart disease.")
    else:
        st.success("The model predicts that you are unlikely to have heart disease.")