import streamlit as st
import pandas as pd
import requests

st.title("Medical Cost Prediction")
st.markdown("Enter data:")

age = st.slider("Age", 18, 100, 35)
bmi = st.number_input("BMI", value=25.0)
children = st.slider("Children", 0, 15, 0)

smoker = st.selectbox("Smoker", ['Yes', 'No'])
sex = st.selectbox("Sex", ['Male', 'Female'])
region = st.selectbox('Region', ['Southwest', 'Southeast', 'Northwest', 'Northeast'])

input_data = {
    "medical_features": {
        "age": age,
        "bmi": bmi,
        "children": children,
        "smoker": smoker,
        "sex": sex,
        "region": region
    }
}

if st.button('Predict Medical Cost'):
    response = requests.post('http://localhost:3000/predict', json=input_data)
    if response.status_code == 200:
        result = response.json()
        predicted = result.get('charges')
        st.success(f"Your medical costs: {round(predicted, 2)}$")
    else:
        st.error('Server Error')