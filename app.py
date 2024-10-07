import streamlit as st
import joblib
import numpy as np

# Load the model and scaler
rf_model = joblib.load('random_forest_model.pkl')
scaler = joblib.load('scaler.pkl')

# Function to predict body fat percentage
def predict_body_fat(data):
    data_scaled = scaler.transform([data])
    prediction = rf_model.predict(data_scaled)
    return prediction[0]

# Streamlit app
st.title("Body Fat Percentage Predictor")
st.write("This app helps you estimate your body fat percentage based on your measurements.")

# Input fields with descriptions
density = st.number_input("Density (gm/cm³)", min_value=0.0, help="Body density measured via underwater weighing. Consult a professional for accurate measurement.")
age = st.number_input("Age (years)", min_value=0, help="Your age in years.")
weight = st.number_input("Weight (lbs)", min_value=0, help="Your weight in pounds. Can be measured using a scale.")
height = st.number_input("Height (inches)", min_value=0, help="Your height in inches. Measure using a measuring tape.")
neck = st.number_input("Neck Circumference (cm)", min_value=0, help="Measure around the neck at the narrowest point.")
chest = st.number_input("Chest Circumference (cm)", min_value=0, help="Measure around the chest at the nipple level.")
abdominal = st.number_input("Abdomen Circumference (cm)", min_value=0, help="Measure at the level of the iliac crest.")
hip = st.number_input("Hip Circumference (cm)", min_value=0, help="Measure around the widest part of the hips.")
thigh = st.number_input("Thigh Circumference (cm)", min_value=0, help="Measure around the largest part of the thigh.")
knee = st.number_input("Knee Circumference (cm)", min_value=0, help="Measure around the knee while standing.")
ankle = st.number_input("Ankle Circumference (cm)", min_value=0, help="Measure around the narrowest part of the ankle.")
biceps = st.number_input("Biceps Circumference (cm)", min_value=0, help="Measure around the largest part of the bicep.")
forearm = st.number_input("Forearm Circumference (cm)", min_value=0, help="Measure around the largest part of the forearm.")
wrist = st.number_input("Wrist Circumference (cm)", min_value=0, help="Measure around the wrist.")

# Button to predict
if st.button("Predict Body Fat Percentage"):
    input_data = [density, age, weight, height, neck, chest, abdominal, hip, thigh, knee, ankle, biceps, forearm, wrist]
    predicted_body_fat = predict_body_fat(input_data)
    st.success(f'Predicted Body Fat Percentage: {predicted_body_fat:.2f}%')

# Optionally add more sections or features here...
