import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load the saved scaler and model
scaler = joblib.load('scaler.pkl')  # Make sure to save this earlier in your code
model = joblib.load('lasso_bodyfat_model.pkl')  # Ensure this is saved

# Function to categorize body fat percentage
def categorize_body_fat(bfp, gender):
    if gender == "Male":
        if bfp < 6:
            return "Essential Fat"
        elif 6 <= bfp < 14:
            return "Athlete"
        elif 14 <= bfp < 18:
            return "Fitness"
        elif 18 <= bfp < 25:
            return "Average"
        else:
            return "Obese"
    else:  # Female
        if bfp < 14:
            return "Essential Fat"
        elif 14 <= bfp < 21:
            return "Athlete"
        elif 21 <= bfp < 25:
            return "Fitness"
        elif 25 <= bfp < 32:
            return "Average"
        else:
            return "Obese"

# Streamlit app layout
st.title("Body Fat Percentage Predictor")

# Input fields for measurements
gender = st.selectbox("Select Gender:", ["Male", "Female"])
density = st.number_input("Body Density (gm/cm³)", min_value=1.0, max_value=1.5, value=1.0708)
age = st.number_input("Age (years)", min_value=1, max_value=120, value=25)
weight = st.number_input("Weight (lbs)", min_value=50, max_value=400, value=180)
height = st.number_input("Height (inches)", min_value=30, max_value=100, value=70)
neck = st.number_input("Neck Circumference (cm)", min_value=30, max_value=100, value=40)
chest = st.number_input("Chest Circumference (cm)", min_value=50, max_value=150, value=95)
abdomen = st.number_input("Abdomen Circumference (cm)", min_value=50, max_value=150, value=85)
hip = st.number_input("Hip Circumference (cm)", min_value=50, max_value=150, value=95)
thigh = st.number_input("Thigh Circumference (cm)", min_value=30, max_value=100, value=60)
knee = st.number_input("Knee Circumference (cm)", min_value=20, max_value=80, value=40)
ankle = st.number_input("Ankle Circumference (cm)", min_value=10, max_value=50, value=25)
biceps = st.number_input("Biceps Circumference (cm)", min_value=20, max_value=60, value=30)
forearm = st.number_input("Forearm Circumference (cm)", min_value=10, max_value=50, value=25)
wrist = st.number_input("Wrist Circumference (cm)", min_value=10, max_value=50, value=20)

# When the user clicks the button, make predictions
if st.button("Predict Body Fat Percentage"):
    # Prepare the input data
    new_data = np.array([[density, age, weight, height, neck, chest, abdomen, hip, thigh, knee, ankle, biceps, forearm, wrist]])
    new_data_scaled = scaler.transform(new_data)

    # Predict body fat percentage
    predicted_body_fat = model.predict(new_data_scaled)[0]
    category = categorize_body_fat(predicted_body_fat, gender)

    # Show the result
    st.write(f"Predicted Body Fat Percentage: {predicted_body_fat:.2f}%")
    st.write(f"Category: {category}")
