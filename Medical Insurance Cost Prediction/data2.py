import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load the trained model
try:
    model = pickle.load(open("insurance_model.pkl", "rb"))
except FileNotFoundError:
    st.error("The model file 'insurance_model.pkl' was not found. Please ensure it is in the same directory as this script.")
    st.stop()

# Define the Streamlit app
def main():
    st.title("Medical Insurance Premium Prediction")
    st.write("Enter the details below to predict the insurance premium.")

    # Input fields
    age = st.slider("Age", min_value=18, max_value=100, value=30, step=1)
    gender = st.selectbox("Gender", ["Male", "Female"])
    bmi = st.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=50.0, value=25.0, step=0.1)
    children = st.slider("Number of Dependents", min_value=0, max_value=10, value=0, step=1)
    smoker = st.selectbox("Smoker", ["Yes", "No"])
    region = st.selectbox("Region", ["Northwest", "Northeast", "Southeast", "Southwest"])

    # Process inputs
    input_data = pd.DataFrame(
        {
            "Age": [age],
            "Gender": [gender],
            "BMI": [bmi],
            "Children": [children],
            "Smoker": [smoker],
            "Region": [region],
        }
    )

    # Prediction
    if st.button("Predict"):
        try:
            # Ensure the input data is compatible with the model
            prediction = model.predict(input_data)[0]
            st.success(f"The predicted insurance premium is: ${prediction:.2f}")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

    # Optional: Add additional insights or visualizations
    st.write("Optional: Add visualizations or additional insights about the dataset.")

if __name__ == "__main__":
    main()

