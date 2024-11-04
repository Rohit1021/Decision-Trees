import streamlit as st
import pandas as pd
import tensorflow as tf
import tensorflow_decision_forests as tfdf
import joblib

# Load models
model_rd = tf.saved_model.load("model_rd")  # Load Random Forest model
model_gbdt = tf.saved_model.load("model_gbdt")  # Load GBDT model

# Load the scaler
scaler = joblib.load("/Users/rohit/Desktop/Decision-Trees/models/scaler_decision_trees.joblib")  # Ensure this path is correct

# Streamlit app layout
st.title("In-Hospital Mortality Prediction")
st.write("Predict if a patient is likely to experience an in-hospital anomaly (mortality) based on medical data.")

# Input fields for the features
st.header("Input Patient Data")
RESP = st.number_input("Respiratory Rate (RESP)", value=20.0)
BP_S = st.number_input("Systolic Blood Pressure (BP-S)", value=120.0)
BP_D = st.number_input("Diastolic Blood Pressure (BP-D)", value=80.0)
SpO2 = st.number_input("Oxygen Saturation (SpO2)", value=98.0)
HR = st.number_input("Heart Rate (HR)", value=75)
PULSE = st.number_input("Pulse Rate (PULSE)", value=75)

# Create a DataFrame for the input features
input_data = pd.DataFrame({
    "RESP": [RESP],
    "BP-S": [BP_S],
    "BP-D": [BP_D],
    "SpO2": [SpO2],
    "HR": [HR],
    "PULSE": [PULSE]
})

# Define function for making predictions
def predict_anomaly(data, model, scaler):
    # Scale the input data
    scaled_input = scaler.transform(data)

    # Create a dictionary from the scaled input for the model
    input_dict = {
        "RESP": tf.convert_to_tensor(scaled_input[0, 0:1], dtype=tf.float32),
        "BP-S": tf.convert_to_tensor(scaled_input[0, 1:2], dtype=tf.float32),
        "BP-D": tf.convert_to_tensor(scaled_input[0, 2:3], dtype=tf.float32),
        "SpO2": tf.convert_to_tensor(scaled_input[0, 3:4], dtype=tf.float32),
        "HR": tf.convert_to_tensor(scaled_input[0, 4:5], dtype=tf.float32),
        "PULSE": tf.convert_to_tensor(scaled_input[0, 5:6], dtype=tf.float32)
    }

    # Make predictions
    predictions = model(input_dict)

    # Assuming a threshold of 0.5 for binary classification
    return "Anomaly Detected" if predictions[0].numpy() >= 0.5 else "No Anomaly Detected"

# Choose patient data
patient = st.selectbox("Select Patient Data:", ["Patient 1", "Patient 2"])

# Prediction button
if st.button("Predict Anomaly"):
    model = model_rd if patient == "Patient 1" else model_gbdt
    result = predict_anomaly(input_data, model, scaler)
    st.write("### Prediction Result:")
    st.write(result)
