import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from datetime import datetime
from sklearn.preprocessing import OneHotEncoder

# Load the trained LSTM model
lstm_model = tf.keras.models.load_model('stacked_lstm_model.keras')

# Load the trained Random Forest Regressor model
rf_model = joblib.load('random_forest_regressor.pkl')

# Load the scaler used for normalization
scaler = joblib.load('scaler.pkl')  # Ensure you have saved the scaler after training

# Load the original training data to fit the encoder
# This should be the same data used to train both models
original_training_data = pd.read_csv('/content/nurse_hourly_pay_rates.csv')  # Replace with your actual data source

# Fit the OneHotEncoder on the original training data
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoder.fit(original_training_data[['Job Title', 'Location', 'Hospital Name']])

# Streamlit app title
st.title("Hourly Rate Prediction for Nurses")

# User inputs
job_title = st.text_input("Job Title")
location = st.text_input("Location")
hospital = st.text_input("Hospital")
contract_start_date = st.date_input("Contract Start Date", value=datetime.today())
contract_end_date = st.date_input("Contract End Date", value=datetime.today())

# Function to prepare input for LSTM prediction
def prepare_lstm_input(job_title, location, hospital, start_date, end_date):
    # Create a dummy input for LSTM
    dummy_sequence = np.zeros((10, 1))  # Create a dummy sequence of zeros
    input_scaled = scaler.transform(dummy_sequence)  # Scale the input
    input_scaled = input_scaled.reshape((1, 10, 1))  # Reshape for LSTM
    return input_scaled

# Function to prepare input for Random Forest prediction
def prepare_rf_input(job_title, location, hospital, start_date, end_date):
    # Create a DataFrame for input
     input_data = pd.DataFrame({
        'Job Title': [job_title],
        'Location': [location],
        'Hospital Name': [hospital] 
      })
     training_features = rf_model.feature_names_in_  # Assumes you used 'rf_model.fit(X_train, y_train)'
     input_encoded = pd.get_dummies(input_data, columns=['Job Title', 'Location', 'Hospital Name'], 
                                    drop_first=True)  # Align with training preprocessing

    # Align features: Add missing columns, remove extra columns
     missing_cols = set(training_features) - set(input_encoded.columns)
     for col in missing_cols:
        input_encoded[col] = 0  # Add missing columns with 0 values

     extra_cols = set(input_encoded.columns) - set(training_features)
     input_encoded = input_encoded.drop(columns=extra_cols)  # Remove extra columns

     input_encoded = input_encoded[training_features]  # Ensure correct order of features

     return input_encoded

# Prediction button
if st.button("Predict Hourly Rate"):
    # Prepare the input for LSTM
    lstm_input_data = prepare_lstm_input(job_title, location, hospital, contract_start_date, contract_end_date)
    
    # Make prediction using LSTM
    lstm_predicted_rate = lstm_model.predict(lstm_input_data)
    lstm_predicted_rate_inverse = scaler.inverse_transform(lstm_predicted_rate)
    
    # Prepare the input for Random Forest
    rf_input_data = prepare_rf_input(job_title, location, hospital, contract_start_date, contract_end_date)
    
    # Make prediction using Random Forest
    rf_predicted_rate = rf_model.predict(rf_input_data)
    
    # Display the results
    st.success(f"The predicted hourly rate using Random Forest Regressor is: ${rf_predicted_rate[0]:.2f}")
    st.success(f"The predicted hourly rate using Stacked LSTM is: ${lstm_predicted_rate[0][0]:.2f}")
