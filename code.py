import streamlit as st
import pandas as pd
import pickle  # Changed from joblib to pickle
import matplotlib.pyplot as plt
import numpy as np

# Set up the app
st.title("🏠 Housing Value Predictor")
st.write("This app predicts median house values based on median income using a pre-trained linear regression model.")

# Load the trained model using pickle
try:
    with open('housing_model.pkl', 'rb') as f:  # Using pickle load
        model = pickle.load(f)
except FileNotFoundError:
    st.error("Model file not found. Please make sure 'housing_model.pkl' exists.")
    st.error("Note: The model file must be created using pickle (not joblib)")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()

# Sidebar for user input
st.sidebar.header("Input Parameters")
income = st.sidebar.slider("Median Income",
                          min_value=0.0,
                          max_value=15.0,
                          value=3.0,
                          step=0.5)

# Make prediction
prediction = model.predict([[income]])[0]

# Display prediction
st.subheader("Prediction")
st.write(f"Predicted Median House Value: ${prediction:,.2f}")

# Visualization section
st.subheader("Model Visualization")

# Generate range of values for the plot
income_range = np.linspace(0, 15, 100).reshape(-1, 1)
predictions_range = model.predict(income_range)

# Create the plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(income_range, predictions_range, color='red', label="Regression line")
ax.scatter([income], [prediction], color='green', s=100, label="Your prediction")
ax.set_xlabel("Median Income")
ax.set_ylabel("Median House Value")
ax.set_title("House Value Prediction")
ax.legend()
ax.grid(True)

# Display the plot
st.pyplot(fig)

# Model info section
st.subheader("Model Information")
st.write("This model was trained on the California housing dataset.")
st.write("Model type: Linear Regression (saved with pickle)")