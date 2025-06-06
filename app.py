#Necessary Imports
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

#Configuring tha pp
st.set_page_config(
    page_title="Rock vs. Mine Identifier",
    page_icon="🌊",
    layout="centered",
    initial_sidebar_state="collapsed"
)

modelPath = os.path.join(os.path.dirname(__file__), "model.pkl")

model = None

try:
    with open(modelPath, 'rb') as file:
        model = pickle.load(file)

    st.sidebar.success("Model loaded successfully")

except FileNotFoundError:
    st.error(f"Error: Model file '{MODEL_FILE_NAME}' not found at '{os.path.abspath(os.path.dirname(__file__))}'. Please ensure your model is trained and saved in the same directory as this Streamlit app.")
    st.stop()
    
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

#App Description
st.title("🌊 Rock vs. Mine Identifier 💣🪨")
st.markdown("""
    This application uses a machine learning model to predict whether a given set of sonar readings
    corresponds to a **Rock** or a **Mine**.
""")

st.subheader("Input Sonar Readings")
st.write("Please enter exactly 60 comma-separated float values, representing the sonar signal features.")

input_string = st.text_area(
    "Sonar Readings (comma-separated values)",
    height=200,
    placeholder="example: 0.0015,0.0186,0.0289,0.0195,0.0515,0.0817,0.1005,0.0124,0.1168,0.1476,0.2118,0.2575,0.2354,0.1334,0.0092,0.1951,0.3685,0.4646,0.5418,0.6260,0.7420,0.8257,0.8609,0.8400,0.8949,0.9945,1.0000,0.9649,0.8747,0.6257,0.2184,0.2945,0.3645,0.5012,0.7843,0.9361,0.8195,0.6207,0.4513,0.3004,0.2674,0.2241,0.3141,0.3693,0.2986,0.2226,0.0849,0.0359,0.0289,0.0122,0.0045,0.0108,0.0075,0.0089,0.0036,0.0029,0.0013,0.0010,0.0032,0.0047"
)

#Prediction button
if st.button("Predict Type"):
    if input_string and model:
        try:
            raw_values = input_string.strip().split(',')
            # Check if exactly 60 values are provided
            if len(raw_values) != 60:
                st.warning(f"Incorrect number of values. Expected 60, but got {len(raw_values)}. Please check your input.")
            else:
                input_data = [float(val.strip()) for val in raw_values]

                # Convert input data to numpy array and reshape for the model
                input_data_as_numpy_array = np.asarray(input_data)
                input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

                # Make prediction
                prediction = model.predict(input_data_reshaped)

                st.subheader("Prediction Result:")
                if prediction[0] == "R":
                    st.success("The sonar readings indicate: **ROCK** 🪨")
                elif prediction[0] == "M":
                    st.error("The sonar readings indicate: **MINE** 💣")
                else:
                    st.info("Unable to classify with high confidence.")

        except ValueError:
            st.error("Invalid input. Please ensure all values are valid numbers (floats).")
        except Exception as e:
            st.error(f"An unexpected error occurred during prediction: {e}")
    elif not input_string:
        st.warning("Please enter the sonar readings to make a prediction.")
    else:
        st.info("Model not ready. Please check server logs for errors.")

# --- Model Information and Performance (from your notebook) ---
st.markdown("---")
st.subheader("Model Information")
st.write("This application utilizes a Logistic Regression model for classification.")
st.write("The model was trained on a dataset containing 208 samples, each with 60 distinct features.")
st.write("The target variable differentiates between 'R' (Rock) and 'M' (Mine).")

st.markdown("**Model Performance:**")
st.write(f"- Accuracy on training data: **0.834**") #
st.write(f"- Accuracy on test data: **0.762**") #

st.markdown("---")
st.caption("Developed with Streamlit by Amandeep Singh [2025]")