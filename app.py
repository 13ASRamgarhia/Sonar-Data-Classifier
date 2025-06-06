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
st.title("🌊 Sonar Reading Classifier")
st.markdown("""
    This application uses a machine learning model to predict whether a given set of sonar readings
    corresponds to a **Rock** 🪨 or a **Mine** 💣. The application uses a set of 60-Dimensional
    Sonar Reading, in which each value represents the energy in a specific frequency band.
    The reading is used in Submarines to distinguish between objects like mines and rocks.
""")

st.subheader("Input Sonar Readings")
st.write("Please enter exactly 60 comma-separated float values, representing the sonar signal features.")

input_string = st.text_area(
    "Sonar Readings (You can find examples at the bottom)",
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

st.markdown("**60-Dimensional Sonar Vector Examples**")
st.write(f"Mine - 0.0131,0.0201,0.0045,0.0217,0.0230,0.0481,0.0742,0.0333,0.1369,0.2079,0.2295,0.1990,0.1184,0.1891,0.2949,0.5343,0.6850,0.7923,0.8220,0.7290,0.7352,0.7918,0.8057,0.4898,0.1934,0.2924,0.6255,0.8546,0.8966,0.7821,0.5168,0.4840,0.4038,0.3411,0.2849,0.2353,0.2699,0.4442,0.4323,0.3314,0.1195,0.1669,0.3702,0.3072,0.0945,0.1545,0.1394,0.0772,0.0615,0.0230,0.0111,0.0168,0.0086,0.0045,0.0062,0.0065,0.0030,0.0066,0.0029,0.0053") #
st.write(f"Mine - 0.0260,0.0363,0.0136,0.0272,0.0214,0.0338,0.0655,0.1400,0.1843,0.2354,0.2720,0.2442,0.1665,0.0336,0.1302,0.1708,0.2177,0.3175,0.3714,0.4552,0.5700,0.7397,0.8062,0.8837,0.9432,1.0000,0.9375,0.7603,0.7123,0.8358,0.7622,0.4567,0.1715,0.1549,0.1641,0.1869,0.2655,0.1713,0.0959,0.0768,0.0847,0.2076,0.2505,0.1862,0.1439,0.1470,0.0991,0.0041,0.0154,0.0116,0.0181,0.0146,0.0129,0.0047,0.0039,0.0061,0.0040,0.0036,0.0061,0.0115") #
st.write(f"Rock - 0.0762,0.0666,0.0481,0.0394,0.0590,0.0649,0.1209,0.2467,0.3564,0.4459,0.4152,0.3952,0.4256,0.4135,0.4528,0.5326,0.7306,0.6193,0.2032,0.4636,0.4148,0.4292,0.5730,0.5399,0.3161,0.2285,0.6995,1.0000,0.7262,0.4724,0.5103,0.5459,0.2881,0.0981,0.1951,0.4181,0.4604,0.3217,0.2828,0.2430,0.1979,0.2444,0.1847,0.0841,0.0692,0.0528,0.0357,0.0085,0.0230,0.0046,0.0156,0.0031,0.0054,0.0105,0.0110,0.0015,0.0072,0.0048,0.0107,0.0094") #
st.write(f"Rock - 0.0453,0.0523,0.0843,0.0689,0.1183,0.2583,0.2156,0.3481,0.3337,0.2872,0.4918,0.6552,0.6919,0.7797,0.7464,0.9444,1.0000,0.8874,0.8024,0.7818,0.5212,0.4052,0.3957,0.3914,0.3250,0.3200,0.3271,0.2767,0.4423,0.2028,0.3788,0.2947,0.1984,0.2341,0.1306,0.4182,0.3835,0.1057,0.1840,0.1970,0.1674,0.0583,0.1401,0.1628,0.0621,0.0203,0.0530,0.0742,0.0409,0.0061,0.0125,0.0084,0.0089,0.0048,0.0094,0.0191,0.0140,0.0049,0.0052,0.0044") #
st.write(f"Rock - 0.0317,0.0956,0.1321,0.1408,0.1674,0.1710,0.0731,0.1401,0.2083,0.3513,0.1786,0.0658,0.0513,0.3752,0.5419,0.5440,0.5150,0.4262,0.2024,0.4233,0.7723,0.9735,0.9390,0.5559,0.5268,0.6826,0.5713,0.5429,0.2177,0.2149,0.5811,0.6323,0.2965,0.1873,0.2969,0.5163,0.6153,0.4283,0.5479,0.6133,0.5017,0.2377,0.1957,0.1749,0.1304,0.0597,0.1124,0.1047,0.0507,0.0159,0.0195,0.0201,0.0248,0.0131,0.0070,0.0138,0.0092,0.0143,0.0036,0.0103") #
st.write(f"Mine - 0.0530,0.0885,0.1997,0.2604,0.3225,0.2247,0.0617,0.2287,0.0950,0.0740,0.1610,0.2226,0.2703,0.3365,0.4266,0.4144,0.5655,0.6921,0.8547,0.9234,0.9171,1.0000,0.9532,0.9101,0.8337,0.7053,0.6534,0.4483,0.2460,0.2020,0.1446,0.0994,0.1510,0.2392,0.4434,0.5023,0.4441,0.4571,0.3927,0.2900,0.3408,0.4990,0.3632,0.1387,0.1800,0.1299,0.0523,0.0817,0.0469,0.0114,0.0299,0.0244,0.0199,0.0257,0.0082,0.0151,0.0171,0.0146,0.0134,0.0056") #

st.markdown("---")
st.caption("Developed with Streamlit by Amandeep Singh [2025]")
