import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the pre-trained model (model.pkl)
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Streamlit app
def main():
    st.title("Diabetes Prediction")
    st.write("Enter the values for the following features to predict diabetes:")

    # Create input fields with typical default values for normal ranges
    pregnancies = st.number_input('Number of Pregnancies', min_value=0, max_value=20, value=3)
    glucose = st.number_input('Plasma Glucose Concentration', min_value=0, max_value=200, value=100)
    blood_pressure = st.number_input('Diastolic Blood Pressure (mm Hg)', min_value=0, max_value=200, value=70)
    skin_thickness = st.number_input('Triceps Skinfold Thickness (mm)', min_value=0, max_value=100, value=20)
    insulin = st.number_input('2-Hour Serum Insulin (mu U/ml)', min_value=0, max_value=1000, value=80)
    bmi = st.number_input('Body Mass Index (BMI)', min_value=0, max_value=100, value=25)
    diabetes_pedigree_function = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=2.5, value=0.5)
    age = st.number_input('Age (Years)', min_value=18, max_value=100, value=30)

    # Prediction when button is clicked
    if st.button('Predict'):
        # Prepare input data
        input_data = np.array([pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(input_data)

        if prediction == 1:
            st.write("**Diabetes Detected**")
        else:
            st.write("**No Diabetes Detected**")
    
    st.markdown(
    """
    <style>
    .button-container {
        display: flex;
        justify-content: center;
        position: fixed;
        bottom: 20px;
        width: 100%;
    }
    .button-container a {
        font-size: 18px;
        padding: 10px 20px;
        background-color: #4CAF50;
        color: white;
        text-align: center;
        border-radius: 5px;
        text-decoration: none;
    }
    .button-container a:hover {
        background-color: #45a049;
    }
    </style>
    <div class="button-container">
        <a href="https://ashokumar.in" target="_blank">About Me</a>
    </div>
    """,
    unsafe_allow_html=True,
)
if __name__ == '__main__':
    main()
