import streamlit as st
import pickle
import numpy as np
import pandas as pd
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Streamlit app
def main():
    st.title("Diabetes Prediction")
    st.write("Enter the values for the following features to predict diabetes:")

    # Create input fields for the features
    pregnancies = st.number_input('Pregnancies', min_value=0, max_value=20, value=0)
    glucose = st.number_input('Glucose', min_value=0, max_value=200, value=0)
    bp = st.number_input('Blood Pressure', min_value=0, max_value=200, value=0)
    skin_thickness = st.number_input('Skin Thickness', min_value=0, max_value=100, value=0)
    insulin = st.number_input('Insulin', min_value=0, max_value=1000, value=0)
    bmi = st.number_input('BMI', min_value=0, max_value=100, value=0)
    fun = st.number_input('Fun', min_value=0, max_value=100, value=0)
    age = st.number_input('Age', min_value=18, max_value=100, value=18)

    if st.button('Predict'):
        # Prepare input data
        input_data = np.array([pregnancies, glucose, bp, skin_thickness, insulin, bmi, fun, age]).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(input_data)

        # Display result
        if prediction == 1:
            st.write("**Diabetes Detected**")
        else:
            st.write("**No Diabetes Detected**")

if __name__ == '__main__':
    main()
