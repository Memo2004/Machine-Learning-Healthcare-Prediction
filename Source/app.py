import numpy as np
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
from pathlib import Path
import time

# Define the base directory (where your models are located)

image_path = Path(__file__).parent / "assets" / "image.webp"

import os


# ... (rest of your imports)

# Get the base directory of the current script
base_dir = os.path.dirname(os.path.abspath(__file__))

# Load models with error handling and path verification
try:
    # Heart Disease Model
    heart_disease_model_path = os.path.join(base_dir, "Saved Model", "heart_disease_model.sav")
    if os.path.exists(heart_disease_model_path):
        with open(heart_disease_model_path, 'rb') as f:
            heart_disease_model = pickle.load(f)
    else:
        raise FileNotFoundError(f"Heart Disease model file not found at: {heart_disease_model_path}")

    # Diabetes Model
    diabetes_model_path = os.path.join(base_dir, "Saved Model", "diabetes_disease_model.sav")
    if os.path.exists(diabetes_model_path):
        with open(diabetes_model_path, 'rb') as f:
            diabetes_disease_model = pickle.load(f)
    else:
        raise FileNotFoundError(f"Diabetes model file not found at: {diabetes_model_path}")

    # Hypertension Model
    hypertension_model_path = os.path.join(base_dir, "Saved Model", "Hypertension_risk_model.sav")
    if os.path.exists(hypertension_model_path):
        with open(hypertension_model_path, 'rb') as f:
            Hypertension_risk_model = pickle.load(f)
    else:
        raise FileNotFoundError(f"Hypertension model file not found at: {hypertension_model_path}")

except FileNotFoundError as e:
    st.error(f"Error loading model: {e}")
    st.stop()  # Stop the app execution if a model file is missing
except Exception as e:
    st.error(f"An error occurred while loading the models: {e}")
    st.stop()  # Stop the app execution in case of other errors


# Page Configuration
st.set_page_config(
    page_title="Healthcare Prediction App",
    page_icon="🏥",
    layout="wide"
)

# Sidebar Navigation
with st.sidebar:
    selected = option_menu(
        menu_title="Disease Prediction",
        options=["Home", "Heart Disease", "Diabetes", "Hypertension"],
        icons=["house", "heart", "droplet", "activity"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "5px", "background-color": "#f0f0f0"},
            "icon": {"color": "blue", "font-size": "20px"},
            "menu-title": {"font-size": "20px", "font-weight": "bold", "color": "black"},
            "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px", "color": "black"},
            "nav-link-selected": {"background-color": "#0288d1", "color": "black"},
        },
    )


# Home Page
if selected == "Home":
    st.title("Welcome to the Healthcare Prediction App! 🏥")
    st.write("""
    This app uses machine learning models to predict the risk of:
    - **Heart Disease**
    - **Diabetes**
    - **Hypertension**
    
    Select a disease from the sidebar to start the prediction process.
    """)
    st.image(image_path, use_container_width=True)# The directory of the current script

    
   

# Heart Disease Prediction
if selected == "Heart Disease":
    st.title("Heart Disease Prediction Using ML 🚀")

    with st.expander("Input Fields", expanded=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.number_input('Age', min_value=0, step=1)

        with col2:
            sex_options = {'Female': 0, 'Male': 1}
            sex = st.selectbox('Sex:', list(sex_options.keys()))
            sex = sex_options[sex]

        with col3:
            cp_options = {
                '0 : Typical Angina': 0,
                '1 : Atypical Angina': 1,
                '2 : Non-Anginal Pain': 2,
                '3 : Asymptomatic': 3
            }
            cp = st.selectbox('Chest Pain types:', list(cp_options.keys()))
            cp = cp_options[cp]

        with col1:
            trestbps = st.number_input('Resting Blood Pressure', min_value=0)

        with col2:
            chol = st.number_input('Serum Cholesterol in mg/dl', min_value=0)

        with col3:
            fbs_options = {'No': 0, 'Yes': 1}
            fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl:', list(fbs_options.keys()))
            fbs = fbs_options[fbs]

        with col1:
            restecg_options = {
                'Normal': 0,
                'ST-T wave abnormality': 1,
                'Left ventricular hypertrophy': 2
            }
            restecg = st.selectbox('Resting ECG results:', list(restecg_options.keys()))
            restecg = restecg_options[restecg]

        with col2:
            thalach = st.number_input('Maximum Heart Rate Achieved', min_value=0)

        with col3:
            exang_options = {'No': 0, 'Yes': 1}
            exang = st.selectbox('Exercise Induced Angina:', list(exang_options.keys()))
            exang = exang_options[exang]

        with col1:
            oldpeak = st.number_input('ST Depression Induced by Exercise', min_value=0.0)

        with col2:
            slope_options = {
                '0 : Downsloping': 0,
                '1 : Flat': 1,
                '2 : Upsloping': 2
            }
            slope = st.selectbox('Slope of the Peak ST Segment:', list(slope_options.keys()))
            slope = slope_options[slope]

        with col3:
            ca_options = {
                'No vessels': 0,
                'One vessel': 1,
                'Two vessels': 2,
                'Three vessels': 3,
                'Four vessels': 4
            }
            ca = st.selectbox('Major Vessels:', list(ca_options.keys()))
            ca = ca_options[ca]

        with col1:
            thal_options = {'0 : Normal': 0, '1 : Fixed Defect': 1, '2 : Reversible Defect': 2}
            thal = st.selectbox('Thal:', list(thal_options.keys()))
            thal = thal_options[thal]

    if st.button('Heart Disease Test Result'):
        user_input = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
        user_input = np.array(user_input).reshape(1, -1)

        with st.spinner("Predicting..."):
            time.sleep(2)  # Simulating prediction delay
            try:
                heart_prediction = heart_disease_model.predict(user_input)

                if heart_prediction[0] == 1:
                    st.markdown(
                        '<div style="background-color:tomato; color:white; padding:10px; border-radius:5px;">The person has heart disease.</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        '<div style="background-color:lightgreen; color:black; padding:10px; border-radius:5px;">The person does not have heart disease.</div>',
                        unsafe_allow_html=True
                    )
            except Exception as e:
                st.error(f"Error during prediction: {e}")

# Add similar code for Diabetes and Hypertension using tabs or collapsible sections
# Diabetes Prediction
if selected == "Diabetes":
    st.title("Diabetes Prediction Using ML 🚀")

    with st.expander("Input Fields", expanded=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            Pregnancies = st.number_input('Number of Pregnancies', min_value=0, step=1)

        with col2:
            Glucose = st.number_input('Glucose Level', min_value=0)

        with col3:
            BloodPressure = st.number_input('Blood Pressure value', min_value=0)

        with col1:
            SkinThickness = st.number_input('Skin Thickness value', min_value=0)

        with col2:
            Insulin = st.number_input('Insulin Level', min_value=0)

        with col3:
            BMI = st.number_input('BMI Values', min_value=0.0)

        with col1:
            DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function value', min_value=0.0)

        with col2:
            Age = st.number_input('Age of the Person', min_value=0)

    if st.button('Diabetes Test Result'):
        user_input = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
        user_input = np.array(user_input).reshape(1, -1)

        with st.spinner("Predicting..."):
            time.sleep(2)  # Simulating a delay
            try:
                diabetes_prediction = diabetes_disease_model.predict(user_input)

                # Displaying the prediction result
                if diabetes_prediction[0] == 1:
                    st.markdown(
                        '<div style="background-color:tomato; color:white; padding:10px; border-radius:5px;">The person is diabetic.</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        '<div style="background-color:lightgreen; color:black; padding:10px; border-radius:5px;">The person is not diabetic.</div>',
                        unsafe_allow_html=True
                    )
            except Exception as e:
                st.error(f"Error during prediction: {e}")




# Hypertension Prediction
if selected == "Hypertension":
    st.title("Hypertension Prediction Using ML 🚀")

    with st.expander("Input Fields", expanded=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            sex_options = {'Female': 0, 'Male': 1}
            sex = st.selectbox('Sex:', list(sex_options.keys()))
            sex = sex_options[sex]

        with col2:
            age = st.number_input('Age', min_value=0, step=1)

        with col3:
            smoker_options = {'No': 0, 'Yes': 1}
            currentSmoker = st.selectbox('Smoking Status:', list(smoker_options.keys()))
            currentSmoker = smoker_options[currentSmoker]

        with col1:
            cigsPerDay = st.number_input('Number of Cigarettes Smoked Per Day', min_value=0)

        with col2:
            bpm_options = {'No': 0, 'Yes': 1}
            BPMeds = st.selectbox('Blood Pressure Medication Usage:', list(bpm_options.keys()))
            BPMeds = bpm_options[BPMeds]

        with col3:
            diabetes_options = {'No': 0, 'Yes': 1}
            diabetes = st.selectbox('Diabetes Status:', list(diabetes_options.keys()))
            diabetes = diabetes_options[diabetes]

        with col1:
            totChol = st.number_input('Total Cholesterol Level', min_value=0)

        with col2:
            sysBP = st.number_input('Systolic Blood Pressure', min_value=0.0)

        with col3:
            diaBP = st.number_input('Diastolic Blood Pressure', min_value=0.0)

        with col1:
            BMI = st.number_input('Body Mass Index', min_value=0.0)

        with col2:
            heartRate = st.number_input('Heart Rate', min_value=0)

        with col3:
            glucose = st.number_input('Glucose Level', min_value=0)

    if st.button('Hypertension Test Result'):
        user_input = [sex, age, currentSmoker, cigsPerDay, BPMeds, diabetes, totChol, sysBP, diaBP, BMI, heartRate, glucose]
        user_input = np.array(user_input).reshape(1, -1)

        with st.spinner("Predicting..."):
            time.sleep(2)  # Simulating a delay
            try:
                hypertension_prediction = Hypertension_risk_model.predict(user_input)

                # Displaying the prediction result
                if hypertension_prediction[0] == 1:
                    st.markdown(
                        '<div style="background-color:tomato; color:white; padding:10px; border-radius:5px;">The person is hypertensive.</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        '<div style="background-color:lightgreen; color:black; padding:10px; border-radius:5px;">The person is not hypertensive.</div>',
                        unsafe_allow_html=True
                    )
            except Exception as e:
                st.error(f"Error during prediction: {e}")
