import streamlit as st
import numpy as np
import joblib
from PIL import Image

# Configure the Streamlit app
st.set_page_config(page_title="Student Depression Predictor", page_icon="🧠", layout="wide")

# Load the trained model
@st.cache_resource
def load_model():
    return joblib.load("my_model.pkl33")  # Load the model

model = load_model()


# Sidebar with App Details
with st.sidebar:
    st.title("🧠 Student Depression Predictor")
    st.markdown("### Predict the likelihood of depression in students.")
    st.write("This app uses machine learning to assess potential depression risks.")
    st.markdown("---")
    st.markdown("👨‍💻 Developed by: **Brajesh Ahirwar**")
    st.markdown("🔗**Github:** [Brajesh Ahirwar](https://github.com/brajesh2306)")
    st.markdown("🔗 [LinkedIn](www.linkedin.com/in/brajesh-ahirwar-6269b728b)")
    st.write("🎯 Designed with Streamlit")
    st.markdown("---")
    st.markdown("✨ **Have fun exploring AI!**")



# Main Header with Stylish Font
st.markdown("<h1 style='text-align: center; color: #2E8B57;'>Student Depression Prediction App 🧠</h1>", unsafe_allow_html=True)
st.markdown(
    """
    <h3 style='text-align: center; color: #808080;'>🎉 Welcome to the Depression Prediction App! </h3>
    """, unsafe_allow_html=True)
# Add a background image (optional)
image = Image.open('Artificial Intelligence Application in Mental Health Research.jpg')  # Add your image path
st.image(image, use_container_width=True)

st.markdown(
    """<p style='text-align: center; font-size: 20px;'>Enter details about a student's lifestyle and habits to predict the likelihood of depression.</p>
    """, unsafe_allow_html=True)

# Add horizontal line for section separation
st.markdown("<hr>", unsafe_allow_html=True)

# Input Form for Student Details (with improved styling)
st.markdown("#### Please provide the student details below:")

# Using columns to organize the inputs
col1, col2 = st.columns(2)

with col1:
    gender = st.radio("Gender of the student", ["Male", "Female"])
    academic_pressure = st.slider("Level of Academic Pressure (1 to 5)", 1.0, 5.0, 3.0, step=0.1)
    sleep_duration = st.selectbox(
        "Average Sleep Duration per night", ["Less than 5 hours", "5-6 hours", "7-8 hours", "More than 8 hours"]
    )
    study_hours = st.number_input("Number of Study Hours per day", min_value=0, max_value=24, step=1)
    family_history = st.radio("Is there a family history of mental illness?", ["Yes", "No"])

with col2:
    age = st.number_input("Age of the student", min_value=1, max_value=120, step=1)
    study_satisfaction = st.slider("Study Satisfaction (1 to 5)", 1.0, 5.0, 3.0, step=0.1)
    dietary_habits = st.radio("Dietary Habits of the student", ["Healthy", "Unhealthy"])
    suicidal_thoughts = st.radio("Has the student ever had suicidal thoughts?", ["Yes", "No"])
    financial_stress = st.slider("Level of Financial Stress (1 to 5)", 1, 5, 3)
    study_pressure_hours = st.number_input("Study Pressure Hours per week", min_value=0, max_value=24, step=1)

# Map categorical inputs to numerical values
gender = 1 if gender == 'Male' else 0
dietary_habits = 1 if dietary_habits == 'Healthy' else 0
suicidal_thoughts = 1 if suicidal_thoughts == 'Yes' else 0
family_history = 1 if family_history == 'Yes' else 0

# Map sleep duration
sleep_mapping = {
    'Less than 5 hours': 4,
    '5-6 hours': 5.5,
    '7-8 hours': 7.5,
    'More than 8 hours': 9
}
sleep_duration = sleep_mapping.get(sleep_duration, 7.5)

# Create the feature array
input_data = np.array([[gender, age, academic_pressure, study_satisfaction, sleep_duration,
                        dietary_habits, suicidal_thoughts, study_hours, financial_stress, family_history,
                        study_pressure_hours]])

# Add a stylish button with hover effect
button_style = """
    <style>
        .stButton>button {
            background-color: #2E8B57;
            color: white;
            font-size: 16px;
            height: 3em;
            width: 100%;
            border-radius: 12px;
            transition: background-color 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #3CB371;
        }
    </style>
"""
st.markdown(button_style, unsafe_allow_html=True)

# Show a loading indicator when the user clicks "Predict"
with st.spinner("Predicting... Please wait."):
    # Predict button
    if st.button("Predict"):
        try:
            # Make a prediction
            prediction_proba = model.predict_proba(input_data)  # Get prediction probabilities

            # Get the probability of depression (class 1)
            depression_prob = prediction_proba[0][1]

            # Define the color-based output conditions
            if depression_prob < 0.2:
                st.markdown(f"<h3 style='color:green;'>The model predicts that this person is very unlikely to suffer from depression.</h3>", unsafe_allow_html=True)
            elif 0.2 <= depression_prob < 0.4:
                st.markdown(f"<h3 style='color:green;'>The model predicts that this person is unlikely to suffer from depression.</h3>", unsafe_allow_html=True)
            elif 0.4 <= depression_prob < 0.6:
                st.markdown(f"<h3 style='color:orange;'>The model predicts that this person may suffer from depression.</h3>", unsafe_allow_html=True)
            elif 0.6 <= depression_prob < 0.8:
                st.markdown(f"<h3 style='color:orange;'>The model predicts that this person is likely to suffer from depression.</h3>", unsafe_allow_html=True)
            else:
                st.markdown(f"<h3 style='color:red;'>The model predicts that this person is highly likely to suffer from depression.</h3>", unsafe_allow_html=True)

            # Display the exact probability for transparency
            st.write(f"Depression Probability: {depression_prob*100:.2f}%")

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
