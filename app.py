import streamlit as st
import joblib
import pandas as pd
import sklearn

# Load the trained model
model = joblib.load('placement_model.pkl')

st.title('Student Placement Prediction App')
st.write('Enter the student details to predict their placement status.')

# Input fields for features
CGPA = st.slider('CGPA', 6.5, 9.1, 7.5, 0.1)
Internships = st.selectbox('Number of Internships', [0, 1, 2])
Projects = st.selectbox('Number of Projects', [0, 1, 2, 3])
Workshops_Certifications = st.selectbox('Number of Workshops/Certifications', [0, 1, 2, 3])
AptitudeTestScore = st.slider('Aptitude Test Score', 60, 90, 75)
SoftSkillsRating = st.slider('Soft Skills Rating', 3.0, 4.8, 4.0, 0.1)
ExtracurricularActivities = st.selectbox('Extracurricular Activities', ['No', 'Yes'])
PlacementTraining = st.selectbox('Placement Training', ['No', 'Yes'])
SSC_Marks = st.slider('SSC Marks', 55, 90, 70)
HSC_Marks = st.slider('HSC Marks', 57, 88, 75)

# Map categorical inputs to numerical values (matching LabelEncoder output)
extra_curricular_map = {'No': 0, 'Yes': 1}
placement_training_map = {'No': 0, 'Yes': 1}

extra_curricular_val = extra_curricular_map[ExtracurricularActivities]
placement_training_val = placement_training_map[PlacementTraining]

# Create a DataFrame for prediction
input_data = pd.DataFrame([[CGPA, Internships, Projects, Workshops_Certifications, AptitudeTestScore, \
                            SoftSkillsRating, extra_curricular_val, placement_training_val, SSC_Marks, HSC_Marks]],
                            columns=['CGPA', 'Internships', 'Projects', 'Workshops/Certifications', 'AptitudeTestScore',
                                     'SoftSkillsRating', 'ExtracurricularActivities', 'PlacementTraining', 'SSC_Marks', 'HSC_Marks'])

if st.button('Predict Placement'):
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)

    if prediction[0] == 1:
        st.success(f"The student is likely to be **Placed** with a probability of {prediction_proba[0][1]:.2f}.")
    else:
        st.error(f"The student is likely to be **Not Placed** with a probability of {prediction_proba[0][0]:.2f}.")
