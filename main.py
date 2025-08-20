import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# import function
from EDA import univariate_analysis, univariate_numeric
from evaluasi import processing, prediction
from run_model import main

df = pd.read_csv('https://raw.githubusercontent.com/mfznakbr/Eksperimen_SML_Muhammad-Fauzani-Akbar/main/personality_datasert.csv')
print(df.head())

# COLLUMNS IN DATASET
cat =['Stage_fear', 'Drained_after_socializing']
target = ['Personality']
num = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 'Friends_circle_size', 'Post_frequency']

# SIDE BAR FOR PREDICT AND FILTER EDA
with st.sidebar:
    # adding logo
    # st.image('blabla')

    # input for user
    st.title('Input Data')

    # Category input
    Stage_fear = st.selectbox('Stage_fear', df['Stage_fear'].unique())
    Drained_after_socializing = st.selectbox('Perasaan setelah bersosialisasi', df['Drained_after_socializing'].unique())

    # Numeric Input 
    time_spent_alone = st.number_input("Waktu Sendirian", min_value=0.0, value=9.0)
    social_event_attendance = st.number_input("Kehadiran Acara Sosial", min_value=0.0, value=1.0)
    going_outside = st.number_input("Frekuensi Keluar Rumah", min_value=0.0, value=3.0)
    friends_circle_size = st.number_input("Ukuran Lingkaran Teman", min_value=0.0, value=2.0)
    post_frequency = st.number_input("Frekuensi Posting", min_value=0.0, value=1.0)

    # masukan nilai inputan ke dict
    input_data = {
        'Stage_fear': Stage_fear,
        'Drained_after_socializing': Drained_after_socializing,
        'Time_spent_Alone': time_spent_alone,
        'Social_event_attendance': social_event_attendance,
        'Going_outside': going_outside,
        'Friends_circle_size': friends_circle_size,
        'Post_frequency': post_frequency
    }

    # convert dict to dataframe
    input_df = pd.DataFrame([input_data])

    if st.button("Predict"):
        result = main(input_df)
        if result == 1:  
            st.success("Prediksi: Introvert")
        else:
            st.success("Prediksi: Ekstrovert")

 # Univariate analysis for Stage_fear

# EXPLORATORY DATA ANALYSIS

st.title('Personality Predict')
st.write('This is a simple app to predict personality using machine learning.')
 
st.subheader("Customer Demographics")
# Univariate analysis for Stage_fear
col1, col2 = st.columns(2)
col3, col4= st.columns(2)
    
# Call the univariate_analysis function for both features
univariate_analysis(0, col1) 
univariate_analysis(1, col2) # Index 0 for Stage_fear
univariate_numeric(0, col3) # Index 0 for Time_spent_Alone
univariate_numeric(1, col4) # Index 1 for Social_event_attendance
