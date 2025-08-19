import streamlit as st
import pandas as pd

df = pd.read_csv('https://raw.githubusercontent.com/mfznakbr/Eksperimen_SML_Muhammad-Fauzani-Akbar/main/personality_datasert.csv')
print(df.head())

# COLLUMNS IN DATASET
cat =['Stage_fear', 'Drained_after_socializing']
target = ['Personality']
num = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 'Friends_circle_size', 'Post_frequency']

with st.sidebar:
    # adding logo
    st.image('blabla')

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

# EXPLORATORY DATA ANALYSIS
st.title('Exploratory Data Analysis')
def univariate_analysis(IndexColumn):
    feature = cat[IndexColumn]
    count = df[feature].value_counts()
    persent = 100 * df[feature].value_counts(normalize=True)
    count_persent = pd.DataFrame({'count': count, 'persent': persent})
    print(count_persent)    
    count.plot(kind='bar', title=f"Inilah jumlah {feature} lek 🕴️")


st.title('Personality Predict')
st.write('This is a simple app to predict personality using machine learning.')