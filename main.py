import streamlit as st
import pandas as pd

df = pd.read_csv('https://raw.githubusercontent.com/mfznakbr/Eksperimen_SML_Muhammad-Fauzani-Akbar/main/personality_datasert.csv')
print(df.head())

st.title('Personality Predict')
st.write('This is a simple app to predict personality using machine learning.')
