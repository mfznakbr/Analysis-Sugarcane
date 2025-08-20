import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st


df = pd.read_csv('https://raw.githubusercontent.com/mfznakbr/Eksperimen_SML_Muhammad-Fauzani-Akbar/main/personality_datasert.csv')
print(df.head())

# COLLUMNS IN DATASET
cat =['Stage_fear', 'Drained_after_socializing']
target = ['Personality']
num = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 'Friends_circle_size', 'Post_frequency']

# COLORS PALETTES
colors = ["#90CAF9", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3"]

feature = cat[1]
datak = df[feature]
print(datak)


def univariate_analysis(IndexColumn, col):
    feature = cat[IndexColumn]
    count = df[feature].value_counts()
    percent = 100 * df[feature].value_counts(normalize=True)
    count_percent = pd.DataFrame({'count': count, 'percent': percent})
    count_percent = count_percent.reset_index()

    with col:
        fig, ax = plt.subplots(figsize=(20, 10))
    
        sns.barplot(
            y="count", 
            x=feature,
            data=df[feature].value_counts().reset_index(),
            palette=colors,
            ax=ax
        )
        ax.set_title(f"Number of People by {feature}", loc="center", fontsize=50)
        ax.set_ylabel(None)
        ax.set_xlabel(None)
        ax.tick_params(axis='x', labelsize=35)
        ax.tick_params(axis='y', labelsize=30)
        st.pyplot(fig)

def univariate_numeric(IndexCol, col):
    with col:
        feature = num[IndexCol]
        fig, ax = plt.subplots(figsize=(20, 10))
        ax.hist(df[feature], bins=15)
        ax.set_title(f"Distribution of {feature}", loc="center", fontsize=30)
        st.pyplot(fig)