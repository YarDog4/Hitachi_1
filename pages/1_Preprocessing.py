from src.preprocessing.load_label import load_labeled_dataset
from src.preprocessing.cleaning_data import clean_text
from src.visualization.avg_word_count import average_word_count
from src.visualization.word_frequency import plotting_word_frequencies
from dotenv import load_dotenv
import streamlit as st
import os

load_dotenv()

st.set_page_config(page_title="Data Preprocessing", layout="wide", page_icon="")

st.title("🧼 Text Cleaning & Word Analysis")
st.sidebar.header("Preprocessing and Data flow")

st.write(
    """
    This page will show you the steps we took to analyze the data before Tokenizing and Vectorizing.
    Everything shown on this page will show preprocessing techniques we used to reach the main funcitonality of our project.
    """
)

data_directory = st.sidebar.text_input(
    "You an choose your choice of Data Directory",
    os.getenv(r"RELATIVE_PATH")
)

@st.cache_data()
def load_data(data_directory: str):
    return load_labeled_dataset(data_directory)

if data_directory:
    df, category_index = load_labeled_dataset(os.getenv(r"RELATIVE_PATH"))
    st.write(f"Loaded **{len(df)}** documents from **{len(category_index)}** categories")
    
    fig, avg_wc = average_word_count(df, text_column='text', category_map=category_index)
    st.pyplot(fig)

    st.subheader("Average Word Count Data")
    st.dataframe(avg_wc)

#Comparing Original vs. Cleaned Text for a given document
st.header("Before vs. After: Text Cleaning Comparison")

doc_index = st.sidebar.number_input(
    "Enter Document Index", 
    min_value=0,
    max_value= len(df)-1,
    value=0,
    step=1
    )

@st.cache_data()
def clean():
    return clean_text(df.copy(), text_column="text", lemmatize=False)

clean_dataframe = clean()

#Texts
original_text = df.iloc[doc_index]["text"]
cleaned_text = clean_dataframe.iloc[doc_index]["text_clean"]

#Displaying
st.subheader("Original Text")
st.text_area("Original Text", value=original_text, height = 350)
st.subheader("Cleaned Text")
st.text_area("Cleaned Text", value=cleaned_text, height=350)

st.subheader("Word Frequencies")
top_n = st.sidebar.slider("Select number of top words", min_value=5, max_value=100, value=20, step=5)

fig, freq_df = plotting_word_frequencies(clean_dataframe, text_column='text_clean', top_n=top_n)
st.pyplot(fig)

st.subheader("Word Frequency Data")
st.dataframe(freq_df)