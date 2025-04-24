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
    
    Use the tools in the sidebar to: choose your choice of data directory, change what document is shown in the text cleaning comparison, change how many words are shown in the word frequencies
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
    st.badge(f"Loaded **{len(df)}** documents from **{len(category_index)}** categories", color="orange")
    st.markdown("---")
    st.markdown("The graph below shows the average word count of each article in each category. We did this analysis to see what approach would be the best way to categorize articles. " \
    "Our initial idea was to categorize political articles on whether they were neutral, left-leaning, or right-leaning. " \
    "We decided to instead take advantage of all of our categories, and you will see on the **Article Classification** page what we implemented.")
    fig, avg_wc = average_word_count(df, text_column='text', category_map=category_index)
    st.pyplot(fig)

    st.markdown("#### Average Word Count Dataframe")
    st.dataframe(avg_wc)

st.markdown("---")

#Comparing Original vs. Cleaned Text for a given document
st.header("Before vs. After: Text Cleaning Comparison")
st.badge("Interactive", color="orange")
st.markdown("You can use the side navigation bar to choose different article indicies to see different raw vs. cleaned articles")

doc_index = st.sidebar.number_input(
    "Enter Document Index for Comparison", 
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
st.text_area("Text box below displays the raw text data", value=original_text, height = 350)
st.subheader("Cleaned Text")
st.text_area("Text box below displays the cleaned text data", value=cleaned_text, height=350)

st.markdown("---")

st.subheader("Word Frequencies")
st.badge("Interactive", color="orange")
st.markdown("The graph below shows the number of times a word shows up throughout the entire datasest in decending frequency. " \
"You can choose how many words to display from the side navigation drawer. The **default** is top 20 words.")
top_n = st.sidebar.slider("Select number of top words", min_value=5, max_value=100, value=20, step=5)

fig, freq_df = plotting_word_frequencies(clean_dataframe, text_column='text_clean', top_n=top_n)
st.pyplot(fig)

st.markdown("#### Word Frequency Dataframe")
st.dataframe(freq_df)