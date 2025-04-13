from preprocessing.load_label import load_labeled_dataset
from preprocessing.cleaning_data import clean_text
from visualization.avg_word_count import average_word_count
from visualization.word_frequency import plotting_word_frequencies
import matplotlib.pyplot as plt
import streamlit as st


#Using streamlit to make visualizations
st.title("capstone")

data_directory = st.sidebar.text_input(
    "Data Directory",
    r"..\Hitachi_1\dataset\20_newsgroup"
)

@st.cache_data()
def load_data(data_directory: str):
    # Call your custom loader which returns df and category_map
    return load_labeled_dataset(data_directory)

if data_directory:
    #First load the dataset
    df, category_index = load_labeled_dataset(r"..\Hitachi_1\dataset\20_newsgroup")

    st.write(f"Loaded **{len(df)}** documents from **{len(category_index)}** categories")
    
    #Create some preprocessing graphs to display
    fig, avg_wc = average_word_count(df, text_column='text', category_map=category_index)
    st.pyplot(fig)

    st.subheader("Average Word Count Data")
    st.dataframe(avg_wc)

#Comparing Original vs. Cleaned Text for a given document
st.header("Before vs. After: Text CLeaning Comparison")

doc_index = st.sidebar.slider("Selec Document Index", 0, len(df)-1, 0)

#cleaned text df
@st.cache_data()
def clean():
    return clean_text(df.copy(), text_column="text", lemmatize=False)

clean_dataframe = clean()

#Texts
original_text = df.iloc[doc_index]["text"]
cleaned_text = clean_dataframe.iloc[doc_index]["text_clean"]

#Displaying
st.subheader("Original Text")
st.text_area("Original Text", value=original_text, height = 200)
st.subheader("Cleaned Text")
st.text_area("Cleaned Text", value=cleaned_text, height=200)

st.subheader("Word Frequencies")
top_n = st.sidebar.slider("Select number of top words", min_value=5, max_value=100, value=20, step=5)

# Use the cleaned text column for the analysis (ensure it exists)
fig, freq_df = plotting_word_frequencies(clean_dataframe, text_column='text_clean', top_n=top_n)
st.pyplot(fig)

# Optionally display the frequency data in a table
st.subheader("Word Frequency Data")
st.dataframe(freq_df)