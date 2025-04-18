from preprocessing.load_label import load_labeled_dataset
from preprocessing.cleaning_data import clean_text
from visualization.avg_word_count import average_word_count
from visualization.word_frequency import plotting_word_frequencies
from visualization.matches import plot_similarity_heatmap
from visualization.matches import plot_top_categories
from visualization.matches import plot_similarity_network
from visualization.matches import plot_tsne_embeddings
from visualization.matches import plot_pie_chart
from visualization.matches import plot_embedding_time
from visualization.matches import plot_cumulative_similarity
from visualization.matches import plot_score_vs_text_length
from visualization.matches import plot_embedding_statistics

import sys
from pathlib import Path

# Add the parent directory of main.py to the Python path
sys.path.append(str(Path(__file__).resolve().parent))

from classification.category import categorization_pipeline, classify_article
import matplotlib.pyplot as plt
import streamlit as st


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

st.header("Article Categorization")


@st.cache_resource()
def get_pinecone():
    return categorization_pipeline()

pc, index_name = get_pinecone()

user_article = st.text_area("Enter your article below to classify it:", height=500)

if st.button("Classify"): 

# Now, to classify a new article:
    if user_article.strip():
        with st.spinner("Classifying..."):
            predicted_category, query_results = classify_article(pc, index_name, user_article, top_k=10)
       
        if predicted_category:
            index_to_category_pred = {i:k for k, i in category_index.items()}
            category_real = index_to_category_pred.get(int(predicted_category))
            st.success(f"Your article is about: **{category_real.upper()}** ({predicted_category}) ")
        else:
            st.warning("❌ Could not determine a category. Try with a more descriptive article.")

        st.subheader("Top Matches")
        low_score = False
        for match in query_results['matches']:
            index_to_category = {i:k for k, i in category_index.items()}
            raw_category = match['metadata'].get('category')
            category_string = index_to_category.get(int(raw_category))

            if match['score'] < 0.75 and not low_score:
                st.warning("⚠️ Warning: Low similarity score — result may be unreliable.")
                low_score = True
                break

            st.markdown(f"-Article ID: {match['id']}, Score: {match['score']:.3f}, Category: {category_string.upper()} ({raw_category})")

        # if st.button('Generate Graphs'):
        heatmap = plot_similarity_heatmap(query_results['matches'])
        st.pyplot(heatmap)

        top_categories = plot_top_categories(query_results['matches'])
        st.pyplot(top_categories)

        similarity_network = plot_similarity_network(query_results['matches'])
        st.pyplot(similarity_network)

        # tsne = plot_tsne_embeddings(query_results['matches'])
        # st.pyplot(tsne)

        pie_plot = plot_pie_chart(query_results['matches'])
        st.pyplot(pie_plot)

        embed = plot_embedding_time()
        st.pyplot(embed)

        cummulative_similarity = plot_cumulative_similarity(query_results['matches'])
        st.pyplot(cummulative_similarity)

        # score_vs_text = plot_score_vs_text_length(query_results['matches'])
        # st.pyplot(score_vs_text)

        # embed_stat = plot_embedding_statistics(query_results['matches'])
        # st.pyplot(embed_stat)

    else:
        st.warning("Please enter text before pressing 'Classify'")