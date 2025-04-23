from src.classification.category_scores import compare_direct_similarity, plot_scores
from src.classification.category import categorization_pipeline, classify_article
from src.preprocessing.load_label import load_labeled_dataset
from src.visualization.bar_graphs import plot_top_categories
from src.visualization.pca_attempt import plot_3d_vectors
from src.visualization.pca_attempt import plot_2d_vectors

import numpy as np
import streamlit as st
import os

#Caching resources for faster integration
@st.cache_resource()
def get_pinecone():
    return categorization_pipeline()

@st.cache_data(show_spinner=False)
def get_user_vector(text: str):
    user_vector = pc.inference.embed(
        model="multilingual-e5-large",
        inputs=[user_article],
        parameters={"input_type": "query"}
    )[0]['values']
    return user_vector

st.set_page_config(page_title="Article Classification and Visualization", layout="wide")
st.title("Article Categorization")

#Upload the category index and the dataframe here
df, category_index = load_labeled_dataset(os.getenv(r"RELATIVE_PATH"))


#This code helps only plot the categories the user article closely matches
def filter_embeddings(reduced_embeddings, labels, selected_ids):
    filtered_embeddings = []
    filtered_labels = []

    for vec, label in zip(reduced_embeddings, labels):
        if label in selected_ids:
            filtered_embeddings.append(vec)
            filtered_labels.append(label)

    return np.array(filtered_embeddings), filtered_labels

pc, index_name, reduced_3d_embeddings, pca_3d_model, reduced_2d_embeddings, pca_2d_model, labels = get_pinecone()

user_article = st.text_area("Enter your article below to classify it:", height=500)

#User can specify how many articles they want outputted from the data set
top_k = st.sidebar.slider("Top K Matches", min_value=1, max_value=20, value=10)

#Classify session state logic, prevents the tool from being refreshed when user chooses a new top K
if "run_classify" not in st.session_state:
    st.session_state.run_classify = False

if st.button("Classify"): 
    st.session_state.run_classify = True

# Now, to classify a new article:
if st.session_state.run_classify:
    if user_article.strip():
        with st.spinner("Classifying..."):

            predicted_category, query_results= classify_article(pc, index_name, user_article, top_k=top_k)

            #getting the vector for the user input
            if 'vector' in query_results:
                user_vector = query_results['vector']
            else: 
                user_vector = get_user_vector(user_article)

            new_point_3d = pca_3d_model.transform([user_vector])[0]
            new_point_2d = pca_2d_model.transform([user_vector])[0]
        
            index_to_category_pred = {i: k for k, i in category_index.items()}
            category_to_index = {v: k for k, v in index_to_category_pred.items()}

            article_ids = []

            top_articles = [
                {
                    "id": match["id"],
                    "score": match["score"],
                    "category": match["metadata"].get("category")
                }
                for match in query_results["matches"]
            ]

            selected_categories = list(set(article["category"] for article in top_articles))

            for cat_name in selected_categories:
                if cat_name in category_to_index:
                    cat_id = category_to_index[cat_name]

            if predicted_category:
                category_real = index_to_category_pred.get(int(predicted_category))
                st.success(f"Your article is about: **{category_real.upper()}** ({predicted_category}) ")
            else:
                st.warning("❌ Could not determine a category. Try with a more descriptive article.")

            st.subheader("Top Matches")
            low_score = False

            article_ids=[]
            for match in query_results['matches']:
                
                raw_category = match['metadata'].get('category')

                try:
                    raw_category_int = int(raw_category)
                    category_string = index_to_category_pred.get(raw_category_int, "Unknown")
                except (ValueError, TypeError):
                    category_string = "Unknown"
                
                article_ids.append(match['id'])

                if match['score'] < 0.75 and not low_score:
                    st.warning("⚠️ Warning: Low similarity score — result may be unreliable.")
                    low_score = True
                    break

                st.markdown(f"-Article ID: {match['id']}, Score: {match['score']:.3f}, Category: {category_string.upper()} ({raw_category})")
                
        st.subheader("Article Lookup")
        doc_index = st.selectbox("Select Article ID", article_ids, key="selected_article")
        original_text = df.iloc[int(doc_index)]['text']
        st.text_area("Original Text", value = original_text, height = 500)

        st.subheader("Visualizations")

        st.markdown("This bar chart visualizes the most common categories returned as top matches during article classification. Each bar represents a category from the dataset, and the height of the bar indicates how frequently that category appeared among the top results. .")
        top_categories = plot_top_categories(query_results["matches"], category_index)
        st.pyplot(top_categories)

        st.subheader("Selected Categories:")
        for cat_id in selected_categories:
            st.markdown(f"- 🟠 **Category Name:** {index_to_category_pred[cat_id].upper()} `{cat_id}`")

        filtered_2d_embeddings, filtered_2d_labels = filter_embeddings(reduced_2d_embeddings, labels, selected_categories)
        d2_plot = plot_2d_vectors(filtered_2d_embeddings, filtered_2d_labels, new_point_2d)
        st.plotly_chart(d2_plot, use_container_width=True)

        filtered_3d_embeddings, filtered_3d_labels = filter_embeddings(reduced_3d_embeddings, labels, selected_categories)
        d3_plot = plot_3d_vectors(filtered_3d_embeddings, filtered_3d_labels, new_point_3d)
        st.plotly_chart(d3_plot, use_container_width=True)

        st.subheader("Category Similarity Comparison")
        st.markdown("This graph uses vector embedding and cosine simularity to create a similarity value to each category for the classified text.")
        df_sorted = compare_direct_similarity(user_article)
        chart = plot_scores(df_sorted, "Direct Similarity to Category Prompts")
        st.pyplot(chart)   
    else:
        st.warning("Please enter text before pressing 'Classify'")
