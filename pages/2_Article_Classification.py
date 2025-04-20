from src.preprocessing.load_label import load_labeled_dataset
from src.classification.category_scores import category_scores
from src.classification.category import categorization_pipeline, classify_article
from src.visualization.bar_graphs import plot_top_categories
from src.visualization.pca_attempt import plot_3d_vectors
from collections import defaultdict
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

st.set_page_config(page_title="Article Classification and Visualization", layout="wide")
st.title("Article Categorization")

df, category_index = load_labeled_dataset(r"..\Hitachi_1\dataset\20_newsgroup")

@st.cache_resource()
def get_pinecone():
    return categorization_pipeline()

@st.cache_resource()
def get_category(text):
    return category_scores(text)

def filter_embeddings(reduced_embeddings, labels, selected_ids):
    filtered_embeddings = []
    filtered_labels = []

    for vec, label in zip(reduced_embeddings, labels):
        if label in selected_ids:
            filtered_embeddings.append(vec)
            filtered_labels.append(label)

    return np.array(filtered_embeddings), filtered_labels

pc, index_name, reduced_embeddings, labels, pca_model = get_pinecone()

user_article = st.text_area("Enter your article below to classify it:", height=500)

top_k = st.sidebar.slider("Top K Matches", min_value=1, max_value=20, value=10)

if "run_classify" not in st.session_state:
    st.session_state.run_classify = False


if st.button("Classify"): 
    st.session_state.run_classify = True

# Now, to classify a new article:
if st.session_state.run_classify:
    if user_article.strip():
        with st.spinner("Classifying..."):

            predicted_category, query_results= classify_article(pc, index_name, user_article, top_k=top_k)
            
            top_articles = [
                {"id": m["id"], "score": m["score"], "category": m["metadata"]["category"]}
                for m in query_results["matches"]
            ]

            selected_categories = list(set(article['category'] for article in top_articles))

            #started the 3D rep
            #getting the vector for the user input
            user_vector = pc.inference.embed(
                model="multilingual-e5-large",
                inputs=[user_article],
                parameters={"input_type": "query"}
            )[0]['values']

            new_point_3d = pca_model.transform([user_vector])[0]
        
        # st.write(f"User vector length: {len(user_vector)}")
        index_to_category_pred = {i:k for k, i in category_index.items()}
        
        label_to_category = index_to_category_pred
        category_to_index = {v:k for k, v in index_to_category_pred.items()}
        selected_category_ids = list(set(match['metadata']['category'] for match in query_results['matches']))

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
            category_string = index_to_category_pred.get(int(raw_category))

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

        st.markdown("JESSI could you explain a little bit about this graph")
        top_categories = plot_top_categories(query_results["matches"])
        st.pyplot(top_categories)

        st.markdown("MATT can you explain a little bit about that this graph means here")
        category_bar = get_category(user_article)
        st.pyplot(category_bar)    


        st.markdown("### 🏷️ Selected Categories:")
        for cat_id in selected_category_ids:
            st.markdown(f"- 🟣 **Category ID:** `{cat_id}`")

        filtered_embeddings, filtered_labels = filter_embeddings(reduced_embeddings, labels, selected_category_ids)
        d_plot = plot_3d_vectors(filtered_embeddings, filtered_labels, new_point_3d)
        st.plotly_chart(d_plot, use_container_width=True)

    else:
        st.warning("Please enter text before pressing 'Classify'")