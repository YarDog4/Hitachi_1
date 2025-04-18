from src.preprocessing.load_label import load_labeled_dataset
# from src.visualization.matches import plot_similarity_heatmap
# from src.visualization.matches import plot_top_categories
# from src.visualization.matches import plot_similarity_network
# from src.visualization.matches import plot_pie_chart
# from src.visualization.matches import plot_embedding_time
# from src.visualization.matches import plot_cumulative_similarity
from src.classification.category import categorization_pipeline, classify_article
import matplotlib.pyplot as plt
import streamlit as st


st.set_page_config(page_title="Article Classification and Visualization", layout="wide")
st.title("Article Categorization")

df, category_index = load_labeled_dataset(r"..\Hitachi_1\dataset\20_newsgroup")

@st.cache_resource()
def get_pinecone():
    return categorization_pipeline()

pc, index_name = get_pinecone()

user_article = st.text_area("Enter your article below to classify it:", height=500)

if st.button("Classify"): 

# Now, to classify a new article:
    if user_article.strip():
        with st.spinner("Classifying..."):

            top_k = st.sidebar.slider("Top K Matches", min_value=1, max_value=20, value=10)
            predicted_category, query_results = classify_article(pc, index_name, user_article, top_k=top_k)
       
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
        # heatmap = plot_similarity_heatmap(query_results['matches'])
        # st.pyplot(heatmap)

        # top_categories = plot_top_categories(query_results['matches'])
        # st.pyplot(top_categories)

        # similarity_network = plot_similarity_network(query_results['matches'])
        # st.pyplot(similarity_network)

        # pie_plot = plot_pie_chart(query_results['matches'])
        # st.pyplot(pie_plot)

        # embed = plot_embedding_time()
        # st.pyplot(embed)

        # cummulative_similarity = plot_cumulative_similarity(query_results['matches'])
        # st.pyplot(cummulative_similarity)

    

    else:
        st.warning("Please enter text before pressing 'Classify'")