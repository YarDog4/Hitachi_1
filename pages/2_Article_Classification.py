from src.preprocessing.load_label import load_labeled_dataset
from src.classification.category_scores import category_scores
from src.classification.category import categorization_pipeline, classify_article
from src.visualization.bar_graphs import plot_top_categories
from src.visualization.vector_rep import plot_vector_space_3d
from collections import defaultdict
import matplotlib.pyplot as plt
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

pc, index_name = get_pinecone()

user_article = st.text_area("Enter your article below to classify it:", height=500)

top_k = st.sidebar.slider("Top K Matches", min_value=1, max_value=20, value=10)

if st.button("Classify"): 

# Now, to classify a new article:
    if user_article.strip():
        with st.spinner("Classifying..."):

            predicted_category, query_results = classify_article(pc, index_name, user_article, top_k=top_k)
            
            #started the 3D rep
            #getting the vector for the user input
            user_vector = pc.inference.embed(
                model="multilingual-e5-large",
                inputs=[user_article],
                parameters={"input_type": "query"}
            )[0]['values']

        st.write(f"User vector length: {len(user_vector)}")
        index_to_category_pred = {i:k for k, i in category_index.items()}
        category_vector = defaultdict(list)

        if predicted_category:
            category_real = index_to_category_pred.get(int(predicted_category))
            st.success(f"Your article is about: **{category_real.upper()}** ({predicted_category}) ")
        else:
            st.warning("❌ Could not determine a category. Try with a more descriptive article.")

        st.subheader("Top Matches")
        low_score = False
        for match in query_results['matches']:
            
            raw_category = match['metadata'].get('category')
            category_string = index_to_category_pred.get(int(raw_category))

            if match['score'] < 0.75 and not low_score:
                st.warning("⚠️ Warning: Low similarity score — result may be unreliable.")
                low_score = True
                break

            st.markdown(f"-Article ID: {match['id']}, Score: {match['score']:.3f}, Category: {category_string.upper()} ({raw_category})")

            cat = int(raw_category)
            category_vector[cat].append(match['values'])

            # for i, vec in enumerate(category_vector[cat]):
                # st.write(f"Vector {i} length: {len(vec)}")
        
        st.subheader("Visualizations")

        st.markdown("JESSI could you explain a little bit about this graph")
        top_categories = plot_top_categories(query_results["matches"])
        st.pyplot(top_categories)

        st.markdown("MATT can you explain a little bit about that this graph means here")
        category_bar = get_category(user_article)
        st.pyplot(category_bar)    

        st.markdown("i hope ts works")
        all_cats = list(category_vector.keys())
        selected_cats = st.multiselect(
            "Select categories to visualize in 3D",
            options=all_cats,
            default=all_cats,
            format_func=lambda x: index_to_category_pred.get(x, f"Category {x}"),
            key="3d_category_selector"
        )
        filtered_vecs = {k: category_vector[k] for k in selected_cats}
        d_plot = plot_vector_space_3d(filtered_vecs, user_vector)
        st.pyplot(d_plot)

    else:
        st.warning("Please enter text before pressing 'Classify'")