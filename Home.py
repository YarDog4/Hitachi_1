import streamlit as st


st.set_page_config(page_title="Hitachi Capstone Dashboard", page_icon="📊", layout="wide")

st.title("📚 Welcome to the Hitachi 499 Capstone App")

st.sidebar.success("Navigate to these pages above.")

st.markdown("""
This Streamlit application is designed to analyze, clean, classify, and visualize articles 
from the 20 Newsgroups dataset.

Use the sidebar to navigate through the sections:

- 🔍 **1 Cleaning and Stats**: Explore raw and cleaned text data with frequency stats.
- 🧠 **2 Article Categorization**: Input a custom article and get predicted categories using Pinecone.
- 📈 **3 Visualizations**: Dive into visual summaries of embeddings and similarity metrics.

---

To get started, choose a section from the sidebar.
""")


st.subheader("Introduction")
st.text("A quick what we did - JESSI")

st.subheader("What does this app do?")
st.text("This app takes a text and analyzes it for you, comparing it to articles in our database and categorizing it.")

st.subheader("How to use?")
st.text("You use our app by - YAREN")
