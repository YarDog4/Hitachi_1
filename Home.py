import streamlit as st


st.set_page_config(page_title="Hitachi Capstone Dashboard", page_icon="📊", layout="wide")

st.title("📚 Welcome to the Hitachi 499 Capstone App")

st.sidebar.success("Navigate to these pages above.")

st.markdown("""
This Streamlit application is designed to analyze, clean, classify, and visualize articles 
from the 20 Newsgroups dataset.

Use the sidebar to navigate through the sections:

- **1 Home**: 
- **2 Preprocessing**: Explore raw and cleaned text data with frequency stats.
- **3 Article Categorization**: Input a custom article and get predicted categories using Pinecone with visual summaries of embeddings and similarity metrics.

---

To get started, choose a section from the sidebar.
""")


st.subheader("Introduction")
st.text("We created an app that categorizes any inputted text and provides category similarity scores and similar articles.")

st.subheader("What does this app do?")
st.text("This tool takes a text and analyzes it for you, comparing it to articles in our database and categorizing it. It also provides visualizations on the articles in the database and your text. ")

st.subheader("How to use?")
st.text("You use our app by - YAREN")
