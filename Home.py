import streamlit as st


st.set_page_config(page_title="Hitachi Capstone Dashboard", page_icon="📊", layout="wide")

st.title("📚 Welcome to the Hitachi Team 1 Capstone Application")

st.sidebar.success("Navigate to these pages above.")

st.markdown("""
This Streamlit application is designed to analyze, clean, classify, and visualize articles 
from Kaggle's [20 Newsgroup original](https://www.kaggle.com/datasets/au1206/20-newsgroup-original/data) dataset.

Use the sidebar (right pointing arrow on the top left of the page) to navigate through these sections:

1) **Home** 
2) **Preprocessing**: Explore raw and cleaned text data with frequency stats.
3) **Article Categorization**: Input a custom article and get predicted categories using Pinecone with visual summaries of embeddings and similarity metrics.

---

""")


st.subheader("Introduction")
st.text("We created an app that categorizes any inputted text and provides category similarity scores and similar articles. Using over 19,000 articals our system can catagorize any artical you submit along with showing up to the 20 most closly related articles from our database.")

st.subheader("What does this app do?")
st.markdown("""
This tool takes a text and analyzes it for you, comparing it to articles in our database and categorizing it. It also provides visualizations on the articles in the database and your text.
            
---

""")

st.subheader("How to use?")
st.markdown("""
📘 **Instructions on how to get started** with this app can be found in our [GitHub Repository](https://github.com/YarDog4/Hitachi_1)'s `README.md`.


##### 🛠️ Getting Started

- When you launch the app, head over to the **Preprocessing** page first.  
We recommend keeping your terminal open to monitor the **Pinecone embedding progress**.

- 🎨 The embedding process is **parallelized** to speed things up while respecting the free Pinecone API limit.  
You may see print statements like:

```bash
🚧 Rate limit hit...
```

- Don't worry — the code will **automatically pause and resume** embedding when the limit resets!

- Once embedding is complete, **preprocessing graphs** (our visual dataset analysis) will appear.


##### 🎛️ Interactive Features

- Every page in this app includes a **sidebar** 🧯 for adjusting variables and observing changes in the graphs.


##### 🧠 Article Classification

- This is the **core feature** of our app!

- Just like before, there's a **sidebar for interactivity**.

- You might see messages like:
```bash
📤 Uploaded batch
```
- These indicate that your vectors are being successfully uploaded to Pinecone.

- Once that's done, a **text box** will appear where you can paste an article of your choice.  
- Hit the **"Classify"** button to see the **custom visualizations** built by the Hitachi Team 1!


##### 🎓 Final Note

- This platform is built as an **educational tool**, but it has plenty of potential for further development.  
We hope you enjoy exploring our application! 🌟
""")