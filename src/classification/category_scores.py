import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()

Categories = [
    "atheism", "space", "pc hardware", "windows computer",
    "for sale", "automobiles", "motorcycles", "baseball", "hockey",
    "crypto", "electronics", "medicine", "Christianity", "guns",
    "middle east", "misc politics", "misc religion", "mac hardware",
    "graphics", "windows misc"
]

#Embedding Functions
@st.cache_data(show_spinner=False)
def embed_text(text: str, input_type="passage") -> list:
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    return pc.inference.embed(
        model="multilingual-e5-large",
        inputs=[text],
        parameters={"input_type": input_type, "truncate": "END"}
    )[0]["values"]

@st.cache_resource
def get_category_vectors() -> dict:
    return {
        cat: embed_text(f"Text that discusses {cat}", input_type="query")
        for cat in Categories
    }

#Cosine Similarity
def compare_direct_similarity(user_text: str):
    user_vec = np.array(embed_text(user_text)).reshape(1, -1)
    categories = get_category_vectors()

    scores = {
        cat: cosine_similarity(user_vec, np.array(vec).reshape(1, -1))[0][0]
        for cat, vec in categories.items()
    }

    return pd.DataFrame([scores]).T.sort_values(by=0, ascending=False)

#plot
def plot_scores(df_sorted, title):
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.bar(df_sorted.index, df_sorted[0], color="orange")
    ax.set_ylim(0.7, 0.9)
    ax.set_xlabel("Categories")
    ax.set_ylabel("Similarity Score")
    ax.set_title(title)
    ax.set_xticks(range(len(df_sorted)))
    ax.set_xticklabels(df_sorted.index, rotation=90)
    fig.tight_layout()
    plt.show()
    return fig
