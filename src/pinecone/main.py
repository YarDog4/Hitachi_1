# Importing the Pinecone library
import pickle
from pinecone import Pinecone
from pinecone import ServerlessSpec
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import pandas as pd
import time
import numpy as np
from collections import Counter
from glob import glob
import os
from dotenv import load_dotenv
from pathlib import Path
import sys

from src.preprocessing.load_label import load_labeled_dataset
from src.preprocessing.cleaning_data import clean_text

# from ..preprocessing.load_label import load_labeled_dataset

load_dotenv()

api_key_ind = os.getenv('PINECONE_API_KEY')
environment = os.getenv("PINECONE_ENVIRONMENT")
# index_name = os.getenv('PINECONE_INDEX')
#Location where the data set is at
data_directory = r"C:\Users\yaren\Desktop\School\499_Data_Capstone\Hitachi_1\dataset\20_newsgroup"

#Autodetecting csv files
def find_csv(dir, pattern="*.csv"):
    print(f"Searching in directory: {dir}")
    files = glob(os.path.join(dir, pattern))
    if files:
        print(f"Files: {files}")
        return files[0]
    else:
        print("No files were found")
        return None

#cleaning function
def clean(df):
    return clean_text(df.copy(), text_column="text", lemmatize=False)

# Function to initialize Pinecone connection
def initialize_pinecone(api_key):
    pc = Pinecone(api_key=api_key)
    return pc

# Function to create Pinecone index
def create_index(pc):
    index_name = input("Enter a name for the Pinecone index: ").lower()

    # Check if the index already exists
    try:
        pc.describe_index(index_name)
        print(f"Index {index_name} already exists.")
    except Exception as e:
        # If index doesn't exist, create it
        print(f"Creating Pinecone index {index_name}...")
        pc.create_index(
            name=index_name,
            dimension=1024,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region=environment
            )
        )
        print(f"Index {index_name} created successfully.")
    return index_name


# Function to send data in batches with delay
def send_in_batches(pc, data, batch_size=96, delay=5):
    all_embeddings = []
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]

        embeddings = pc.inference.embed(
            model="multilingual-e5-large",
            inputs=[str(d['text_clean']) for d in batch],
            parameters={"input_type": "passage", "truncate": "END"}
        )

        all_embeddings.extend(embeddings)
        print(f"✅ Embedded batch {i//batch_size + 1}/{(len(data) + batch_size - 1) // batch_size}")
        time.sleep(delay)
    return all_embeddings

# Function to process and insert vectors into Pinecone
def process_and_insert_vectors(pc, index_name, data, embeddings):
    index = pc.Index(index_name)
    vectors = []
    for d, e in zip(data, embeddings):
        vectors.append({
            "id": d['id'],
            "values": e['values'],
            "metadata": {
                'text': d['text'],
                'clean_text': d['clean_text'],
                'category': d.get('category')
            }

        })
    index.upsert(vectors=vectors, namespace="ns1")
    print(index.describe_index_stats())

#classifying articles
def classify_article(pc, index_name, article, top_k=5):
    embedding = pc.inference.embed(
        model="multilingual-e5-large",
        inputs=[article],
        parameters={"input_type": "query"}
    )

    index = pc.Index(index_name)
    results = index.query(
        namespace="ns1",
        vector=embedding[0].values,
        top_k=top_k,
        include_metadata=True
    )

    categories = [match['metadata'].get('category') for match in results['matches']
                  if 'category' in match['metadata']]
    
    if not categories:
        print("No category metadata found.")
        return None, results
    
    most_common, count = Counter(categories).most_common(1)[0]
    return most_common, results

# Function for querying the index
def query_index(pc, index_name, query):
    embedding = pc.inference.embed(
        model="multilingual-e5-large",
        inputs=[query],
        parameters={"input_type": "query"}
    )
    index = pc.Index(index_name)
    results = index.query(
        namespace="ns1",
        vector=embedding[0].values,
        top_k= int(input("How many results do you want? ")),
        include_values=False,
        include_metadata=True
    )
    print(results)
    return results

# def vis(results):
#     matches = results['matches']
#     df = pd.DataFrame([{
#         'id': match['id'],
#         'score': match['score'],
#         'text': match['metadata']['text']
#     } for match in matches])

#     # --- Bar Chart of Similarity Scores by ID ---
#     plt.figure(figsize=(10, 6))
#     sns.barplot(x='id', y='score', data=df, palette='coolwarm', hue='id', legend=False)
#     plt.title('Similarity Scores by Match ID')
#     plt.xlabel('Vector ID')
#     plt.ylabel('Similarity Score')
#     plt.ylim(0.75, 1.0)
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()

#     # --- Histogram of Score Distribution ---
#     plt.figure(figsize=(8, 5))
#     sns.histplot(df['score'], bins=10, kde=True, color='skyblue')
#     plt.title('Distribution of Similarity Scores')
#     plt.xlabel('Score')
#     plt.ylabel('Frequency')
#     plt.tight_layout()
#     plt.show()

#     # --- Scatter Plot of Scores ---
#     plt.figure(figsize=(10, 6))
#     plt.scatter(range(len(df)), df['score'], color='darkred')
#     plt.title('Scatter Plot of Similarity Scores')
#     plt.xlabel('Match Index')
#     plt.ylabel('Score')
#     plt.xticks(range(len(df)), df['id'], rotation=45)
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()

#     # --- Word Cloud from Metadata Text ---
#     combined_text = " ".join(df['text'].tolist())
#     wordcloud = WordCloud(width=1000, height=500, background_color='white', colormap='plasma').generate(combined_text)

#     plt.figure(figsize=(15, 7))
#     plt.imshow(wordcloud, interpolation='bilinear')
#     plt.axis('off')
#     plt.title('Word Cloud of Metadata Text')
#     plt.show()

# Main function for user interaction
def main():

    api_key = api_key_ind
    pc = initialize_pinecone(api_key)

    print("Creating Pinecone index...")
    index_name = create_index(pc)

    #loading data directory and making a Dataframe
    default_data_directory = Path(__file__).resolve().parent.parent.parent
    cleaned_csv_path = default_data_directory / "dataset" / "csv" / "cleaned.csv"
    embeddings_path = default_data_directory / "dataset" / "csv" / "embeddings.npy"
    metadata_path = default_data_directory / "dataset" / "csv" / "metadata.pkl"

    if cleaned_csv_path.exists():
        print("✅ Using cached cleaned.csv")
        df_clean = pd.read_csv(cleaned_csv_path)
    else:
        dir = default_data_directory / "dataset" / "20_newsgroup"
        df, category_index = load_labeled_dataset(dir)
        #clean the text
        df_clean = clean(df)
        df_clean.to_csv(cleaned_csv_path, index=False)
        

    print(f"Loaded **{len(df_clean)}** documents")

    # Ensure that the CSV file has 'id' and 'text' columns
    if 'id' not in df_clean.columns or 'text' not in df_clean.columns or 'text_clean' not in df_clean.columns:
        print("CSV file must contain 'id' and 'text' columns.")
        return

    # Prepare data from the CSV file
    df_clean['id'] = df_clean['id'].astype(str)
    df_clean['text_clean'] = df_clean['text_clean'].fillna("").astype(str)
    data_from_df = df_clean[['id', 'text', 'category', 'text_clean']].to_dict(orient='records')

    if embeddings_path.exists() and metadata_path.exists():
        print("✅ Loaded cached embeddings")
        embeddings = np.load(embeddings_path, allow_pickle=True)
        with open(metadata_path, 'rb') as f:
            data_from_df = pickle.load(f)
    else:
        print("🧠 Sending data in batches...")
        embeddings = send_in_batches(pc, data_from_df, batch_size=96, delay=5)
        print("✅ Finished embedding in batches. Caching results")
        np.save(embeddings_path, embeddings)
        with open(metadata_path, 'wb') as f:
            pickle.dump(data_from_df, f)


    # Wait until the index is ready
    stats = pc.describe_index(index_name).namespaces
    vector_count = stats.get("ns1", {}).get("vector_count", 0) if stats else 0

    if vector_count > 0:
        print("📦 Index already populated. Skipping upload.")
    else:
        print("📤Inserting vectors into Pinecone...")
        process_and_insert_vectors(pc, index_name, data_from_df, embeddings)

    
    # while not pc.describe_index(index_name).status['ready']:
    #     time.sleep(1)

    # Insert vectors
    # embeddings = pc.inference.embed(
    #     model="multilingual-e5-large",
    #     inputs=[d['text_clean'] for d in data_from_df],
    #     parameters={"input_type": "passage", "truncate": "END"}
    # )

    # # User interaction for querying
    # query = input("Enter a query for similarity search: ")
    # results = query_index(pc, index_name, query)
    # vis(results)

    # Now, to classify a new article:
    while True:
        article = input("Please input any article of your choice to classify (or type exit to discontinue): ")
        if article.lower() == "exit":
            break
        predicted_category, query_results = classify_article(pc, index_name, article, top_k=5)
        if predicted_category:
            print(f"Your article is about: **{predicted_category}**")
            print("Similarity scores from retrived vectors:")
            for match in query_results['matches']:
                print(f"-ID: {match['id']}, Score: {match['score']:.3f}, Category: {match['metadata'].get('category')}")
        else:
            print("❌ Could not determine a category. Try with a more descriptive article.")


if __name__ == '__main__':
    main()
