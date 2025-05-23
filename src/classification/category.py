# Importing the Pinecone library
import pickle
import httpx
from pinecone import Pinecone
from pinecone import ServerlessSpec
import pandas as pd
import time
import random
import numpy as np
from collections import Counter
from glob import glob
import os
from dotenv import load_dotenv
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.decomposition import PCA


from src.preprocessing.load_label import load_labeled_dataset
from src.preprocessing.cleaning_data import clean_text

load_dotenv()

environment = os.getenv("PINECONE_ENVIRONMENT")
data_directory = os.getenv(r"DATASET_PATH")

#performing PCA
def pca_3d_vec(embedding_vectors):
    pca= PCA(n_components=3)
    reduced_embedding = pca.fit_transform(embedding_vectors)
    return reduced_embedding, pca

def pca_2d_vec(embedding_vectors):
    pca= PCA(n_components=2)
    reduced_embedding = pca.fit_transform(embedding_vectors)
    return reduced_embedding, pca

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

#Initialize Pinecone connection
def initialize_pinecone(api_key):
    pc = Pinecone(api_key=api_key)
    return pc

#Create Pinecone index
def create_index(pc):
    index_name = os.getenv('PINECONE_INDEX').lower()

    #Check if the index already exists
    try:
        pc.describe_index(index_name)
        print(f"Index {index_name} already exists.")
    except Exception as e:
        #If index doesn't exist, create it
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

def embed_batch(pc, batch, attempt=1, max_attempts=5):
    try:
        return pc.inference.embed(
                model="multilingual-e5-large",
                inputs=[str(d['text_clean']) for d in batch],
                parameters={"input_type": "passage", "truncate": "END"}
            )
    except Exception as e:
        if "RESOURCE_EXHAUSTED" in str(e) and attempt < max_attempts:
            wait = 60 + random.uniform(1, 10)  # wait 60-70s
            print(f"🚧 Rate limit hit. Retrying in {wait:.1f}s (Attempt {attempt}/{max_attempts})")
            time.sleep(wait)
            return embed_batch(pc, batch, attempt + 1)
        else:
            raise e

def send_in_batches_parallel(pc, data, batch_size=96, max_workers=2, delay=2):
    all_embeddings = []
    batches = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
    print(f"🚀 Embedding {len(batches)} batches in parallel with {max_workers} threads...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(embed_batch, pc, batch): idx for idx, batch in enumerate(batches)}

        for future in as_completed(futures):
            idx = futures[future]
            try:
                result = future.result()
                all_embeddings.extend(result)
                print(f"✅ Batch {idx + 1}/{len(batches)} embedded.")
                time.sleep(delay)
            except Exception as e:
                print(f"❌ Batch {idx + 1} failed: {e}")

    return all_embeddings

#Process and insert vectors into Pinecone
def process_and_insert_vectors(pc, index_name, data, embeddings, batch_size=100):
    index = pc.Index(index_name)
    vectors = []
    for d, e in zip(data, embeddings):
        vectors.append({
            "id": d['id'],
            "values": e,
            "metadata": {
                'category': d.get('category')
            }
        })
    
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i + batch_size]
        success = False

        #I used some AI here in attempt to solve my Batch upload problem. Pinecone has limits, so it needs to rest before uploading batches
        #All batches should be uploaded
        for attempt in range(5):  # 5 retry attempts
            try:
                index.upsert(vectors=batch, namespace="ns1")
                print(f"📤 Uploaded batch {i // batch_size + 1}/{(len(vectors) + batch_size - 1) // batch_size}")
                success = True
                break
            except httpx.HTTPStatusError as e:
                wait = 2 ** attempt
                print(f"❌ HTTP error on batch {i // batch_size + 1}: {e}")
                print(f"⏳ Retrying in {wait} seconds...")
                time.sleep(wait)
            except Exception as e:
                wait = 2 ** attempt
                print(f"❌ General error on batch {i // batch_size + 1}: {e}")
                print(f"⏳ Retrying in {wait} seconds...")
                time.sleep(wait)

        if not success:
            print(f"💥 Failed to upload batch {i // batch_size + 1} after multiple attempts.")

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

    matches = results['matches']
    if not matches:
        return None, results
    
    categories = [match['metadata'].get('category') for match in results['matches']
                  if 'category' in match['metadata']]
    
    if categories:
        most_common, count = Counter(categories).most_common(1)[0]
        return most_common, results
    else:
        fallback = matches[0]
        fallback_category = fallback['metadata'].get('category', "Unknown")
        return f"Most similar to: {fallback_category}", results
    
#Querying the index
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
    return results

def categorization_pipeline():

    api_key_ind = os.getenv('PINECONE_API_KEY')
    pc = initialize_pinecone(api_key_ind)

    print("Creating Pinecone index...")
    index_name = create_index(pc)

    #loading data directory and making a Dataframe
    default_data_directory = Path(__file__).resolve().parent.parent.parent
    cleaned_csv_path = default_data_directory / "dataset" / "csv" / "cleaned" / "cleaned.csv"
    embeddings_path = default_data_directory / "dataset" / "csv" / "embeddings.npy"
    metadata_path = default_data_directory / "dataset" / "csv" / "metadata.pkl"

    cleaned_csv_path.parent.mkdir(parents=True, exist_ok=True)

    if cleaned_csv_path.exists():
        df_clean = pd.read_csv(cleaned_csv_path)
    else:
        dir = default_data_directory / "dataset" / "20_newsgroup"
        df, category_index = load_labeled_dataset(dir)
        #clean the text
        df_clean = clean(df)
        df_clean.to_csv(cleaned_csv_path, index=False)
        
    print(f"Loaded **{len(df_clean)}** documents")

    #Prepare data from the CSV file
    df_clean['id'] = df_clean['id'].astype(str)
    df_clean['text_clean'] = df_clean['text_clean'].fillna("").astype(str)
    data_from_df = df_clean[['id', 'text', 'category', 'text_clean']].to_dict(orient='records')
    
    if embeddings_path.exists() and metadata_path.exists():
        print("✅ Loaded cached embeddings")
        embedding_vectors = np.load(embeddings_path, allow_pickle=True)
        with open(metadata_path, 'rb') as f:
            data_from_df = pickle.load(f)

    else:
        print("🧠 Sending data in batches...")
        embeddings = send_in_batches_parallel(pc, data_from_df, batch_size=96, max_workers=4)
        print("✅ Finished embedding in batches. Caching results")
        
        embedding_vectors = np.array([e['values'] for e in embeddings])

        np.save(embeddings_path, embedding_vectors)
        with open(metadata_path, 'wb') as f:
            pickle.dump(data_from_df, f)

    #need this for graphing
    labels = [entry['category'] for entry in data_from_df]
    reduced_3d_embeddings, pca_3d_model = pca_3d_vec(embedding_vectors)
    reduced_2d_embeddings, pca_2d_model = pca_2d_vec(embedding_vectors)

    #Wait until the index is ready
    stats = pc.describe_index(index_name).namespaces
    vector_count = stats.get("ns1", {}).get("vector_count", 0) if stats else 0

    if vector_count > 0:
        print("📦 Index already populated. Skipping upload.")
    else:
        print("📤Inserting vectors into Pinecone...")
        process_and_insert_vectors(pc, index_name, data_from_df, embedding_vectors, 100)

    return pc, index_name, reduced_3d_embeddings, pca_3d_model, reduced_2d_embeddings, pca_2d_model, labels
    
                