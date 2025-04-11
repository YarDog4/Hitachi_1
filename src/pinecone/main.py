# Importing the Pinecone library
from pinecone import Pinecone
from pinecone import ServerlessSpec
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import pandas as pd
import time
from collections import Counter
from glob import glob
import os

from ..preprocessing.load_label import load_labeled_dataset

#Location where the data set is at
data_directory = r"C:\Users\andre\Desktop\Yaren's Stuff\DataScience\Hitachi_1\dataset\20_newsgroup"

#Autodetecting csv files
def find_csv(dir, pattern="*.csv"):
    files = glob(os.path.join(dir, pattern))
    if files:
        return files[0]
    else:
        return None

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
                region="us-east-1"
            )
        )
        print(f"Index {index_name} created successfully.")
    return index_name


# Function to send data in batches with delay
def send_in_batches(pc, data, batch_size=96, delay=5):
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]

        embeddings = pc.inference.embed(
            model="multilingual-e5-large",
            inputs=[d['text'] for d in batch],
            parameters={"input_type": "passage", "truncate": "END"}
        )
        print(embeddings[0])  # Example print, you can process it differently here
        time.sleep(delay)


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

def vis(results):
    matches = results['matches']
    df = pd.DataFrame([{
        'id': match['id'],
        'score': match['score'],
        'text': match['metadata']['text']
    } for match in matches])

    # --- Bar Chart of Similarity Scores by ID ---
    plt.figure(figsize=(10, 6))
    sns.barplot(x='id', y='score', data=df, palette='coolwarm', hue='id', legend=False)
    plt.title('Similarity Scores by Match ID')
    plt.xlabel('Vector ID')
    plt.ylabel('Similarity Score')
    plt.ylim(0.75, 1.0)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # --- Histogram of Score Distribution ---
    plt.figure(figsize=(8, 5))
    sns.histplot(df['score'], bins=10, kde=True, color='skyblue')
    plt.title('Distribution of Similarity Scores')
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()

    # --- Scatter Plot of Scores ---
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(df)), df['score'], color='darkred')
    plt.title('Scatter Plot of Similarity Scores')
    plt.xlabel('Match Index')
    plt.ylabel('Score')
    plt.xticks(range(len(df)), df['id'], rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # --- Word Cloud from Metadata Text ---
    combined_text = " ".join(df['text'].tolist())
    wordcloud = WordCloud(width=1000, height=500, background_color='white', colormap='plasma').generate(combined_text)

    plt.figure(figsize=(15, 7))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud of Metadata Text')
    plt.show()

# Main function for user interaction
def main():

    default_data_directory = os.path.join(os.getcwd(), "dataset")
    default_data_directory = os.path.join(default_data_directory, "csv")

    api_key = input("Enter your Pinecone API key: ")
    pc = initialize_pinecone(api_key)

    print("Creating Pinecone index...")
    index_name = create_index(pc)

    # Initialize data_from_df with your data
    # file_path = input("Enter the path to your CSV file: ")

    file_path = find_csv(default_data_directory)

    if file_path:
        print(f"Automatically found CSV file: {file_path}")
        df = pd.read_csv(file_path)
    else:
        print("Error")

    # Ensure that the CSV file has 'id' and 'text' columns
    if 'id' not in df.columns or 'text' not in df.columns:
        print("CSV file must contain 'id' and 'text' columns.")
        return

    df['id'] = df['id'].astype(str)
    # Prepare data from the CSV file
    data_from_df = df[['id', 'text', 'category']].to_dict(orient='records')

    send_in_batches(pc, data_from_df, batch_size=96, delay=5)

    # Wait until the index is ready
    while not pc.describe_index(index_name).status['ready']:
        time.sleep(1)

    # Insert vectors
    print("Inserting vectors into Pinecone...")
    embeddings = pc.inference.embed(
        model="multilingual-e5-large",
        inputs=[d['text'] for d in data_from_df],
        parameters={"input_type": "passage", "truncate": "END"}
    )
    process_and_insert_vectors(pc, index_name, data_from_df, embeddings)

    # # User interaction for querying
    # query = input("Enter a query for similarity search: ")
    # results = query_index(pc, index_name, query)
    # vis(results)

    # Now, to classify a new article:
    article = input("Enter an article to classify: ")
    predicted_category, query_results = classify_article(pc, index_name, article, top_k=5)
    print("Predicted Category:", predicted_category)


if __name__ == '__main__':
    main()
