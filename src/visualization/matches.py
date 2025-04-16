from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.manifold import TSNE

# --- Visualizations for Pinecone Matches ---
def plot_similarity_heatmap(matches):
    scores = np.array([match['score'] for match in matches])
    data = scores.reshape(1, -1)
    fig, ax = plt.subplots(figsize=(10, 2))
    sns.heatmap(data, cmap="YlGnBu", annot=True, cbar=True, xticklabels=[m['id'] for m in matches], ax=ax)
    ax.set_title("Similarity Scores Heatmap")
    ax.set_yticks([])
    return fig

def plot_top_categories(matches):
    categories = [match['metadata']['category'] for match in matches]
    counter = Counter(categories)
    fig, ax = plt.subplots()
    ax.bar(counter.keys(), counter.values())
    ax.set_title("Top Categories among Matches")
    ax.set_xlabel("Category")
    ax.set_ylabel("Frequency")
    return fig

def plot_similarity_network(matches, threshold=0.5):
    G = nx.Graph()
    for match in matches:
        G.add_node(match['id'], category=match['metadata']['category'])
    for i in range(len(matches)):
        for j in range(i + 1, len(matches)):
            sim = np.dot(matches[i]['values'], matches[j]['values']) / (
                np.linalg.norm(matches[i]['values']) * np.linalg.norm(matches[j]['values']))
            if sim > threshold:
                G.add_edge(matches[i]['id'], matches[j]['id'], weight=sim)
    pos = nx.spring_layout(G)

    fig, ax = plt.subplots(figsize=(10,6))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', ax=ax)
    ax.set_title("Similarity Network Graph")
    return fig

def plot_tsne_embeddings(matches):
    embeddings = np.array([match['values'] for match in matches])
    tsne = TSNE(n_components=2, random_state=42, perplexity=5)
    reduced = tsne.fit_transform(embeddings)
    categories = [match['metadata']['category'] for match in matches]
    df = pd.DataFrame({'x': reduced[:, 0], 'y': reduced[:, 1], 'category': categories})
    fig, ax = plt.subplots(figsize=(8,6))
    sns.scatterplot(data=df, x='x', y='y', hue='category', ax=ax)
    ax.set_title("t-SNE of Embeddings")
    return fig

def plot_pie_chart(matches):
    categories = [match['metadata']['category'] for match in matches]
    counter = Counter(categories)
    fig, ax = plt.subplots()
    ax.pie(counter.values(), labels=counter.keys(), autopct='%1.1f%%')
    ax.set_title("Metadata Category Distribution")
    return fig

def plot_embedding_time():
    times = np.random.uniform(0.1, 1.0, 20)
    fig, ax = plt.subplots()
    ax.plot(times, marker='o')
    ax.set_title("Simulated Embedding Time per Batch")
    ax.set_xlabel("Batch Index")
    ax.set_ylabel("Time (s)")
    return fig

def plot_cumulative_similarity(matches):
    scores = sorted([match['score'] for match in matches], reverse=True)
    cumulative = np.cumsum(scores)
    fig, ax = plt.subplots()
    ax.plot(cumulative, marker='o')
    ax.set_title("Cumulative Similarity Scores")
    ax.set_xlabel("Top-N Matches")
    ax.set_ylabel("Cumulative Score")
    return fig

def plot_score_vs_text_length(matches):
    scores = [match['score'] for match in matches]
    lengths = [len(match['metadata']['text']) for match in matches]
    fig, ax = plt.subplots()
    ax.scatter(lengths, scores)
    ax.set_title("Score vs Text Length")
    ax.set_xlabel("Text Length")
    ax.set_ylabel("Similarity Score")
    return fig

def plot_embedding_statistics(matches):
    vectors = np.array([match['values'] for match in matches])
    means = np.mean(vectors, axis=1)
    fig, ax = plt.subplots()
    ax.hist(means, bins=10)
    ax.set_title("Distribution of Embedding Vector Means")
    ax.set_xlabel("Mean Value")
    ax.set_ylabel("Frequency")
    return fig


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
