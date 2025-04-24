import plotly.graph_objects as go
import numpy as np

def plot_2d_vectors(reduced_embeddings, labels, new_point=None):

    categories = np.unique(labels)

    fig = go.Figure()

    for category in categories:
        ind = np.where(labels == category)[0]
        fig.add_trace(go.Scatter(
            x=reduced_embeddings[ind, 0],
            y=reduced_embeddings[ind, 1],
            mode="markers",
            marker=dict(size=6),
            name=str(category)
        ))

    if new_point is not None:
        fig.add_trace(go.Scatter(
            x=[new_point[0]],
            y=[new_point[1]],
            mode="markers",
            marker=dict(size=10, color='black', symbol='square'),
            name='New Article'
        ))

    fig.update_layout(
        title="Interactive 2D PCA of Article Embeddings",
        scene=dict(
            xaxis_title='PCA 1',
            yaxis_title='PCA 2',
        ),
        legend_title="Categories",
        margin=dict(l=0, r=0, b=0, t=40)
    )
    return fig

def plot_3d_vectors(reduced_embeddings, labels, new_point=None):

    categories = np.unique(labels)

    fig = go.Figure()

    for category in categories:
        ind = np.where(labels == category)[0]
        fig.add_trace(go.Scatter3d(
            x=reduced_embeddings[ind, 0],
            y=reduced_embeddings[ind, 1],
            z=reduced_embeddings[ind, 2],
            mode="markers",
            marker=dict(size=4),
            name=str(category)
        ))

    if new_point is not None:
        fig.add_trace(go.Scatter3d(
            x=[new_point[0]],
            y=[new_point[1]],
            z=[new_point[2]],
            mode="markers",
            marker=dict(size=10, color='black', symbol='square'),
            name='New Article'
        ))

    fig.update_layout(
        title="Interactive 3D PCA of Article Embeddings",
        scene=dict(
            xaxis_title='PCA 1',
            yaxis_title='PCA 2',
            zaxis_title='PCA 3'
        ),
        legend_title="Categories",
        margin=dict(l=0, r=0, b=0, t=30)
    )
    return fig
