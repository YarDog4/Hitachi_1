from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import random

def plot_vector_space_3d(category_vectors_dict, user_vector):
    
    """
    category_vectors_dict: dict of {category_label: list_of_vectors}
    user_vector: single embedding vector (list or np.array)
    """
    all_vectors = []
    labels = []
    
    for category, vectors in category_vectors_dict.items():
        all_vectors.extend(vectors)
        labels.extend([category] * len(vectors))
    
    all_vectors.append(user_vector)
    labels.append("User Input")

    # Reduce to 3D with PCA
    pca = PCA(n_components=3)
    reduced = pca.fit_transform(np.array(all_vectors))

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Plot each category group
    unique_labels = list(category_vectors_dict.keys())
    color_map = plt.cm.get_cmap('tab20', len(unique_labels))  # 20-category color palette

    start = 0
    for i, label in enumerate(unique_labels):
        num_points = len(category_vectors_dict[label])
        reduced_cat = reduced[start:start + num_points]
        ax.scatter(reduced_cat[:, 0], reduced_cat[:, 1], reduced_cat[:, 2],
                   label=f"Category {label}",
                   color=color_map(i))
        start += num_points

    # Plot the user input vector
    reduced_user = reduced[-1]
    ax.scatter(reduced_user[0], reduced_user[1], reduced_user[2],
               c='red', s=120, marker='X', label='User Input')

    ax.set_title("🧠 3D Vector Space Projection (PCA)")
    ax.legend(loc='best')
    return fig
