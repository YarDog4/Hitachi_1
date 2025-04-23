import matplotlib.pyplot as plt
from collections import Counter

import matplotlib.pyplot as plt
from collections import Counter

def plot_top_categories(matches, category_index):
    # Reverse the category index to map indices to names
    index_to_category = {v: k for k, v in category_index.items()}

    # Extract and count categories
    categories = [match['metadata']['category'] for match in matches]
    counter = Counter(categories)

    # Map numeric categories to their names
    named_categories = [index_to_category.get(int(cat), f"Unknown ({cat})") for cat in counter.keys()]
    frequencies = list(counter.values())

    # Plot
    fig, ax = plt.subplots()
    ax.bar(named_categories, frequencies, color="orange")
    ax.set_title("Top Categories Among Matches")
    ax.set_xlabel("Category")
    ax.set_ylabel("Frequency")
    plt.xticks(rotation=90)
    fig.tight_layout()

    return fig

