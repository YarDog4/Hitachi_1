import matplotlib.pyplot as plt
from collections import Counter

def plot_top_categories(matches):
    # Define category name mapping
    category_names = {
        0: "alt.atheism",
        1: "comp.graphics",
        2: "comp.os.ms-windows.misc",
        3: "comp.sys.ibm.pc.hardware",
        4: "comp.sys.mac.hardware",
        5: "comp.windows.x",
        6: "misc.forsale",
        7: "rec.autos",
        8: "rec.motorcycles",
        9: "rec.sport.baseball",
        10: "rec.sport.hockey",
        11: "sci.crypt",
        12: "sci.electronics",
        13: "sci.med",
        14: "sci.space",
        15: "soc.religion.christian",
        16: "talk.politics.guns",
        17: "talk.politics.mideast",
        18: "talk.politics.misc",
        19: "talk.religion.misc"
    }

    # Extract and count categories
    categories = [match['metadata']['category'] for match in matches]
    counter = Counter(categories)

    # Map numeric categories to names
    named_categories = [category_names.get(cat, f"Unknown ({cat})") for cat in counter.keys()]
    frequencies = list(counter.values())

    # Plot
    fig, ax = plt.subplots()
    ax.bar(named_categories, frequencies, color="orange")
    ax.set_title("Top Categories among Matches")
    ax.set_xlabel("Category")
    ax.set_ylabel("Frequency")
    plt.xticks(rotation=45)
    fig.tight_layout()

    return fig
