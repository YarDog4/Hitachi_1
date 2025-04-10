#I would like to get the number of words in each of the 20 categories here without the duplicate data
import matplotlib.pyplot as plt

def average_word_count(df, text_column="text", category_map=None):
    
    #Adding a new column for word count
    df['word_count'] = df[text_column].apply(lambda x: len(str(x).split()))
    
    #Taking the average of the word counts based on category of Article
    avg_word_count = df.groupby('category')['word_count'].mean().reset_index()
    avg_word_count.columns = ['category_index', 'average_word_count']

    
    #Mapping category names
    if category_map:
        index_to_category = {i: k for k, i in category_map.items()}
        avg_word_count['category_name'] = avg_word_count['category_index'].map(index_to_category)
    else:
        avg_word_count['category_name'] = avg_word_count['category_index'].astype(str)

        
    avg_word_count = avg_word_count.sort_values('category_index')
    
    #creating the bar plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(avg_word_count['category_index'], avg_word_count['average_word_count'], color='orange')
    ax.set_xticks(avg_word_count['category_index'])
    ax.set_xticklabels(avg_word_count['category_name'], rotation=90)

    ax.set_xlabel("Category")
    ax.set_title("Average word count of each category")
    plt.tight_layout()
    
    return fig, avg_word_count
