import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd

#Visualization of the most common words in the dataset

def plotting_word_frequencies(df, text_column='text_clean', top_n=20, dataset_name='20 Newsgroup original'):
    """
    This function plots a bar graph of the top N most frequent words in the cleaned text column of the DataFrame random_shuffle
    
    The parameters include:
    -df - The DataFrame containing the text data
    -text_column -The name of the column of the random_shuffle DataFrame
    -top_n: Number of words to display
    """
    
    #Make all text into one big string
    text_all = ' '.join(df[text_column].dropna().astype(str))
    
    #Counting all word frequencies
    word_counts = Counter(text_all.split())
    
    #Most common words based on the top_n parameter
    word_common = word_counts.most_common(top_n)
    words, frequencies = zip(*word_common)
    
    frequency_df = pd.DataFrame(word_common, columns=['word', 'frequncy'])

    #Plotting bar graph
    fig, ax = plt.subplots(figsize=(12, top_n * 0.4)) #Adjusts the size of the graph based on top_n parameter
    plt.barh(words, frequencies, color='orange')
    ax.invert_yaxis() #This code inverts the y-axis to show the most frequent word on top
    ax.set_title(f'Top {top_n} words in {dataset_name} dataset')
    ax.set_xlabel('Words')
    ax.set_ylabel('Frequency')
    plt.tight_layout(pad=0.5)

    return fig, frequency_df
    
    