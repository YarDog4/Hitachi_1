import nltk
import re

#Ensure a resource is downloaded before proceeding
def safe_download(resource, path):
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(resource, quiet=True)

def prepare_nltk():
    #Ensure both stopwords and wordnet are available
    safe_download('stopwords', 'corpora/stopwords')
    safe_download('wordnet', 'corpora/wordnet')
    stop_words = set(nltk.corpus.stopwords.words('english'))
    lemmatizer = nltk.stem.WordNetLemmatizer()
    return stop_words, lemmatizer

def clean_text(df, text_column="text", lemmatize=False):
    """
    This function will clean the dataset for preprocessing visualizations.
    The steps include:
    -Lowercase all object columns
    -Stripping whitespace from text column
    -Removes emails, URLs, digits, punctuation
    -Removes stopwords
    """
    stop_words, lemmatizer = prepare_nltk()
    #Lowercase all stings
    df = df.apply(lambda x: x.str.lower() if x.dtype == "object" else x)
    
    #Stripping white space and remove ., from text column
    df[text_column] = df[text_column].str.strip()
    df[text_column] = df[text_column].str.replace('[.,]', '', regex=True)

    #More cleaning
    clean_text = df['text'].astype(str)
    clean_text = clean_text.apply(lambda x: re.sub(r'\S+@\S+', '', x)) #Removing emails
    clean_text = clean_text.apply(lambda x: re.sub(r'http\S+|www.\S+', '', x)) #Removing URLs
    clean_text = clean_text.apply(lambda x: re.sub(r'\d+', '', x)) #Removing digits
    clean_text = clean_text.apply(lambda x: re.sub(r'\b\w{1,2}\b|\b\w{13,}\b', '', x)) #Removing too big or too small words
    clean_text = clean_text.apply(lambda x: re.sub(r'[^\w\s]', '', x)) #Removing punctuation such as ? or !
    clean_text = clean_text.apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words])) #Removing stop words
    
    #Remove stopwords (and optionally lemmatize)
    def process_text(text):
        words = text.split()
        if lemmatize:
            words = [lemmatizer.lemmatize(w) for w in words]
        return ' '.join([w for w in words if w not in stop_words])

    df[f"{text_column}_clean"] = clean_text.apply(process_text)

    
    return df
