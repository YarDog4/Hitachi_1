import os
import pandas as pd

data_directory = os.getenv(r"DATASET_PATH")
contents = os.listdir(data_directory)

def load_labeled_dataset(data_directory: str):
    texts = [] #list to store text from each file
    category = [] #list to store each category
    index = [] #list to store each category
    index_counter = 0 #counter for unique row IDs

    category_index = {} #mapping category names to numeric labels

    #looping through each folder in teh directory
    for folder_name in sorted(os.listdir(data_directory)):
        path =  os.path.join(data_directory, folder_name)

        if os.path.isdir(path):
            #storing new category ID for each directory
            category_id = len(category_index)
            category_index[folder_name] = category_id

            #looping through each file in the folder
            for file_name in sorted(os.listdir(path)):
                file_path = os.path.join(path, file_name)

                if os.path.isfile(file_path):
                    with open(file_path, encoding='latin-1') as file:
                        text = file.read()

                        #this skips the header if there is one
                        header = text.find('\n\n')
                        if header != 1:
                            text = text[header:]
                        texts.append(text.strip())
                        category.append(category_id)
                        index.append(index_counter)
                        index_counter += 1
                        
    df = pd.DataFrame({'id': index, 'text': texts, 'category': category})
    print(f"✅ Loaded {len(df)} documents from {len(category_index)} categories")
    return df, category_index
