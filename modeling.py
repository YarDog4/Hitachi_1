import os
import pandas as pd

data_directory = r"C:\Users\yaren\Desktop\Hitachi\Hitachi_1\dataset\20_newsgroup"
contents = os.listdir(data_directory)

print(contents)

texts = [] #list to store text from each file
category = [] #list to store each category
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

#sanity checking
print(f'Number of texts: {len(texts)} \n')
print(f'Number of categories: {len(category)} \n')
print(f'Categories to Index Mapping: \n {category_index}')

#creating a dataframe

df = pd.DataFrame({
    'text': texts,
    'category': category
})

print(df)