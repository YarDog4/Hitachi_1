# Hitachi Team 1 Visualizations Dashboard
This application is a Python based application that uses Streamlit to display the visualizations and uses Pinecone to tokenize, vectorize and categorize text from a chosen dataset. Below is a walkthrough for how to get the code, set it up, and run the Streamlit appication on your local machine

# Prerequisites
This application was developed using Python 3.13, so Python 3.13 is recommended to run and add to this application. You can download the latest version of Python [here](https://www.python.org/downloads/)

⭐You will need a computer as well (duh)⭐

# Getting started
Make sure you have created an empty folder or clone this repository to a location on your device where you can access. To clone this repository, you can use an IDE like VSCode, and put the commands shown below into your terminal.

1. **Fork the repository**

   This is a public repository, so you much fork this repository before cloning (next step!)
3. **Clone the repository**
    ```bash
    git clone https://github.com/<Github_Username>/Hitachi_1.git
    ```
4. **Setup a Python Virtual Environment (Optional but recommended)**

    **First, be sure to be in the Hitachi_1 directory** _(more information on how to tranverse to that below)_
    #### Windows (CMD or Powershell)
    ```bash
    python -m venv <your_venv_name>
    ```
    Activate the virtual environment:
    ```bash
    .\<your_venv_name>\Scripts\activate
    ```
    Deactivate the virtual environment when you are done with your environment:
    ```bash
    scripts\deactivate
    ```
    
    #### macOS
    ```bash
    python3 -m venv <your_venv_name>
    ```
    Activate the virtual environment:
    ```bash
    source <your_venv_name>/bin/activate
    ```
    Deactivate the virtual environment when you are done with your environment:
    ```bash
    source bin/deactivate
    ```
    #### Linux
    ```bash
    python3 -m venv <your_venv_name>
    ```
    Activate the virtual environment:
    ```bash
    source <your_venv_name>/bin/activate
    ```
    Deactivate the virtual environment when you are done with your environment:
    ```bash
    source bin/deactivate
    ```
5. **Install Python Dependencies**
    There is a requirements.txt file that contains all of the necessary dependencies for the application. With your virtual environment active, run this:

    #### Windows
    ```bash
    pip install -r requirements.txt
    ```
    #### macOS or Linux
     ```bash
    pip install --upgrade
    ```
     ```bash
    pip install -r requirements.txt
    ```
6. **Create a .env**
    In order to connect to Pinecone and use your data directories, you must create a .env that defines the environment variables this Python Application will use. You can create a Pinecone account [here](https://www.pinecone.io/) It should have the following variables:

    ```
    #PINECONE CREDENTIALS
    PINECONE_API_KEY="YOUR_PINECONE_API_KEY"
    PINECONE_ENVIRONMENT="YOUR_PINECONE_REGION" #Ex: us-east-1

    #DEFAULT INDEX NAME
    PINECONE_INDEX="PINECONE_INDEX_NAME" #Edit here
                                         #Can declare it here - code will create a new index if given index is nonexistent
                                         #Ex: "test"

    #DATASET PATH
    DATASET_PATH="FULL_PATH_TO_DATASET_FOLDER" #Edit here
    RELATIVE_PATH="RELATIVE_PATH_TO_DATASET_FOLDER" #Edit here
    ```
    There is an .env.example that will act as your template, just remember when you start developing, remove the .example at the end of this file so it is only named .env.

   **The .gitignore file will ignore any changes to this file, so you don't need to worry about leaking sensitive information, specifically the Pinecone API key.**

# How to run
Assuming you have the repository cloned and Python 3.13 installed, go into your home directory (should be Hitachi_1) like this:
```bash
cd Hitachi_1
```

and run:
```bash
python -m streamlit run Home.py
```

You will be prompted to enter in an email to receive streamlit emails. Press 'Enter' to bypass this.

# Additional Information
When first running the application, it may take a little bit to load the visualizations and features, depending on how big your dataset is. The defualt dataset this application uses is around 20,000 thousand files, so Pinecone vectorizing and embeddings take a long time. To combat this heavy overhead, this application caches your Pinecone vector embeddings and metadata. These files will be located in the ```../dataset/csv``` directory.


When asking for an article to be categorized, for the most accurate experience, ensure your inputs are descriptive and well-formed, as very short or vague articles may return lower-quality matches. 

The application is pretty intuitive, and there are descriptions for what everything does throughout the application. Additionally, there will be a video as an additional resource for users

🌟We hope you enjoy our application!🌟
