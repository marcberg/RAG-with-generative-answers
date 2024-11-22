import os 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

# Download NLTK stopwords if not already done
import nltk
from nltk.data import find

# Set a custom directory for NLTK resources
NLTK_DATA_DIR = os.path.expanduser("~/nltk_data")

def download_nltk_resource(resource_name, nltk_data_dir=NLTK_DATA_DIR):
    # Ensure the custom directory exists
    if not os.path.exists(nltk_data_dir):
        os.makedirs(nltk_data_dir)

    # Add the directory to NLTK's search path
    if nltk_data_dir not in nltk.data.path:
        nltk.data.path.append(nltk_data_dir)

    # Check if the resource directory or file exists
    resource_path = os.path.join(nltk_data_dir, "corpora" if resource_name == "stopwords" else "tokenizers", resource_name)
    if not os.path.exists(resource_path):
        print(f"Downloading resource '{resource_name}' to '{nltk_data_dir}'...")
        nltk.download(resource_name, download_dir=nltk_data_dir)

# Check and download resources

def extract_keywords(text):
    download_nltk_resource('stopwords')
    download_nltk_resource('punkt')
    download_nltk_resource('punkt_tab')
    
    stop_words = set(stopwords.words('english'))  # Common "basic" words
    # Remove special characters and convert to lowercase
    cleaned_text = re.sub(r'[^a-zA-Z\s]', '', text).lower()
    words = word_tokenize(cleaned_text)
    keywords = [word for word in words if word not in stop_words]
    return keywords
