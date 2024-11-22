import os
import json
import numpy as np

import faiss
from sentence_transformers import SentenceTransformer

from src.documents_and_embeddings.save_and_load import save_index_and_documents, load_index_and_documents, save_model_metadata
from src.documents_and_embeddings.process_entry import process_entry_categories, process_entry_products, process_entry_stores
from src.documents_and_embeddings.keywords import extract_keywords
from src.documents_and_embeddings.fine_tune_model import fine_tune_model

def document_setup():
    data_folder = "data/"
    #files = ["categories.json", "products.json", "stores.json"]

    files = ["products.json"]

    save_folder = os.path.join(os.getcwd(), "saved_embeddings")
    os.makedirs(save_folder, exist_ok=True)  

    index_path = os.path.join(save_folder, "faiss_index.index")
    documents_path = os.path.join(save_folder, "documents.json")

    return data_folder, files, index_path, documents_path

def embedding_documents(SentenceTransformer_model="all-MiniLM-L6-v2", 
                        keep_current_index_and_documents=False,
                        use_keywords=False,
                        return_data=False):

    model = SentenceTransformer(SentenceTransformer_model)

    data_folder, files, index_path, documents_path = document_setup()

    if os.path.exists(index_path) and os.path.exists(documents_path) and keep_current_index_and_documents:
        print("loading!")
        index, documents, faiss_to_doc_id = load_index_and_documents(index_path=index_path, documents_path=documents_path)
    else:
        embedding_dim = len(model.encode(["Test"])[0])
        index = faiss.IndexFlatL2(embedding_dim)

        documents = {} 
        faiss_to_doc_id = [] 

        for file_name in files:
            print(file_name)
            with open(os.path.join(data_folder, file_name), "r") as file:
                data = json.load(file)
                length_data = len(data)
                print(f"Number of entries: {length_data}")
                for i, entry in enumerate(data):
                    print(f"Processing entry: {i+1}, {round(((i+1)/length_data)*100, 2)}%\t\t", end="\r")
                    if file_name == "categories.json":
                        text = process_entry_categories(entry)
                        unique_id = entry.get("id")

                    elif file_name == "products.json":
                        text = process_entry_products(entry)
                        unique_id = entry.get("sku")

                    elif file_name == "stores.json":
                        text = process_entry_stores(entry)
                        unique_id = entry.get("id")

                    document_id = f"{file_name.split('.')[0]}_{unique_id}"
                    documents[document_id] = {"text": text, "data": entry}

                    if use_keywords:                    
                        # Extract keywords for embedding
                        keywords = extract_keywords(text)
                        keyword_text = " ".join(keywords)

                        embedding = model.encode([keyword_text])[0].astype("float32")
                    else:
                        embedding = model.encode([text])[0].astype("float32")

                    index.add(np.array([embedding]).astype("float32"))
                    faiss_to_doc_id.append(document_id)

        # Save the FAISS index and documents metadata
        save_index_and_documents(index_path=index_path, 
                                 documents_path=documents_path, 
                                 index=index, 
                                 documents=documents, 
                                 faiss_to_doc_id=faiss_to_doc_id)
        
        save_model_metadata(SentenceTransformer_model)
    
    if return_data:
        return index, documents, faiss_to_doc_id, model




def embedding_documents_with_fine_tuned_model(SentenceTransformer_model="all-MiniLM-L6-v2", 
                                              keep_current_index_and_documents=False,
                                              use_keywords=False,
                                              return_data=False
                                              ):

    fine_tuned_model_path = 'fine_tuned_model'
        
    data_folder, files, index_path, documents_path = document_setup()

    if os.path.exists(index_path) and os.path.exists(documents_path) and os.path.exists(fine_tuned_model_path) and keep_current_index_and_documents:
        print("loading!")
        index, documents, faiss_to_doc_id = load_index_and_documents(index_path=index_path, documents_path=documents_path)
        model = SentenceTransformer(fine_tuned_model_path)
    else:
        embedding_dim = len(model.encode(["Test"])[0])
        index = faiss.IndexFlatL2(embedding_dim)

        documents = {} 
        faiss_to_doc_id = [] 

        for file_name in files:
            with open(os.path.join(data_folder, file_name), "r") as file:
                data = json.load(file)
                for entry in data:
                    if file_name == "categories.json":
                        text = process_entry_categories(entry)
                        unique_id = entry.get("id")

                    elif file_name == "products.json":
                        text = process_entry_products(entry)
                        unique_id = entry.get("sku")

                    elif file_name == "stores.json":
                        text = process_entry_stores(entry)
                        unique_id = entry.get("id")

                    document_id = f"{file_name.split('.')[0]}_{unique_id}"
                    documents[document_id] = {"text": text, "data": entry}

        print("Fine-tuning the model on the documents...")
        model = SentenceTransformer(SentenceTransformer_model)
        model = fine_tune_model(model, documents, use_keywords=use_keywords)
        
        model.save(fine_tuned_model_path)
        save_model_metadata(fine_tuned_model_path)

        # Embed documents and add them to FAISS index
        for doc_id, doc_data in documents.items():
            text = doc_data["text"]

            if use_keywords:
                # Extract keywords for embedding
                keywords = extract_keywords(text)
                keyword_text = " ".join(keywords)
                embedding = model.encode([keyword_text])[0].astype("float32")
            else:
                embedding = model.encode([text])[0].astype("float32")

            index.add(np.array([embedding]).astype("float32"))
            faiss_to_doc_id.append(doc_id)

        # Save the FAISS index and documents metadata
        save_index_and_documents(index_path=index_path, 
                                 documents_path=documents_path, 
                                 index=index, 
                                 documents=documents, 
                                 faiss_to_doc_id=faiss_to_doc_id)
        
    if return_data:
        return index, documents, faiss_to_doc_id, model