import json
import faiss
import os

def save_index_and_documents(index_path, documents_path, index, documents, faiss_to_doc_id):
    # Save the FAISS index
    faiss.write_index(index, index_path)
    
    # Save documents and faiss_to_doc_id together in JSON
    with open(documents_path, "w") as f:
        json.dump({"documents": documents, "faiss_to_doc_id": faiss_to_doc_id}, f)
    

def load_index_and_documents(index_path, documents_path):
    # Load the FAISS index
    index = faiss.read_index(index_path)
    
    # Load documents and faiss_to_doc_id
    with open(documents_path, "r") as f:
        data = json.load(f)
        documents = data["documents"]
        faiss_to_doc_id = data["faiss_to_doc_id"]
    
    return index, documents, faiss_to_doc_id

def save_model_metadata(model_identifier):
    metadata = {"model": model_identifier}
    
    metadata_folder = os.path.join(os.getcwd(), "metadata")
    os.makedirs(metadata_folder, exist_ok=True)  
    
    metadata_path = f"{metadata_folder}\\model_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f)

def load_model_metadata():
    metadata_folder = os.path.join(os.getcwd(), "metadata")
    metadata_path = f"{metadata_folder}\\model_metadata.json"
    try:
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        print(f"Loaded metadata: {metadata}")
        return metadata["model"]
    except FileNotFoundError:
        raise ValueError(f"Metadata file not found for index: {metadata_path}")