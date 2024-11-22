from src.documents_and_embeddings.keywords import extract_keywords
from src.document_retrieval.document_retrieval import density_filtered_query_faiss_index_with_adaptive_dbscan

def query_documents(query, include_score_below=0.7, use_keywords=False):

    # Dynamically retrieve relevant documents
    if use_keywords:
        keywords_query = extract_keywords(query)
        context_docs = density_filtered_query_faiss_index_with_adaptive_dbscan(keywords_query, include_score_below = include_score_below)
    else:
        context_docs = density_filtered_query_faiss_index_with_adaptive_dbscan(query, include_score_below = include_score_below)
    
    # Join the top document texts for context (limiting length per document)
    context = "\n".join(doc["text"] for doc in context_docs) 

    return context

