from sentence_transformers import SentenceTransformer
import os
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import numpy as np

from src.documents_and_embeddings.save_and_load import load_model_metadata, load_index_and_documents
from src.documents_and_embeddings.embedding_documents import document_setup

def density_filtered_query_faiss_index_with_adaptive_dbscan(query, include_score_below=0.7, max_k=100, min_samples=1):

    _, _, index_path, documents_path = document_setup()
    index, documents, faiss_to_doc_id = load_index_and_documents(index_path=index_path, documents_path=documents_path)

    try:
        model_name = load_model_metadata()
    except FileNotFoundError:
        raise ValueError(f"Metadata file not found for model")
    
    model = SentenceTransformer(model_name)
        
    query_embedding = model.encode([query])[0].astype("float32")

    # Perform FAISS search with a large initial k
    distances, indices = index.search(np.array([query_embedding]), max_k)

    # Filter out invalid indices
    valid_distances = []
    valid_indices = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx != -1:  # Only include valid entries
            valid_distances.append(dist)
            valid_indices.append(idx)

    X = np.array(valid_distances).reshape(-1, 1)

    # Test different k values
    k_values = range(2, 11)
    eps_values = []
    silhouette_scores = []

    for k in k_values:
        # Compute nearest neighbors distances
        neighbors = NearestNeighbors(n_neighbors=k)
        neighbors_fit = neighbors.fit(X)
        distances, indices = neighbors_fit.kneighbors(X)
        
        # Sort distances and find the elbow point
        distances = np.sort(distances[:, k - 1], axis=0)
        kneedle = KneeLocator(range(len(distances)), distances, curve="convex", direction="increasing")
        eps = distances[kneedle.elbow] if kneedle.elbow else None
        eps_values.append(eps)
        
        # Compute DBSCAN with the found eps and calculate silhouette score
        if eps:
            db = DBSCAN(eps=eps, min_samples=k).fit(X)
            labels = db.labels_
            if len(set(labels)) > 1:  # Avoid silhouette score on single clusters
                score = silhouette_score(X, labels)
                silhouette_scores.append((k, eps, score))

    _, best_eps, _ = max(silhouette_scores, key=lambda x: x[2])
    
    valid_distances = np.array(valid_distances).reshape(-1, 1)  # Reshape for DBSCAN
    clustering = DBSCAN(eps=best_eps, min_samples=min_samples).fit(valid_distances)

    # Get the labels and find the most relevant cluster
    labels = clustering.labels_
    print(best_eps) # REMOVE
    print(valid_distances) # REMOVE
    print(labels) # REMOVE
    unique_labels= np.unique(labels)

    # Find the label with closest match to query
    main_cluster_label = np.min(unique_labels)

    # Filter results based on the main cluster label or score
    results = [
        documents[faiss_to_doc_id[valid_indices[i]]]
        for i, (label, distance) in enumerate(zip(labels, valid_distances))
        if label == main_cluster_label or distance[0] < include_score_below
    ]

    return results