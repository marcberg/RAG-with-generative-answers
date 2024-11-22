from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from src.documents_and_embeddings.keywords import extract_keywords

# Fine-tune the model
def fine_tune_model(model, documents, use_keywords=True, batch_size=16, epochs=3):
    # Create training data with similarity labels
    train_examples = []
    for doc_id, doc_data in documents.items():
        text = doc_data["text"]
        if use_keywords:
            keywords = extract_keywords(text)
            keyword_text = " ".join(keywords)
            train_examples.append(InputExample(texts=[keyword_text, keyword_text], label=1.0))
        else:
            train_examples.append(InputExample(texts=[text, text], label=1.0))

    # DataLoader for training
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
    train_loss = losses.CosineSimilarityLoss(model)

    # Fine-tune the model
    model.fit(train_objectives=[(train_dataloader, train_loss)], 
              epochs=epochs, 
              output_path="fine_tuned_model")

    return SentenceTransformer("fine_tuned_model")