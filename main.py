import argparse

from src.documents_and_embeddings.embedding_documents import embedding_documents, embedding_documents_with_fine_tuned_model
from src.qa.llm import get_gpt4all_model
from src.qa.conversation import interactive_conversation


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run embedding, conversation, or both.")
    parser.add_argument(
        "--task",
        choices=["all", "embedding", "conversation"],
        default="all",
        help="Choose to run 'all', only 'embedding', or only 'conversation'. Default is 'all'."
    )
    parser.add_argument(
        "--keep_index",
        type=bool,
        default=True,
        help="Set to False to overwrite the current index and documents. Default is True."
    )

    args = parser.parse_args()

    if args.task in ["all", "embedding"]:
        print("Running embedding...")
        embedding_documents(SentenceTransformer_model="all-MPNet-base-v2", keep_current_index_and_documents=args.keep_index)

    if args.task in ["all", "conversation"]:
        n_token=1024*2

        #model = 'wizardlm-13b-v1.2.Q4_0.gguf'
        #model = 'Meta-Llama-3.1-8B-Instruct-128k-Q4_0.gguf'
        #model = 'qwen2-1_5b-instruct-q4_0.gguf'
        #model = 'Meta-Llama-3-8B-Instruct.Q4_0.gguf'
        model = 'mistral-7b-instruct-v0.2.Q4_K_M.gguf'

        gpt4all_model = get_gpt4all_model(model_filename=model, n_token=n_token)
        interactive_conversation(gpt4all_model, n_token=n_token)


if __name__ == "__main__":
    main()