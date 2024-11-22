# prompt_generator.py
import tiktoken

def generate_prompt_with_history(query, context, conversation_history=None, max_tokens=2048):
    if conversation_history is None:
        conversation_history = []  # Initialize if not provided

    tokenizer = tiktoken.get_encoding("cl100k_base") 
    initial_context = "I'm asking about our products, their categories, and our stores. You are a helpful AI assistant that answers."

    # Build conversation history text
    history_text = "\n---\n".join([f"Question: {q}\nAnswer: {a}" for q, a in conversation_history])

    # Create the full prompt
    full_prompt = f"{initial_context}\n{history_text}\nContext: {context}\nQuestion: {query}\nAnswer:"

    # Ensure the prompt length doesn't exceed max_tokens
    tokens = tokenizer.encode(full_prompt)
    if len(tokens) > max_tokens:
        history_tokens = tokenizer.encode(history_text)
        while len(tokens) > max_tokens and len(history_tokens) > 0:
            history_text = "\n---\n".join(history_text.split("\n---\n")[1:])  # Remove oldest exchanges
            history_tokens = tokenizer.encode(history_text)
            full_prompt = f"{initial_context}\n{history_text}\nContext: {context}\nQuestion: {query}\nAnswer:"
            tokens = tokenizer.encode(full_prompt)

        context_tokens = tokenizer.encode(context)
        while len(tokens) > max_tokens and len(context_tokens) > 0:
            context = context[: int(len(context) * 0.9)]  # Trim context
            context_tokens = tokenizer.encode(context)
            full_prompt = f"{initial_context}\n{history_text}\nContext: {context}\nQuestion: {query}\nAnswer:"
            tokens = tokenizer.encode(full_prompt)

    return full_prompt
