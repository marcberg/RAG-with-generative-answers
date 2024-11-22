import os
from gpt4all import GPT4All

def get_gpt4all_model(model_filename='Meta-Llama-3.1-8B-Instruct-128k-Q4_0.gguf', n_token=2048):
    model_dir = "llm_models/"
    os.makedirs(model_dir, exist_ok=True) 

    if os.path.exists(os.path.join(model_dir, model_filename)):
        gpt4all_model = GPT4All(model_filename, 
                                model_path=model_dir, 
                                model_type="LLaMA2", 
                                allow_download=False, 
                                n_ctx=n_token)
    else:
        print("\nModel file not found. Downloading:")
        gpt4all_model = GPT4All(model_filename, 
                                model_path=model_dir, 
                                model_type="LLaMA2", 
                                allow_download=True, 
                                n_ctx=n_token)
        print("\n")


    return gpt4all_model