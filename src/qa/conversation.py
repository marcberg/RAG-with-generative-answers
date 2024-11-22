from src.document_retrieval.querying import query_documents
from src.qa.prompt_generator import generate_prompt_with_history

def interactive_conversation(gpt4all_model, n_token):
    
    # Initialize conversation history
    conversation_history = []

    # Function to reset the conversation history
    def reset_conversation_history():
        nonlocal conversation_history
        conversation_history = []

    # Ensure conversation history is empty on first run
    reset_conversation_history()

    print("You can start asking questions. Type 'exit' to quit.\n")

    while True:
        # Step 1: Get input for the question
        user_query = input("Your question: ")

        # Exit condition
        if user_query.lower() == "exit":
            print("\nEnding conversation.")
            break


        while True:
            # Step 2: Get context for the query
            context = query_documents(user_query, use_keywords=True, include_score_below=1.0)
            
            # Display the context
            print(f"\nContext Retrieved:\n{context}\n")
            
            # Step 3: Allow user to decide to proceed or refine the query
            proceed_or_refine = input("Is the context okay? ('yes' to proceed / 'rewrite' to refine / 'exit' to exit): ").strip().lower()
            print("\n")

            if proceed_or_refine == "rewrite":
                # Allow user to refine the query
                user_query = input("Rewrite your query: ")
                print("You:", user_query)
            elif proceed_or_refine == "yes":
                break
            elif proceed_or_refine == "exit":
                print("Ending conversation.")
                return  # Exit the function completely
            else:
                print("Invalid input. Please type 'yes' to proceed, 'rewrite' to refine, or 'exit' to quit.")

        # Display the question
        print("You:\n", user_query)        

        # Step 4: Generate the prompt
        full_prompt = generate_prompt_with_history(
            query=user_query,
            context=context,
            conversation_history=conversation_history,
            max_tokens=n_token
        )

        # Generate response
        response = gpt4all_model.generate(
            full_prompt, 
            max_tokens=200, 
            temp=0.2, 
            top_k=10, 
            top_p=0.7
        )    

        # Step 5: Append the current query and response to conversation history
        conversation_history.append((user_query, response))
        print("AI Response:\n", response, flush=True)
        
        # Option to follow up or exit
        follow_up = input("\nWould you like to ask a follow-up question? (yes to continue / exit to quit): ").strip().lower()
        if follow_up == "exit":
            print("\nEnding conversation.")
            break
