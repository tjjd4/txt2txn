import json
from src.utils import create_open_ai_client, load_schema, create_ollama_client, remove_think_tag_from_deepseek
from ollama import ChatResponse

# Load the JSON schemas for swap and simple transfer
swap_schema = load_schema("src/schemas/swap.json")
transfer_schema = load_schema("src/schemas/transfer.json")

# Initialize OpenAI client
# client = create_open_ai_client()
client = create_ollama_client()


def classify_transaction(transaction_text):
    # System message explaining the task
    system_message = {
        "role": "system",
        "content": "Determine if the following transaction text is for a token swap or a transfer. Use the appropriate schema to understand the transaction. Return '1' for transfer, '2' for swap, and '0' for neither. Do not output anything besides this number. If one number is classified for the output, make sure to omit the other two in your generated response."
    }

    # Messages to set up schema contexts
    swap_schema_message = {
        "role": "system",
        "content": "[Swap Schema] Token Swap Schema:\n" + json.dumps(swap_schema, indent=2)
    }
    transfer_schema_message = {
        "role": "system",
        "content": "[Transfer Schema] Simple Transfer/Send Schema:\n" + json.dumps(transfer_schema, indent=2)
    }

    # User message with the transaction text
    user_message = {"role": "user", "content": transaction_text}

    # Sending the prompt to ChatGPT
    # completion = client.chat.completions.create(
    #     model="gpt-3.5-turbo",
    #     messages=[
    #         system_message,
    #         swap_schema_message,
    #         transfer_schema_message,
    #         user_message,
    #     ]
    # )

    completion: ChatResponse = client.chat(
        model='deepseek-r1', 
        messages=[
            system_message,
            swap_schema_message,
            transfer_schema_message,
            user_message,
        ],
    )


    # Extracting and interpreting the last message from the completion
    response = remove_think_tag_from_deepseek(completion.message.content.strip())

    print("classification: ", response)
    return get_valid_response(response)


def get_valid_response(response):
    valid_responses = ["0", "1", "2"]
    found = None
    
    for valid in valid_responses:
        # Count the occurrences of each valid response in the string
        if response.count(valid) == 1:
            if found is not None:
                # If another valid response was already found, return 0
                return 0
            found = int(valid)  # Store the found valid response
    
    return found if found is not None else 0