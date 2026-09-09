# Prework: Testing AWS Bedrock Component to Ensure Connection
import boto3
import json

def call_claude_46(prompt_text):
    # 1. Initialize the AWS Bedrock Runtime client
    # Note: Use 'us-east-1' since this is where I verfied access
    client = boto3.client(service_name='bedrock-runtime', region_name='us-east-1')

    # 2. Define the Model ID for Claude Sonnet 4.6
    # Using the Inference Profile ID for better reliability
    model_id = "us.anthropic.claude-sonnet-4-6"

    # 3. Construct the payload (Anthropic Messages API format)
    native_request = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 512,
        "temperature": 0.0, # Setting to 0.0 for clinical consistency
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt_text}],
            }
        ],
    }

    # 4. Invoke the model
    try:
        response = client.invoke_model(
            modelId=model_id,
            body=json.dumps(native_request)
        )

        # 5. Decode and print the response
        model_response = json.loads(response["body"].read())
        return model_response["content"][0]["text"]

    except Exception as e:
        return f"Error invoking model: {str(e)}"

# --- TEST THE CONNECTION ---
if __name__ == "__main__":
    test_prompt = "Hello Claude 4.6. I am building a suicide risk screener capstone. Are you ready to assist with clinical reasoning?"
    print("Sending prompt to Claude Sonnet 4.6...")
    result = call_claude_46(test_prompt)
    print(f"\nClaude's Response:\n{result}")

# -------------------------------------------------------------------------
# INTERNET USAGE CITATION
# Tool: YouTube
# Usage: Used as Instruction to test my connection to AWS Bedrock
# Citation: https://www.youtube.com/watch?v=0qW0dGF6tXA "How to call Llama 3.1 through AWS Bedrock using Python by Woyera AI"
# -------------------------------------------------------------------------