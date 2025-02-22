from openai import OpenAI
import time
import os

def gpt_chat(User_message, api_mode='llama3', stop='12', max_retries=3):
    client = OpenAI(
        api_key="sk-xxx",
        base_url="https://xxx",
    )

    for attempt in range(max_retries):
        try:
            # Create the message payload
            our_messages = [{'role': 'user', 'content': User_message}]

            # Make the API call
            response = client.chat.completions.create(messages=our_messages, model=api_mode)
            
            # Process and return the response
            llm_response = response.choices[0].message.content
            # print("User Message:", User_message)
            return llm_response
        except Exception as e:
            print(f"Error on attempt {attempt + 1}: {e}")
            time.sleep(2 ** attempt)
    
    print("Failed to get response from LLM after multiple attempts.")
    return "Sorry, I couldn't process your request at this time."

# Test the function
# print(gpt_chat(User_message="hello, how are you?", api_mode="llama3"))