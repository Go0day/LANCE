import pandas as pd
import swifter
import json
import math
from tqdm import tqdm
from os import path
from pathlib import Path
import random
from nltk.tokenize import word_tokenize
import numpy as np
import csv
import importlib
import torch

import pickle
from llm_api import *
import os
import argparse

def set_seeds(seed=2025):
    """
    Set all random seeds for reproducibility.
    Args:
        seed: The seed value to set (default: 2025).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f"Seeds set to {seed} for reproducibility.")


def iter_target_news(target_path, group_keys=["top_index", "medium_index", "bottom_index"]):
    """
    Iterate through target news in the specified groups and format each news.
    Args:
        target_path: Path to the target file (e.g., target_train.pkl or target_test.pkl).
        group_keys: List of keys specifying the groups to iterate over (default: top, medium, bottom).
    Yields:
        A dictionary {"title": truncated_title, "abstract": truncated_abstract} for each news.
    """
    # Load the target file
    with open(target_path, 'rb') as f:
        target_data = pickle.load(f)

    # Define truncation limits
    max_title_words = 30
    max_abstract_words = 60

    # Helper function to truncate text
    def truncate_text(text, max_words):
        words = text.split()
        return " ".join(words[:max_words])

    # Iterate through the specified groups
    for group_key in group_keys:
        if group_key not in target_data:
            print(f"Warning: {group_key} not found in target data.")
            continue

        for index in target_data[group_key]:
            # Retrieve and format the news
            print("News ID is: " + str(target_data["news_ID"][index]))
            print("News title is: " + target_data["news_title"][index])
            title = target_data["news_title"][index]
            abstract = target_data["news_abstract"][index]

            formatted_news = {
                "title": truncate_text(title, max_title_words),
                "abstract": truncate_text(abstract, max_abstract_words),
            }

            # Yield the formatted news
            yield formatted_news

def load_prompt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        prompt = file.read()

    prompt = prompt.replace("{", "{{").replace("}", "}}")
    # Restore valid placeholders like {original_news}
    prompt = prompt.replace("{{original_news}}", "{original_news}")
    return prompt

def convert_news_to_string(news):
    """
    Convert a news dictionary into a structured string.
    Args:
        news: A dictionary {"title": ..., "abstract": ...}.
    Returns:
        A formatted string representing the news.
    """
    return f" \nTitle: {news['title']} \nAbstract: {news['abstract']}"

import json

def extract_json_from_response(response):
    """
    Extract and parse the JSON content from an LLM response string.
    Args:
        response: The LLM response string containing JSON data.
    Returns:
        A Python dictionary parsed from the JSON content.
    """
    try:
        # Check for a "Sorry" response or invalid content
        if "sorry" in response.lower():
            raise ValueError("LLM returned an apologetic response.")

        # Remove backticks and `json` tag if present
        json_start = response.find('{')
        json_end = response.rfind('}')
        if json_start == -1 or json_end == -1:
            raise ValueError("No valid JSON content found in the response.")

        # Extract and parse the JSON string
        json_content = response[json_start:json_end + 1]
        return json.loads(json_content)

    except (json.JSONDecodeError, ValueError) as e:
        print(f"Failed to parse JSON: {e}")
        return None


def rewrite_process(news, prompt_path, api_mode='gpt-4o', max_retries=5):
    """
    Rewrite a news dictionary using a prompt loaded from a file.
    Args:
        news: A dictionary {"title": ..., "abstract": ...}.
        prompt_path: Path to the text file containing the prompt template.
    Returns:
        A modified news dictionary with the rewritten content.
    """
    # Load the prompt template
    prompt_rewrite = load_prompt(prompt_path)

    # Convert the news dictionary to a string
    original_news = convert_news_to_string(news)

    # Inject the original news into the prompt
    formatted_prompt = prompt_rewrite.format(original_news=original_news)
    formatted_prompt = formatted_prompt.replace("{{", "{").replace("}}", "}")
    # Example rewriting logic (can be replaced with an actual LLM call)
    for attempt in range(max_retries):
        try:
            # Call the LLM API
            llm_response = gpt_chat(formatted_prompt, api_mode=api_mode)

            # Validate and extract the JSON response
            rewrite_result = extract_json_from_response(llm_response)
            if rewrite_result is None:
                print(llm_response)
                raise ValueError("Invalid response: JSON extraction failed.")

            # Return successfully extracted and validated result
            return rewrite_result

        except ValueError as e:
            print(f"Validation Error: {e}. Retrying {attempt + 1}/{max_retries}...")

        except Exception as e:
            print(f"Unexpected Error: {e}. Retrying {attempt + 1}/{max_retries}...")
            time.sleep(2 * attempt)
    # Fallback if all retries fail
    print("Failed to rewrite news after multiple attempts. Returning fallback.")
    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Set configuration for the model")
    parser.add_argument('--target_trainORtest', type=str, default='train',
                        help='target file name, target_train.pkl or target_test.pkl')
    parser.add_argument('--diverse_mode', type=str, default='sentiment',
                        help='Select one from `writing_style`, `sentiment` and `persona`')

    # Parse arguments
    args = parser.parse_args()

    set_seeds(seed=2025)

    diverse_prompt_dict = {
        "writing_style": "prompt_writing_style.txt",
        "sentiment": "prompt_sentiment.txt",
        "persona": "prompt_persona.txt"
    }

    train_or_test = {
        "train": "target_train.pkl",
        "test": "target_test.pkl"
    }



    target_path = "/var/scratch/XXXXXXXX/data_attack/target_gpt/target_train_output3000_new.pkl"
    prompts_path = '/var/scratch/XXXXXXXX/data_attack/prompts/' + diverse_prompt_dict[args.diverse_mode]

    with open(target_path, 'rb') as f:
        target_data = pickle.load(f)

    rewrited_news_collection = {
        "index": [],
        "top_index": [],
        "medium_index": [],
        "bottom_index": [],
        "news_ID": [],
        "news_title": [],
        "news_abstract": [],
    }

    failed_news = []  # To collect news IDs, titles, and indices for failed cases

    for news in iter_target_news(target_path):
        try:
            rewrited_news = rewrite_process(news, prompts_path, api_mode='gpt-4o')
            if not rewrited_news:
                # Log failed news for future comparison
                news_index = target_data["index"][len(rewrited_news_collection["index"])]
                news_id = target_data["news_ID"][news_index]
                news_title = target_data["news_title"][news_index]

                failed_news.append({"index": news_index, "news_ID": news_id, "news_title": news_title})
                print(f"Failed to rewrite news (index: {news_index}, ID: {news_id}).")
            else:
                # Successfully rewritten news
                news_index = target_data["index"][len(rewrited_news_collection["index"])]
                news_id = target_data["news_ID"][news_index]

                # Extract rewritten titles and abstracts
                rewritten_titles = [value['title'] for value in rewrited_news.values()]
                rewritten_abstracts = [value['abstract'] for value in rewrited_news.values()]

                # Append to the rewritten collection
                rewrited_news_collection["index"].append(news_index)
                rewrited_news_collection["news_ID"].append(news_id)
                rewrited_news_collection["news_title"].append(rewritten_titles)
                rewrited_news_collection["news_abstract"].append(rewritten_abstracts)
        except Exception as e:
            print(f"Failed to rewrite news: {e}")
    # Save the rewritten news collection
    target_dir = "/var/scratch/XXXXXXXX/data_attack/target_gpt/"
    rewrited_target_path = path.join(target_dir, '3000_new_rewrited_target_' + args.target_trainORtest + "_" + args.diverse_mode + '.pkl')
    with open(rewrited_target_path, 'wb') as f:
        pickle.dump(rewrited_news_collection, f)
    print(f"Rewritten news collection saved to {rewrited_target_path}.")
    success_len = len(rewrited_news_collection['index'])
    print(f"Successfully {args.diverse_mode}, rewritten {success_len} news.")


