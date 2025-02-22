from config import model_name
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
from utils import *
import pickle


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
    

def truncate_text(text, max_words):
    words = text.split()
    return " ".join(words[:max_words])

def convert_news_to_string(news):
    """
    Convert a news dictionary into a structured string.
    Args:
        news: A dictionary {"title": ..., "abstract": ...}.
    Returns:
        A formatted string representing the news.
    """
    return f" \nTitle: {news['title']} \nAbstract: {news['abstract']}"

def load_prompt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        prompt = file.read()

    prompt = prompt.replace("{", "{{").replace("}", "}}")
    # Restore valid placeholders like {original_news}
    prompt = prompt.replace("{{original_news}}", "{original_news}")
    return prompt

def merge_samples(old_path, new_path):
    with open(old_path, 'rb') as f:
        old_data = pickle.load(f)

    with open(new_path, 'rb') as f:
        new_data = pickle.load(f)

    old_len = len(old_data['index'])
    for key in old_data.keys():
        if key == 'index' or key == 'top_index' or key == 'medium_index' or key == 'bottom_index':
            old_data[key] = old_data[key] + [old_len + i for i in new_data[key]]
        else:
            old_data[key] = old_data[key] + new_data[key]

    with open(old_path, 'wb') as f:
        pickle.dump(old_data, f)
    print("Merged sample number: ", len(old_data['index']))
    print(f"Merge {old_path} and {new_path} successfully.")
    return old_data


if __name__ == '__main__':
    target_dir = "/var/scratch/XXXXXXXX/data_attack/target_gpt/"
    # rewrited_target_path = path.join(target_dir, 'rewrited_target_' + 'train' + "_" + diverse_mode + '.pkl')
    # rewrited_target_path_new = path.join(target_dir, '3000_new_rewrited_target_' + 'train' + "_" + diverse_mode + '.pkl')
    path_old = '/var/scratch/XXXXXXXX/data_attack/target_gpt/target_train_output.pkl'
    path_new = '/var/scratch/XXXXXXXX/data_attack/target_gpt/target_train_output3000_new.pkl'

    diverse_prompt_dict = {
        "writing_style": "prompt_writing_style.txt",
        "sentiment": "prompt_sentiment.txt",
        "persona": "prompt_persona.txt"
    }
    mode = ['writing_style', 'sentiment', 'persona']

    merge_samples(path_old, path_new)
    for diverse_mode in mode:
        rewrited_target_path = path.join(target_dir, 'rewrited_target_' + 'train' + "_" + diverse_mode + '.pkl')
        rewrited_target_path_new = path.join(target_dir, '3000_new_rewrited_target_' + 'train' + "_" + diverse_mode + '.pkl')
        merge_samples(rewrited_target_path, rewrited_target_path_new)

    # # ============== check the old data ==============
    # for diverse_mode in mode:
    #     rewrited_target_path_new = path.join(target_dir, '3000_new_rewrited_target_' + 'train' + "_" + diverse_mode + '.pkl')
    #     with open(rewrited_target_path_new, 'rb') as f:
    #         cur_rewrited_data = pickle.load(f)
    #     if len(cur_rewrited_data['index']) != 1500:
    #         print(f"Error: {diverse_mode} has {len(cur_rewrited_data['index'])} news.")
    #         prompt_path = '/var/scratch/XXXXXXXX/data_attack/prompts/' + diverse_prompt_dict[diverse_mode]
    #         # check the news
    #         idx = 0
    #         new_ID_set = set(target_data_path_new["news_ID"])
    #         rewrited_new_ID_set = set(cur_rewrited_data['news_ID'])
    #         failed_id = new_ID_set - rewrited_new_ID_set
    #         failed_id_idx = [target_data_path_new["news_ID"].index(id) for id in failed_id]
    #         for idx in failed_id_idx:
    #             news_id = target_data_path_new["news_ID"][idx]
    #             news_title = target_data_path_new["news_title"][idx]
    #             news_abstract = target_data_path_new["news_abstract"][idx]
    #             formatted_news = {
    #                 "title": truncate_text(news_title, 30),
    #                 "abstract": truncate_text(news_abstract, 60),
    #             }
    #             prompt_rewrite = load_prompt(prompt_path)
    #             original_news = convert_news_to_string(formatted_news)
    #             formatted_prompt = prompt_rewrite.format(original_news=original_news)
    #             formatted_prompt = formatted_prompt.replace("{{", "{").replace("}}", "}")
    #             from llm_api import gpt_chat
    #             llm_response = gpt_chat(formatted_prompt)
    #             # Validate and extract the JSON response
    #             rewrite_result = extract_json_from_response(llm_response)

    #             rewritten_titles = [value['title'] for value in rewrite_result.values()]
    #             rewritten_abstracts = [value['abstract'] for value in rewrite_result.values()]

    #             cur_rewrited_data['index'].append(cur_rewrited_data['index'][-1] + 1)
    #             cur_rewrited_data['news_ID'].insert(idx, news_id)
    #             cur_rewrited_data['news_title'].insert(idx, rewritten_titles)
    #             cur_rewrited_data['news_abstract'].insert(idx, rewritten_abstracts)
    #         if set(cur_rewrited_data['news_ID']) == set(target_data_path_new["news_ID"]):
    #             print(f"Success: {diverse_mode} has {len(cur_rewrited_data['index'])} news.")
    #             with open(rewrited_target_path_new, 'wb') as f:
    #                 pickle.dump(cur_rewrited_data, f)
    # # ============== check the old data ==============
    
    

    print(123)