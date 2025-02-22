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

def validate_news(news_data):
    """
    Validate news entries based on title and abstract.
    Args:
        news_data: DataFrame containing news data.
    Returns:
        A DataFrame indicating valid and invalid news.
    """
    news_data['is_valid'] = news_data['abstract'].apply(lambda x: isinstance(x, str) and len(x.strip()) >= 10)
    return news_data


def replace_invalid_news(news_data, sampled_news, category_mapping, news_dict):
    """
    Replace invalid news entries with resampled news.
    Args:
        news_data: Original news DataFrame.
        sampled_news: Dictionary with resampled news IDs by category.
        category_mapping: Mapping of news ID to popularity category.
        news_dict: Dictionary containing news metadata.
    Returns:
        Updated news DataFrame with invalid entries replaced.
    """
    invalid_indices = news_data[~news_data['is_valid']].index
    for idx in invalid_indices:
        news_id = news_data.loc[idx, 'id']
        category = category_mapping.get(news_id, None)
        if category and sampled_news[category]:
            new_id = sampled_news[category].pop(0)  # Use a resampled news ID
            news_data.loc[idx, 'id'] = new_id
            news_data.loc[idx, 'title'] = news_dict[new_id]['title']
            news_data.loc[idx, 'abstract'] = news_dict[new_id]['abstract']
    return news_data


def process_target_file(file_path, behaviors_path, news_path, sample_sizes):
    """
    Load the prepared file, and get new sampled files excluding the invalid news and prepared file.
    Args:
        file_path: Path to the prepared file.
        behaviors_path: Path to the behaviors file.
        news_path: Path to the news file.
        sample_sizes: Dictionary specifying sample sizes for popularity ranges.
    """
    # Load the original target data
    with open(file_path, 'rb') as f:
        target_data = pickle.load(f)
    target_data_cur = {
        "index": [],
        "top_index": [],
        "medium_index": [],
        "bottom_index": [],
        "news_ID": [],
        "news_title": [],
        "news_abstract": [],
    }
    # Load news metadata
    news = pd.read_table(
        news_path,
        header=None,
        usecols=[0, 1, 2, 3, 4, 6, 7],
        quoting=csv.QUOTE_NONE,
        names=['id', 'category', 'subcategory', 'title', 'abstract', 'title_entities', 'abstract_entities']
    )
    news.fillna(' ', inplace=True)
    news_dict = news.set_index('id').to_dict(orient='index')

    # Validate news
    def is_valid_abstract(abstract):
        return isinstance(abstract, str) and len(abstract.strip()) >= 10

    # Sample replacement news by popularity
    behaviors = pd.read_table(behaviors_path, header=None, names=['impression_id', 'user', 'time', 'clicked_news', 'impressions'])
    news_freq = {}
    for clicked_list in behaviors['clicked_news'].fillna(' '):
        for news_id in clicked_list.split():
            news_freq[news_id] = news_freq.get(news_id, 0) + 1

    sorted_news = sorted(news_freq.items(), key=lambda x: x[1], reverse=True)
    total_news = len(sorted_news)
    print(f'Total news number is: {total_news}')

    top_1_3 = [x[0] for x in sorted_news[:total_news // 3]]
    medium_1_3 = [x[0] for x in sorted_news[total_news // 3:2 * total_news // 3]]
    bottom_1_3 = [x[0] for x in sorted_news[2 * total_news // 3:]]

    # Ensure only news with valid abstracts are sampled
    top_1_3 = [id for id in top_1_3 if is_valid_abstract(news_dict.get(id, {}).get('abstract', ''))]
    medium_1_3 = [id for id in medium_1_3 if is_valid_abstract(news_dict.get(id, {}).get('abstract', ''))]
    bottom_1_3 = [id for id in bottom_1_3 if is_valid_abstract(news_dict.get(id, {}).get('abstract', ''))]

    existed_news_ID_set = set(target_data["news_ID"])
    top_1_3 = [id for id in top_1_3 if id not in existed_news_ID_set]
    medium_1_3 = [id for id in medium_1_3 if id not in existed_news_ID_set]
    bottom_1_3 = [id for id in bottom_1_3 if id not in existed_news_ID_set]
    # Generate sampled news with retries for unique resampling
    def sample_news_one_by_one(pool, size, seed):
        random.seed(seed)
        sampled = []
        while len(sampled) < size and pool:
            candidate = random.choice(pool)
            if candidate not in target_data['news_ID'] and candidate not in sampled:
                sampled.append(candidate)
                pool.remove(candidate)
        return sampled

    seed_offset = 2025
    sampled_news = {}
    for category, pool, size in zip(
        ["top", "medium", "bottom"], [top_1_3, medium_1_3, bottom_1_3], [sample_sizes["top"], sample_sizes["medium"], sample_sizes["bottom"]]
    ):
        sampled = sample_news_one_by_one(pool, size, 2025 + seed_offset)
        sampled_news[category] = sampled

    category_mapping = {id: "top" for id in top_1_3}
    category_mapping.update({id: "medium" for id in medium_1_3})
    category_mapping.update({id: "bottom" for id in bottom_1_3})

    target_data_cur['index'] = [idx for idx in range(0, 3*sample_sizes['top'])]
    target_data_cur['top_index'] = [idx for idx in range(0, sample_sizes['top'])]
    target_data_cur['medium_index'] = [idx for idx in range(sample_sizes['top'], sample_sizes['top'] + sample_sizes['medium'])]
    target_data_cur['bottom_index'] = [idx for idx in range(sample_sizes['top'] + sample_sizes['medium'], 3*sample_sizes['top'])]
    target_data_cur['news_ID'] = sampled_news['top'] + sampled_news['medium'] + sampled_news['bottom']
    target_data_cur['news_title'] = [news_dict[id]['title'] for id in target_data_cur['news_ID']]
    target_data_cur['news_abstract'] = [news_dict[id]['abstract'] for id in target_data_cur['news_ID']]
    
    output_path = '/var/scratch/XXXXXXXX/data_attack/target_gpt/target_train_output3000_new.pkl'
    # Save updated target data
    with open(output_path, 'wb') as f:
        pickle.dump(target_data_cur, f)

    print(f"Updated file saved to {output_path}")


if __name__ == '__main__':
    set_seeds(seed=2025)


    target_dir_train = '/var/scratch/XXXXXXXX/data_attack/target_gpt/target_train_output.pkl'
    target_dir_test = '/var/scratch/XXXXXXXX/data_attack/target_gpt/target_test_output.pkl'

    train_behaviors_path = '/var/scratch/XXXXXXXX/newsRec/data/original/train/behaviors.tsv'
    train_news_path = '/var/scratch/XXXXXXXX/newsRec/data/original/train/news.tsv'
    test_behaviors_path = '/var/scratch/XXXXXXXX/newsRec/data/original/test/behaviors.tsv'
    test_news_path = '/var/scratch/XXXXXXXX/newsRec/data/original/test/news.tsv'

    print('Processing training file...')
    process_target_file(target_dir_train, train_behaviors_path, train_news_path, {"top": 500, "medium": 500, "bottom": 500})

    # print('Processing testing file...')
    # process_target_file(target_dir_test, test_behaviors_path, test_news_path, target_dir_test_output, target_dir_test_output_fail, {"top": 100, "medium": 100, "bottom": 100})
    
    