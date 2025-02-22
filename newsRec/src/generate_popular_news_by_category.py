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

def truncate_text(text, max_words):
    """Truncate text to a maximum number of words."""
    words = text.split()
    return ' '.join(words[:max_words])

def get_popular_test_file(test_file_path, train_behaviors_path, train_news_path, test_news_path, num_per_category=3):
    """
    load the test file, and get the popular news for each category according to training data.
    Args:
        test_file_path: Path to the test file.
        train_behaviors_path: Path to the behaviors file.
        train_news_path: Path to the news file.
    return:

    """

    with open(test_file_path, 'rb') as f:
        target_data = pickle.load(f)
    
    # Load news metadata
    train_news = pd.read_table(
        train_news_path,
        header=None,
        usecols=[0, 1, 2, 3, 4, 6, 7],
        quoting=csv.QUOTE_NONE,
        names=['id', 'category', 'subcategory', 'title', 'abstract', 'title_entities', 'abstract_entities']
    )
    train_news.fillna(' ', inplace=True)
    train_news_dict = train_news.set_index('id').to_dict(orient='index')
    # Load news metadata
    test_news = pd.read_table(
        test_news_path,
        header=None,
        usecols=[0, 1, 2, 3, 4, 6, 7],
        quoting=csv.QUOTE_NONE,
        names=['id', 'category', 'subcategory', 'title', 'abstract', 'title_entities', 'abstract_entities']
    )
    test_news.fillna(' ', inplace=True)
    test_news_dict = test_news.set_index('id').to_dict(orient='index')
    target_data["news_category"] = [test_news_dict[id]['category'] for id in target_data['news_ID']]

    train_categories = set(train_news['category'].unique())
    test_categories = test_news['category'].unique()
    categories = set(train_categories.union(test_categories))

    print(f'Number of categories: {len(categories)}')

    category_news = {key:[] for key in categories}
    # get popular news for each category
    behaviors = pd.read_table(train_behaviors_path, header=None, names=['impression_id', 'user', 'time', 'clicked_news', 'impressions'])
    news_freq = {}
    for clicked_list in behaviors['clicked_news'].fillna(' '):
        for news_id in clicked_list.split():
            news_freq[news_id] = news_freq.get(news_id, 0) + 1
    # get the popular news for each category
    # 1. assign most popular `num_per_category` category news to categories_in_train
    sorted_news = sorted(news_freq.items(), key=lambda x: x[1], reverse=True)
    for category in train_categories:
        category_news[category] = [truncate_text(str(train_news_dict[x[0]]['title']), 20) + " \t " + truncate_text(str(train_news_dict[x[0]]['abstract']), 50) for x in sorted_news if train_news_dict[x[0]]['category'] == category][:num_per_category]
    # 2. assign most popular `num_per_category` news to categories_not_in_train
    categories_not_in_train = [category for category in category_news if len(category_news[category]) == 0]
    for category in categories_not_in_train:
        category_news[category] = [truncate_text(str(train_news_dict[x[0]]['title']), 20) + " \t " + truncate_text(str(train_news_dict[x[0]]['abstract']), 50) for x in sorted_news[:num_per_category]]

    with open('/var/scratch/XXXXXXXX/data_attack/target_gpt/train_popular_category.pkl', 'wb') as f:
        pickle.dump(category_news, f)
    
    # get and save the popular news for each category in test data
    target_data["news_popuilar_per_category"] = [category_news[category] for category in target_data["news_category"]]
    with open('/var/scratch/XXXXXXXX/data_attack/target_gpt/target_test_output_popular_category.pkl', 'wb') as f:
        pickle.dump(target_data, f)




if __name__ == '__main__':
    set_seeds(seed=2025)


    target_dir_train = '/var/scratch/XXXXXXXX/data_attack/target_gpt/target_train_output.pkl'
    target_dir_test = '/var/scratch/XXXXXXXX/data_attack/target_gpt/target_test_output.pkl'

    train_behaviors_path = '/var/scratch/XXXXXXXX/newsRec/data/original/train/behaviors.tsv'
    train_news_path = '/var/scratch/XXXXXXXX/newsRec/data/original/train/news.tsv'
    test_behaviors_path = '/var/scratch/XXXXXXXX/newsRec/data/original/test/behaviors.tsv'
    test_news_path = '/var/scratch/XXXXXXXX/newsRec/data/original/test/news.tsv'

    print('Processing training file...')
    get_popular_test_file(target_dir_test, train_behaviors_path, train_news_path, test_news_path, num_per_category=3)

    # print('Processing testing file...')
    # process_target_file(target_dir_test, test_behaviors_path, test_news_path, target_dir_test_output, target_dir_test_output_fail, {"top": 100, "medium": 100, "bottom": 100})
    
     