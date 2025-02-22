# This is a function to get the attack performance among four metrics
#       {
    #     "Exposure": exposure,
    #     "Boost Success Rate": boost_success_rate,
    #     "Average Predicted Rank": average_predicted_rank,
    #     "Hit Ratio@K": hit_ratio,
    #   }


import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import torch
from config import model_name
from torch.utils.data import Dataset, DataLoader
from os import path
import sys
import pandas as pd
from ast import literal_eval
import importlib
from multiprocessing import Pool
import os
from utils import pretrained_encode_bert, pretrained_encode_glove, pretrained_encode_llama
from utils import save_news_dataset, load_news_dataset
import ast
import pickle
from transformers import BertTokenizer
import copy
import matplotlib.pyplot as plt

import evaluate

# Load the standard library module (e.g., random)
bleu = evaluate.load("bleu")
perplexity = evaluate.load("perplexity", module_type="metric")

try:
    Model = getattr(importlib.import_module(f"model.{model_name}"), model_name)
    config = getattr(importlib.import_module('config'), f"{model_name}Config")
except AttributeError:
    print(f"{model_name} not included!")
    exit()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

bert_path = "/var/scratch/XXXXXXXX/llms/bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(bert_path, legacy=False, clean_up_tokenization_spaces=True)


def should_display_progress():
    return sys.stdout.isatty()


def value2rank(d):
    values = list(d.values())
    ranks = [sorted(values, reverse=True).index(x) for x in values]
    return {k: ranks[i] + 1 for i, k in enumerate(d.keys())}

class NewsDataset(Dataset):
    """
    Load news for evaluation.
    """
    def __init__(self, news_path):
        super(NewsDataset, self).__init__()
        self.news_parsed = pd.read_table(
            news_path,
            usecols=['id'] + config.dataset_attributes['news'],
            converters={
                attribute: literal_eval
                for attribute in set(config.dataset_attributes['news']) & set([
                    'title_entities', 'abstract_entities'
                ])
            })

        self.news2dict = self.news_parsed.to_dict('index')

        for key1 in self.news2dict.keys():
            keys_to_iterate = list(self.news2dict[key1].keys())
            for key2 in keys_to_iterate:
                if key2 in ['title', 'abstract']:
                    self.news2dict[key1][key2] = ast.literal_eval(self.news2dict[key1][key2])
                    assert torch.tensor(self.news2dict[key1][key2]['input_ids']).shape == torch.tensor(self.news2dict[key1][key2]['attention_mask']).shape
                    self.news2dict[key1][key2] = torch.cat([torch.tensor(self.news2dict[key1][key2]['input_ids']).unsqueeze(0), torch.tensor(self.news2dict[key1][key2]['attention_mask']).unsqueeze(0)], dim=0)
                elif type(self.news2dict[key1][key2]) != str:
                    self.news2dict[key1][key2] = torch.tensor(
                        self.news2dict[key1][key2])

    def __len__(self):
        return len(self.news2dict)

    def __getitem__(self, idx):
        item = self.news2dict[idx]
        return item

class UserDataset(Dataset):
    """
    Load users for evaluation, duplicated rows will be dropped
    """
    def __init__(self, behaviors_path, user2int_path):
        super(UserDataset, self).__init__()
        self.behaviors = pd.read_table(behaviors_path,
                                       header=None,
                                       usecols=[1, 3],
                                       names=['user', 'clicked_news'])
        # self.behaviors.clicked_news.fillna(' ', inplace=True)
        self.behaviors['clicked_news'] = self.behaviors['clicked_news'].fillna(' ')
        self.behaviors.drop_duplicates(inplace=True)
        user2int = dict(pd.read_table(user2int_path).values.tolist())
        user_total = 0
        user_missed = 0
        for row in self.behaviors.itertuples():
            user_total += 1
            if row.user in user2int:
                self.behaviors.at[row.Index, 'user'] = user2int[row.user]
            else:
                user_missed += 1
                self.behaviors.at[row.Index, 'user'] = 0
        if model_name == 'LSTUR' or model_name == 'LSTURlinear':
            print(f'User miss rate: {user_missed/user_total:.4f}')

    def __len__(self):
        return len(self.behaviors)

    def __getitem__(self, idx):
        row = self.behaviors.iloc[idx]
        item = {
            "user":
            row.user,
            "clicked_news_string":
            row.clicked_news,
            "clicked_news":
            row.clicked_news.split()[:config.num_clicked_news_a_user]
        }
        item['clicked_news_length'] = len(item["clicked_news"])
        repeated_times = config.num_clicked_news_a_user - len(
            item["clicked_news"])
        assert repeated_times >= 0
        item["clicked_news"] = ['PADDED_NEWS'
                                ] * repeated_times + item["clicked_news"]

        return item


class BehaviorsDataset(Dataset):
    """
    Load behaviors for evaluation, (user, time) pair as session
    """
    def __init__(self, behaviors_path):
        super(BehaviorsDataset, self).__init__()
        self.behaviors = pd.read_table(behaviors_path,
                                       header=None,
                                       usecols=range(5),
                                       names=[
                                           'impression_id', 'user', 'time',
                                           'clicked_news', 'impressions'
                                       ])
        # self.behaviors.clicked_news.fillna(' ', inplace=True)
        self.behaviors['clicked_news'] = self.behaviors['clicked_news'].fillna(' ')
        self.behaviors.impressions = self.behaviors.impressions.str.split()

    def __len__(self):
        return len(self.behaviors)

    def __getitem__(self, idx):
        row = self.behaviors.iloc[idx]
        item = {
            "impression_id": row.impression_id,
            "user": row.user,
            "time": row.time,
            "clicked_news_string": row.clicked_news,
            "impressions": row.impressions
        }
        return item



def replace_rewrite_news(tmp_news_dataset, rewrited_news_collection, position):
    for ori_idx in tmp_news_dataset.news2dict:
        for re_idx in rewrited_news_collection['index']:
            if tmp_news_dataset.news2dict[ori_idx]['id'] == rewrited_news_collection['news_ID'][re_idx]:
                tokenized_title = tokenizer(rewrited_news_collection['news_title'][re_idx][position].lower(), max_length=config.num_words_title,padding='max_length', truncation=True)
                tmp_news_dataset.news2dict[ori_idx]['title'] = torch.cat([torch.tensor(tokenized_title['input_ids']).unsqueeze(0), torch.tensor(tokenized_title['attention_mask']).unsqueeze(0)], dim=0)
                if "abstract" in tmp_news_dataset.news2dict[0].keys():
                    tokenized_abstract = tokenizer(rewrited_news_collection['news_abstract'][re_idx][position].lower(), max_length=config.num_words_abstract,padding='max_length', truncation=True)
                    tmp_news_dataset.news2dict[ori_idx]['abstract'] = torch.cat([torch.tensor(tokenized_abstract['input_ids']).unsqueeze(0), torch.tensor(tokenized_abstract['attention_mask']).unsqueeze(0)], dim=0)
    return tmp_news_dataset

@torch.no_grad()
def get_news2vector(news_dataset, model):
    news_dataloader = DataLoader(news_dataset,
                                 batch_size=config.batch_size * 16,
                                 shuffle=False,
                                 num_workers=config.num_workers,
                                 drop_last=False,
                                 pin_memory=True)
    news2vector = {}
    progress = news_dataloader
    if should_display_progress():
        progress = tqdm(progress, desc="Calculating vectors for news")
    for minibatch in progress:
        news_ids = minibatch["id"]
        if any(id not in news2vector for id in news_ids):
            news_vector = model.get_news_vector(minibatch)
            for id, vector in zip(news_ids, news_vector):
                if id not in news2vector:
                    news2vector[id] = vector

    news2vector['PADDED_NEWS'] = torch.zeros(
        list(news2vector.values())[0].size())
    
    return news2vector

@torch.no_grad()
def get_user2vector(user_dataset, news2vector, model):
    user_dataloader = DataLoader(user_dataset,
                                 batch_size=config.batch_size * 16,
                                 shuffle=False,
                                 num_workers=config.num_workers,
                                 drop_last=False,
                                 pin_memory=True)

    user2vector = {}
    progress = user_dataloader
    if should_display_progress():
        progress = tqdm(progress, desc="Calculating vectors for users")
    for minibatch in progress:
        user_strings = minibatch["clicked_news_string"]
        if any(user_string not in user2vector for user_string in user_strings):
            clicked_news_vector = torch.stack([
                torch.stack([news2vector[x].to(device) for x in news_list],
                            dim=0) for news_list in minibatch["clicked_news"]
            ],
                                              dim=0).transpose(0, 1)
            if model_name == 'LSTUR' or model_name == 'LSTURlinear' or model_name== 'LSTURbert':
                user_vector = model.get_user_vector(
                    minibatch['user'], minibatch['clicked_news_length'],
                    clicked_news_vector)
            else:
                user_vector = model.get_user_vector(clicked_news_vector)
            for user, vector in zip(user_strings, user_vector):
                if user not in user2vector:
                    user2vector[user] = vector

    return user2vector


@torch.no_grad()
def calculate_metrics_old(news_id, old_user2vector, old_news2vector, new_user2vector, new_news2vector, batch_size=512, K=[10, 20, 50, 100]):
    """
    Calculate metrics for evaluating rewritten news performance for multiple K values.
    
    Args:
        news_id: List of target news IDs.
        old_user2vector: Dictionary of user vectors for the original dataset (on GPU).
        old_news2vector: Dictionary of news vectors for the original dataset (on GPU).
        new_user2vector: Dictionary of user vectors for the rewritten dataset (on GPU).
        new_news2vector: Dictionary of news vectors for the rewritten dataset (on GPU).
        batch_size: Number of users to process in a single batch.
        K: List of top-K values for evaluation.
    
    Returns:
        dict: Dictionary containing computed metrics for each K.
    """
    
    # Convert news2vector and user2vector to tensors
    old_news_ids = list(old_news2vector.keys())
    # new_news_ids = list(new_news2vector.keys())
    target_news_index = torch.tensor([old_news_ids.index(n_id) for n_id in news_id]).to(device)
    
    old_news2vector = {k: v.to(device) for k, v in old_news2vector.items()}
    new_news2vector = {k: v.to(device) for k, v in new_news2vector.items()}
    
    old_user2vector = {k: v.to(device) for k, v in old_user2vector.items()}
    new_user2vector = {k: v.to(device) for k, v in new_user2vector.items()}
    
    old_news_vectors = torch.stack(list(old_news2vector.values()))  # Shape: (num_news, dim)
    new_news_vectors = torch.stack(list(new_news2vector.values()))  # Shape: (num_news, dim)

    old_user_vectors = torch.stack(list(old_user2vector.values()))  # Shape: (num_users, dim)
    new_user_vectors = torch.stack(list(new_user2vector.values()))  # Shape: (num_users, dim)

    num_users = old_user_vectors.shape[0]
    num_targets = len(news_id)

    # Initialize metrics
    exposure_counts = {k: 0 for k in K}
    hit_counts = {k: 0 for k in K}
    total_rank_improvement = 0
    total_old_ranks = []
    total_new_ranks = []

    # Process users in batches
    for start in tqdm(range(0, num_users, batch_size), desc="Processing Users", unit="batch"):
        end = min(start + batch_size, num_users)
        old_batch_user_vectors = old_user_vectors[start:end]
        new_batch_user_vectors = new_user_vectors[start:end]
        
        # Compute scores for all news
        old_scores = torch.matmul(old_batch_user_vectors, old_news_vectors.T)  # Shape: (batch_size, num_news)
        new_scores = torch.matmul(new_batch_user_vectors, new_news_vectors.T)  # Shape: (batch_size, num_news)

        # Get ranks for all target news
        old_ranks = torch.argsort(-old_scores, dim=1)  # Shape: (batch_size, num_news)
        new_ranks = torch.argsort(-new_scores, dim=1)  # Shape: (batch_size, num_news)
        old_ranks_indices = torch.argsort(old_ranks, dim=1)[:, target_news_index]  # Shape: (batch_size, num_targets)
        new_ranks_indices = torch.argsort(new_ranks, dim=1)[:, target_news_index]  # Shape: (batch_size, num_targets)
        
        # Compute metrics
        rank_improvement = (old_ranks_indices.float() - new_ranks_indices.float()).sum(dim=1)
        total_rank_improvement += rank_improvement.sum().item()
        
        total_old_ranks.extend(old_ranks_indices.cpu().numpy())
        total_new_ranks.extend(new_ranks_indices.cpu().numpy())

        for k in K:
            exposure_counts[k] += (new_ranks[:, :k].unsqueeze(2) == target_news_index).any(dim=1).float().sum().item()
            hit_counts[k] += (new_ranks_indices < k).float().sum().item()

    # Final metrics
    total_old_ranks = np.concatenate(total_old_ranks)
    total_new_ranks = np.concatenate(total_new_ranks)

    num_users = len(total_old_ranks)
    exposure = [exposure_counts[k] / num_users for k in K]
    hit_ratio = [hit_counts[k] / (num_users * num_targets) for k in K]
    boost_success_rate = total_rank_improvement / (num_users * num_targets)
    average_predicted_rank = total_new_ranks.mean()

    return {
        "Exposure": exposure,
        "Boost Success Rate": boost_success_rate,
        "Average Predicted Rank": average_predicted_rank,
        "Hit Ratio@K": hit_ratio,
    }

@torch.no_grad()
def calculate_metrics(news_id, old_user2vector, old_news2vector, new_user2vector, new_news2vector, batch_size=512, K=[10, 20, 50, 100]):
    """
    Calculate metrics for evaluating rewritten news performance for multiple K values.
    
    Args:
        news_id: List of target news IDs.
        old_user2vector: Dictionary of user vectors for the original dataset (on GPU).
        old_news2vector: Dictionary of news vectors for the original dataset (on GPU).
        new_user2vector: Dictionary of user vectors for the rewritten dataset (on GPU).
        new_news2vector: Dictionary of news vectors for the rewritten dataset (on GPU).
        batch_size: Number of users to process in a single batch.
        K: List of top-K values for evaluation.
    
    Returns:
        dict: Dictionary containing computed metrics for each K.
    """
    
    # Convert news2vector and user2vector to tensors
    old_news_ids = list(old_news2vector.keys())
    target_news_index = torch.tensor([old_news_ids.index(n_id) for n_id in news_id]).to(device)
    
    old_news2vector = {k: v.to(device) for k, v in old_news2vector.items()}
    new_news2vector = {k: v.to(device) for k, v in new_news2vector.items()}
    
    old_user2vector = {k: v.to(device) for k, v in old_user2vector.items()}
    new_user2vector = {k: v.to(device) for k, v in new_user2vector.items()}
    
    old_news_vectors = torch.stack(list(old_news2vector.values()))  # Shape: (num_news, dim)
    new_news_vectors = torch.stack(list(new_news2vector.values()))  # Shape: (num_news, dim)

    old_user_vectors = torch.stack(list(old_user2vector.values()))  # Shape: (num_users, dim)
    new_user_vectors = torch.stack(list(new_user2vector.values()))  # Shape: (num_users, dim)

    num_users = old_user_vectors.shape[0]
    num_targets = len(news_id)

    # Initialize metrics
    exposure_counts = {k: 0 for k in K}
    hit_counts = {k: 0 for k in K}
    appear_counts = {k: 0 for k in K}
    boost_success_count = 0
    total_new_ranks = []

    # Process users in batches
    for start in tqdm(range(0, num_users, batch_size), desc="Processing Users", unit="batch"):
        end = min(start + batch_size, num_users)
        old_batch_user_vectors = old_user_vectors[start:end]
        new_batch_user_vectors = new_user_vectors[start:end]
        
        # Compute scores for all news
        old_scores = torch.matmul(old_batch_user_vectors, old_news_vectors.T)  # Shape: (batch_size, num_news)
        new_scores = torch.matmul(new_batch_user_vectors, new_news_vectors.T)  # Shape: (batch_size, num_news)

        # Get ranks for all target news
        old_ranks = torch.argsort(-old_scores, dim=1)  # Shape: (batch_size, num_news)
        new_ranks = torch.argsort(-new_scores, dim=1)  # Shape: (batch_size, num_news)
        old_ranks_indices = torch.argsort(old_ranks, dim=1)[:, target_news_index]  # Shape: (batch_size, num_targets)
        new_ranks_indices = torch.argsort(new_ranks, dim=1)[:, target_news_index]  # Shape: (batch_size, num_targets)

        # Compute total rank improvement
        boost_success_count += (new_ranks_indices < old_ranks_indices).sum().item()

        # Store total ranks for computing average rank
        total_new_ranks.extend(new_ranks_indices.cpu().numpy())

        for k in K:
            # Compute Exposure@K
            exposure_counts[k] += (new_ranks[:, :k].unsqueeze(2) == target_news_index).any(dim=1).float().sum().item()
            
            # Compute Hit Ratio@K
            hit_counts[k] += (new_ranks_indices < k).any(dim=1).sum().item()
            
            # Compute Appear@K
            appear_counts[k] += (new_ranks[:, :k].unsqueeze(2) == target_news_index).float().sum().item()

    # Final metrics
    total_new_ranks = np.concatenate(total_new_ranks)
    
    exposure = [exposure_counts[k] / num_users for k in K]
    hit_ratio = [hit_counts[k] / num_users for k in K]
    appear = [appear_counts[k] / (num_users * k) for k in K]
    boost_success_rate = boost_success_count / (num_users * num_targets)
    average_predicted_rank = total_new_ranks.mean()

    return {
        "Exposure@K": exposure,
        "Hit Ratio@K": hit_ratio,
        "Appear@K": appear,
        "Boost Success Rate": boost_success_rate,
        "Average Predicted Rank": average_predicted_rank,
    }

@torch.no_grad()
def calculate_metrics_on_original_old(news_id, old_user2vector, old_news2vector, batch_size=512, K=[10, 20, 50, 100]):
    """
    Calculate metrics for evaluating rewritten news performance for multiple K values.
    
    Args:
        news_id: List of target news IDs.
        old_user2vector: Dictionary of user vectors for the original dataset (on GPU).
        old_news2vector: Dictionary of news vectors for the original dataset (on GPU).
        batch_size: Number of users to process in a single batch.
        K: List of top-K values for evaluation.
    
    Returns:
        dict: Dictionary containing computed metrics for each K.
    """
    # , new_user2vector, new_news2vector, 
    # Convert news2vector and user2vector to tensors
    old_news_ids = list(old_news2vector.keys())
    # new_news_ids = list(new_news2vector.keys())
    target_news_index = torch.tensor([old_news_ids.index(n_id) for n_id in news_id]).to(device)
    
    old_news2vector = {k: v.to(device) for k, v in old_news2vector.items()}
    old_user2vector = {k: v.to(device) for k, v in old_user2vector.items()}
    
    old_news_vectors = torch.stack(list(old_news2vector.values()))  # Shape: (num_news, dim)
    old_user_vectors = torch.stack(list(old_user2vector.values()))  # Shape: (num_users, dim)

    num_users = old_user_vectors.shape[0]
    num_targets = len(news_id)

    # Initialize metrics
    exposure_counts = {k: 0 for k in K}
    hit_counts = {k: 0 for k in K}
    total_old_ranks = []

    # Process users in batches
    for start in tqdm(range(0, num_users, batch_size), desc="Processing Users", unit="batch"):
        end = min(start + batch_size, num_users)
        old_batch_user_vectors = old_user_vectors[start:end]
        
        # Compute scores for all news
        old_scores = torch.matmul(old_batch_user_vectors, old_news_vectors.T)  # Shape: (batch_size, num_news)

        # Get ranks for all target news
        old_ranks = torch.argsort(-old_scores, dim=1)  # Shape: (batch_size, num_news)
        old_ranks_indices = torch.argsort(old_ranks, dim=1)[:, target_news_index]  # Shape: (batch_size, num_targets)
        
        # Compute metrics
        
        total_old_ranks.extend(old_ranks_indices.cpu().numpy())

        for k in K:
            exposure_counts[k] += (old_ranks[:, :k].unsqueeze(2) == target_news_index).any(dim=1).float().sum().item()
            hit_counts[k] += (old_ranks_indices < k).float().sum().item()

    # Final metrics
    total_old_ranks = np.concatenate(total_old_ranks)

    num_users = len(total_old_ranks)
    exposure = [exposure_counts[k] / num_users for k in K]
    hit_ratio = [hit_counts[k] / (num_users * num_targets) for k in K]
    average_predicted_rank = total_old_ranks.mean()

    return {
        "Exposure": exposure,
        "Average Predicted Rank": average_predicted_rank,
        "Hit Ratio@K": hit_ratio,
    }

@torch.no_grad()
def calculate_metrics_on_original(news_id, old_user2vector, old_news2vector, batch_size=512, K=[10, 20, 50, 100]):
    # Convert news2vector and user2vector to tensors
    old_news_ids = list(old_news2vector.keys())
    target_news_index = torch.tensor([old_news_ids.index(n_id) for n_id in news_id]).to(device)
    
    old_news2vector = {k: v.to(device) for k, v in old_news2vector.items()}
    old_user2vector = {k: v.to(device) for k, v in old_user2vector.items()}
    
    old_news_vectors = torch.stack(list(old_news2vector.values()))  # Shape: (num_news, dim)
    old_user_vectors = torch.stack(list(old_user2vector.values()))  # Shape: (num_users, dim)

    num_users = old_user_vectors.shape[0]
    num_targets = len(news_id)

    # Initialize metrics
    exposure_counts = {k: 0 for k in K}
    hit_counts = {k: 0 for k in K}
    appear_counts = {k: 0 for k in K}
    total_old_ranks = []

    # Process users in batches
    for start in tqdm(range(0, num_users, batch_size), desc="Processing Users", unit="batch"):
        end = min(start + batch_size, num_users)
        old_batch_user_vectors = old_user_vectors[start:end]
        
        # Compute scores for all news
        old_scores = torch.matmul(old_batch_user_vectors, old_news_vectors.T)  # Shape: (batch_size, num_news)

        # Get ranks for all target news
        old_ranks = torch.argsort(-old_scores, dim=1)  # Shape: (batch_size, num_news)
        old_ranks_indices = torch.argsort(old_ranks, dim=1)[:, target_news_index]  # Shape: (batch_size, num_targets)

        # Store total ranks for computing average rank
        total_old_ranks.extend(old_ranks_indices.cpu().numpy())

        for k in K:
            # Compute Exposure@K
            exposure_counts[k] += (old_ranks[:, :k].unsqueeze(2) == target_news_index).any(dim=1).float().sum().item()
            
            # Compute Hit Ratio@K
            hit_counts[k] += (old_ranks_indices < k).any(dim=1).sum().item()
            
            # Compute Appear@K
            appear_counts[k] += (old_ranks[:, :k].unsqueeze(2) == target_news_index).float().sum().item()

    # Final metrics
    total_old_ranks = np.concatenate(total_old_ranks)
    
    exposure = [exposure_counts[k] / num_users for k in K]
    hit_ratio = [hit_counts[k] / num_users for k in K]
    appear = [appear_counts[k] / (num_users * k) for k in K]
    average_predicted_rank = total_old_ranks.mean()

    return {
        "Exposure@K": exposure,
        "Hit Ratio@K": hit_ratio,
        "Appear@K": appear,
        "Average Predicted Rank": average_predicted_rank,
    }

@torch.no_grad()
def calculate_metrics_by_group(news_id, old_user2vector, old_news2vector, new_user2vector, new_news2vector, pop_group, batch_size=512, K=[10, 20, 50, 100]):
    """
    Calculate metrics for evaluating rewritten news performance for each popularity group.
    
    Args:
        news_id: List of target news IDs.
        old_user2vector: Dictionary of user vectors for the original dataset (on GPU).
        old_news2vector: Dictionary of news vectors for the original dataset (on GPU).
        new_user2vector: Dictionary of user vectors for the rewritten dataset (on GPU).
        new_news2vector: Dictionary of news vectors for the rewritten dataset (on GPU).
        pop_group: [[top_index], [medium_index], [bottom_index]] (list of lists of indices).
        batch_size: Number of users to process in a single batch.
        K: List of top-K values for evaluation.
    
    Returns:
        dict: Dictionary containing computed metrics for each group and each K.
    """
    metrics_by_group = {}
    group_names = ["Top", "Medium", "Bottom"]

    for group_name, indices in zip(group_names, pop_group):
        group_news_id = [news_id[i] for i in indices]  # Get news IDs for the current group
        print(f"Calculating metrics for {group_name} group with {len(group_news_id)} news items...")

        # Call the original calculate_metrics function for the current group
        group_metrics = calculate_metrics(
            group_news_id,
            old_user2vector,
            old_news2vector,
            new_user2vector,
            new_news2vector,
            batch_size=batch_size,
            K=K
        )

        metrics_by_group[group_name] = group_metrics

    return metrics_by_group

@torch.no_grad()
def calculate_metrics_by_group_original(news_id, old_user2vector, old_news2vector, pop_group, batch_size=512, K=[10, 20, 50, 100]):
    """
    Calculate metrics for evaluating rewritten news performance for each popularity group.
    
    Args:
        news_id: List of target news IDs.
        old_user2vector: Dictionary of user vectors for the original dataset (on GPU).
        old_news2vector: Dictionary of news vectors for the original dataset (on GPU).
        new_user2vector: Dictionary of user vectors for the rewritten dataset (on GPU).
        new_news2vector: Dictionary of news vectors for the rewritten dataset (on GPU).
        pop_group: [[top_index], [medium_index], [bottom_index]] (list of lists of indices).
        batch_size: Number of users to process in a single batch.
        K: List of top-K values for evaluation.
    
    Returns:
        dict: Dictionary containing computed metrics for each group and each K.
    """
    metrics_by_group = {}
    group_names = ["Top", "Medium", "Bottom"]

    for group_name, indices in zip(group_names, pop_group):
        group_news_id = [news_id[i] for i in indices]  # Get news IDs for the current group
        print(f"Calculating metrics for {group_name} group with {len(group_news_id)} news items...")

        # Call the original calculate_metrics function for the current group
        group_metrics = calculate_metrics_on_original(
            group_news_id,
            old_user2vector,
            old_news2vector,
            batch_size=batch_size,
            K=K
        )

        metrics_by_group[group_name] = group_metrics

    return metrics_by_group


@torch.no_grad()
def compute_rank_metrics(target_original_path, target_rewritten_path, test_directory, model):

    with open(target_original_path, 'rb') as f:
        target_test = pickle.load(f)
    with open(target_rewritten_path, 'rb') as f:
        rewrited_news_collection = pickle.load(f)

    # Step 1: Calculate news2vector
    news_dataset = NewsDataset(path.join(test_directory, 'news_parsed.tsv'))
    user_dataset = UserDataset(path.join(test_directory, 'behaviors.tsv'),
                               path.join(config.original_data_path, 'train/user2int.tsv'))

    original_news2vector = get_news2vector(news_dataset, model)
    original_user2vector = get_user2vector(user_dataset, original_news2vector, model)
    
    # Step 2: compute average rank of target news among all users
    # start here
    result = {
        "original": {
            "index": target_test["index"],
            "news_ID": target_test["news_ID"],
            "rank": [],
            "inverse_rank": [],
            "news_title": target_test['news_title'],
            "result": [],
            "news_abstract": target_test['news_abstract']
        },
        "rewrited": {
            "index": target_test["index"],
            "news_ID": target_test["news_ID"],
            "result": [[] for _ in rewrited_news_collection['news_title'][0]],
            "inverse_rank": [[] for _ in rewrited_news_collection['news_title'][0]],
            "rank_gap": [[] for _ in rewrited_news_collection['news_title'][0]],
            "inverse_rank_gap": [[] for _ in rewrited_news_collection['news_title'][0]],
            "news_title": rewrited_news_collection['news_title'],
            "news_abstract": rewrited_news_collection['news_abstract']
        }
    }

    top_medium_bottom = [target_test['top_index'], target_test['medium_index'], target_test['bottom_index']]

    K_values = [10, 20, 50, 100]
    result["original"]["result"] = calculate_metrics_on_original(
            result["original"]["news_ID"], original_user2vector, original_news2vector, batch_size=512, K=K_values
        )
    print(f'Evaluation Result of Original news, Top [10, 20, 50, 100] is: ')
    for key_metric, value in result["original"]["result"].items():
        if isinstance(value, list):
            # Join list elements with tabs
            value_str = "\t".join(map(str, value))
            print(f"{key_metric}\t\t\t{value_str}")
        else:
            print(f"{key_metric}\t\t\t{value}")
    # for easier to write in table
    key_metric_name = ["Average Predicted Rank", "Exposure@K", "Hit Ratio@K", "Appear@K"]
    value_metrics = [result["original"]["result"][key] for key in key_metric_name]
    print("\t".join(key_metric_name))
    # print("\t".join(map(str, value_metric[2] if isinstance(value_metric, list) else value_metric)))
    print("\t".join(map(lambda value_metric: str(value_metric[2]) if isinstance(value_metric, list) else str(value_metric), value_metrics)))
    group_result = calculate_metrics_by_group_original(
        result["original"]["news_ID"], original_user2vector, original_news2vector, pop_group=top_medium_bottom, batch_size=512, K=K_values
    )
    # Print results
    for group, metrics in group_result.items():
        print(f"{group} Group Metrics:")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value}")
        # for easier to write in table
        key_metric_name = ["Average Predicted Rank", "Exposure@K", "Hit Ratio@K", "Appear@K"]
        value_metrics = [metrics[key] for key in key_metric_name]
        print("\t".join(key_metric_name))
        print("\t".join(map(lambda value_metric: str(value_metric[2]) if isinstance(value_metric, list) else str(value_metric), value_metrics)))

    
    
    # Step 3: Process Rewritten News and Get Their Vectors
    print("Calculating vectors for rewritten news...")
    num_rewrite = len(rewrited_news_collection["news_title"][0])

    

    del news_dataset.news_parsed
    
    for i in range(num_rewrite):
        tmp_news_dataset = copy.deepcopy(news_dataset)
        tmp_rewrited_news_dataset = replace_rewrite_news(tmp_news_dataset, rewrited_news_collection, position=i)
        tmp_rewrited_news2vector = get_news2vector(tmp_rewrited_news_dataset, model)
        tmp_rewrited_user2vector = get_user2vector(user_dataset, tmp_rewrited_news2vector, model)
        # Step 4: compute average rank of rewrited target news among all users, and get the gap value
        result["rewrited"]["result"][i] = calculate_metrics(
            result["original"]["news_ID"], original_user2vector, original_news2vector, tmp_rewrited_user2vector, tmp_rewrited_news2vector, batch_size=512, K=K_values
        )
        print(f'Evaluation Result of Rewrited Version {i}, Top [10, 20, 50, 100] is: ')
        for key_metric, value in result["rewrited"]["result"][i].items():
            if isinstance(value, list):
                # Join list elements with tabs
                value_str = "\t".join(map(str, value))
                print(f"{key_metric}\t\t\t{value_str}")
            else:
                print(f"{key_metric}\t\t\t{value}")
        # for easier to write in table
        key_metric_name = ["Boost Success Rate", "Average Predicted Rank", "Exposure@K", "Hit Ratio@K", "Appear@K"]
        value_metrics = [result["rewrited"]["result"][i][key] for key in key_metric_name]
        print("\t".join(key_metric_name))
        print("\t".join(map(lambda value_metric: str(value_metric[2]) if isinstance(value_metric, list) else str(value_metric), value_metrics)))

        group_result = calculate_metrics_by_group(
            result["original"]["news_ID"], original_user2vector, original_news2vector, tmp_rewrited_user2vector, tmp_rewrited_news2vector, pop_group=top_medium_bottom, batch_size=512, K=K_values
        )
        # Print results
        for group, metrics in group_result.items():
            print(f"{group} Group Metrics:")
            for metric_name, value in metrics.items():
                print(f"  {metric_name}: {value}")
            # for easier to write in table
            key_metric_name = ["Boost Success Rate", "Average Predicted Rank", "Exposure@K", "Hit Ratio@K", "Appear@K"]
            value_metrics = [metrics[key] for key in key_metric_name]
            print("\t".join(key_metric_name))
            print("\t".join(map(lambda value_metric: str(value_metric[2]) if isinstance(value_metric, list) else str(value_metric), value_metrics)))

    return result

def check_text_valid(text_list):
    for i, t in enumerate(text_list):
        if len(t) < 10:
            print(f'Num is {i}, content is {t}.')

def calculate_naturalnessScore(target_original_path, target_rewritten_path):
    with open(target_path, 'rb') as f:
        target_test = pickle.load(f)
    with open(rewrited_target_path, 'rb') as f:
        rewrited_news_collection = pickle.load(f)
    
    references_title = target_test['news_title']
    references_abstract = target_test['news_abstract']
    predictions_title = [n[0] for n in rewrited_news_collection['news_title']]
    predictions_abstract = [n[0] for n in rewrited_news_collection['news_abstract']]

    references_title_and_abstract = [title + '\t' + abstract for title, abstract in zip(references_title, references_abstract)]
    predictions_title_and_abstract = [title + '\t' + abstract for title, abstract in zip(predictions_title, predictions_abstract)]

    bleu = evaluate.load("bleu")
    perplexity = evaluate.load("perplexity", module_type="metric")
    rouge = evaluate.load("rouge")
    bertscore = evaluate.load("bertscore")

    # calculate bleu score
    bleu_title = bleu.compute(predictions=predictions_title, references=references_title)
    bleu_title = bleu_title['bleu']

    bleu_abstract = bleu.compute(predictions=predictions_abstract, references=references_abstract)
    bleu_abstract = bleu_abstract['bleu']

    bleu_title_and_abstract = bleu.compute(predictions=predictions_title_and_abstract, references=references_title_and_abstract)
    bleu_title_and_abstract = bleu_title_and_abstract['bleu']

    # calculate perplexity score
    preplexity_title = perplexity.compute(predictions=predictions_title, model_id='gpt2')
    preplexity_abstract = perplexity.compute(predictions=predictions_abstract, model_id='gpt2')
    preplexity_title_abstract = perplexity.compute(predictions=predictions_title_and_abstract, model_id='gpt2')
    preplexity_title = preplexity_title['mean_perplexity']
    preplexity_abstract = preplexity_abstract['mean_perplexity']
    preplexity_title_abstract = preplexity_title_abstract['mean_perplexity']

    preplexity_original_title = perplexity.compute(predictions=references_title, model_id='gpt2')
    preplexity_original_abstract = perplexity.compute(predictions=references_abstract, model_id='gpt2')
    preplexity_original_title_abstract = perplexity.compute(predictions=references_title_and_abstract, model_id='gpt2')
    preplexity_original_title = preplexity_original_title['mean_perplexity']
    preplexity_original_abstract = preplexity_original_abstract['mean_perplexity']
    preplexity_original_title_abstract = preplexity_original_title_abstract['mean_perplexity']

    # calculate rouge score
    rouge_title = rouge.compute(predictions=predictions_title, references=references_title)
    rouge_abstract = rouge.compute(predictions=predictions_abstract, references=references_abstract)
    rouge_title_and_abstract = rouge.compute(predictions=predictions_title_and_abstract, references=references_title_and_abstract)
    rougeL_title = rouge_title['rougeL']
    rougeL_abstract = rouge_abstract['rougeL']
    rougeL_title_and_abstract = rouge_title_and_abstract['rougeL']

    # calculate bertscore
    bertscore_title = bertscore.compute(predictions=predictions_title, references=references_title, model_type='bert-base-uncased')
    bertscore_abstract = bertscore.compute(predictions=predictions_abstract, references=references_abstract, model_type='bert-base-uncased')
    bertscore_title_and_abstract = bertscore.compute(predictions=predictions_title_and_abstract, references=references_title_and_abstract, model_type='bert-base-uncased')

    mean_bertscore_title = np.mean(bertscore_title['f1'])
    mean_bertscore_abstract = np.mean(bertscore_abstract['f1'])
    mean_bertscore_title_and_abstract = np.mean(bertscore_title_and_abstract['f1'])

    from tabulate import tabulate

    # Creating the table data
    table_natutal_data = [
        ["Metric", "Title", "Abstract", "Title + Abstract"],
        ["Original Perplexity", preplexity_original_title, preplexity_original_abstract, preplexity_original_title_abstract],
        ["Rewritten BLEU", bleu_title, bleu_abstract, bleu_title_and_abstract],
        ["Rewritten Perplexity", preplexity_title, preplexity_abstract, preplexity_title_abstract],
        ["Rewritten RougeL", rougeL_title, rougeL_abstract, rougeL_title_and_abstract],
        ["Rewritten BERTScore", mean_bertscore_title, mean_bertscore_abstract, mean_bertscore_title_and_abstract],
    ]
    print(tabulate(table_natutal_data, headers="firstrow", tablefmt="grid"))

    return bleu_title, bleu_abstract, bleu_title_and_abstract, preplexity_title, preplexity_abstract, preplexity_title_abstract, preplexity_original_title, preplexity_original_abstract, preplexity_original_title_abstract, rougeL_title, rougeL_abstract, rougeL_title_and_abstract, mean_bertscore_title, mean_bertscore_abstract, mean_bertscore_title_and_abstract

if __name__ == '__main__':
    print('Using device:', device)
    print(f'Evaluating model {model_name}')
    # Don't need to load pretrained word/entity/context embedding
    # since it will be loaded from checkpoint later
    model = Model(config).to(device)
    from train import latest_checkpoint  # Avoid circular imports
    checkpoint_path = latest_checkpoint(os.path.join(config.current_data_path + '/checkpoint', config.pretrained_mode, model_name))
    # checkpoint_path = "/data/XXXXXXXX/ndata/data/checkpoint/bert/NAML/ckpt-18000.pth"
    if checkpoint_path is None:
        print('No checkpoint file found!')
        exit()
    print(f"Load saved parameters in {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    target_dir = '/var/scratch/XXXXXXXX/data_attack/target_gpt'
    rewrited_news_collection = {
        "index": [],
        "top_index": [],
        "medium_index": [],
        "bottom_index": [],
        "news_ID": [],
        "news_title": [],
        "news_abstract": [],
    }
    train_or_test = {
        "train": "target_train_output.pkl",
        "test": "target_test_output.pkl"
    }

    test_directory = path.join(config.original_data_path, config.target_trainORtest)
    target_path = path.join(target_dir, train_or_test[config.target_trainORtest])
    if config.llm_test_version != 0:
        target_llama_dir = path.join(target_dir, config.llm_name + "V" + str(config.llm_test_version))
    else:
        target_llama_dir = path.join(target_dir, config.llm_name)
    tmp_model_name = model_name
    if config.llm_name == 'llama3_base' or config.llm_name == 'llama3_mind_pretrain_news_content' or config.llm_name == 'llama3_multi_merged_dpo':
        model_name = 'ALL'
    if "NAMLbert" in config.llm_name:
        model_name = 'NAMLbert'
    elif "NRMSbert" in config.llm_name:
        model_name = 'NRMSbert'
    elif "LSTURbert" in config.llm_name:
        model_name = 'LSTURbert'
    rewrited_target_path = path.join(target_llama_dir, 'rewrited_target_' + model_name + "_" + config.news_encoder + "_" + config.target_trainORtest + "_" + config.diverse_mode + '.pkl')
    model_name = tmp_model_name
    result_attack = compute_rank_metrics(
        target_original_path=target_path,
        target_rewritten_path=rewrited_target_path,
        test_directory=test_directory,
        model=model
    )

    result_naturalness = calculate_naturalnessScore(
        target_original_path=target_path,
        target_rewritten_path=rewrited_target_path
    )