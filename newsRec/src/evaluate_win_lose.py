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
import random

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
def calculate_exposure(news_id, new_user2vector, new_news2vector, batch_size=512, K=[10, 20, 50, 100]):
    """
    Calculate metrics for evaluating rewritten news performance for multiple K values.
    
    Args:
        news_id: List of target news IDs.
        new_user2vector: Dictionary of user vectors for the rewritten dataset (on GPU).
        new_news2vector: Dictionary of news vectors for the rewritten dataset (on GPU).
        batch_size: Number of users to process in a single batch.
        K: List of top-K values for evaluation.
    
    Returns:
        dict: Dictionary containing computed metrics for each K.
    """
    
    # Convert news2vector and user2vector to tensors
    news_ids = list(new_news2vector.keys())
    # new_news_ids = list(new_news2vector.keys())
    target_news_index = torch.tensor([news_ids.index(n_id) for n_id in news_id]).to(device)
    
    new_news2vector = {k: v.to(device) for k, v in new_news2vector.items()}
    new_user2vector = {k: v.to(device) for k, v in new_user2vector.items()}
    
    new_news_vectors = torch.stack(list(new_news2vector.values()))  # Shape: (num_news, dim)
    new_user_vectors = torch.stack(list(new_user2vector.values()))  # Shape: (num_users, dim)

    num_users = new_user_vectors.shape[0]
    num_targets = len(news_id)

    # Initialize exposure lists for each K
    exposure_lists = {k: torch.zeros(num_targets, device=device) for k in K}

    # Process users in batches
    for start in tqdm(range(0, num_users, batch_size), desc="Processing Users", unit="batch"):
        end = min(start + batch_size, num_users)
        new_batch_user_vectors = new_user_vectors[start:end]
        
        # Compute scores for all news
        new_scores = torch.matmul(new_batch_user_vectors, new_news_vectors.T)  # Shape: (batch_size, num_news)

        # Get ranks for all target news
        new_ranks = torch.argsort(-new_scores, dim=1)  # Shape: (batch_size, num_news)
        
        for k in K:
            # Check if target news is in the top-K for each user in the batch
            in_top_k = (new_ranks[:, :k].unsqueeze(2) == target_news_index).any(dim=1)  # Shape: (batch_size, num_targets)
            exposure_lists[k] += in_top_k.sum(dim=0).float()

    # Convert exposure counts to lists for each target news
    exposure_lists = {k: exposure_lists[k].cpu().tolist() for k in K}

    return exposure_lists


@torch.no_grad()
def compute_rank_metrics(target_original_path, target_dir, test_directory, model, diverse_mode, target_trainORtest):

    with open(target_original_path, 'rb') as f:
        target_test = pickle.load(f)

    rewrited_news_collection = 0
    for diverse in diverse_mode:
        tmp_target_rewritten_path = path.join(target_dir, 'rewrited_target_' + target_trainORtest + "_" + diverse + '.pkl')
        with open(tmp_target_rewritten_path, 'rb') as f:
            tmp_rewrited_news_collection = pickle.load(f)

        if not rewrited_news_collection:
            rewrited_news_collection = tmp_rewrited_news_collection
        else:
            rewrited_news_collection['news_title'] = [rewrited_news_collection['news_title'][idx] + tmp_rewrited_news_collection['news_title'][idx] for idx in rewrited_news_collection['index']]
            rewrited_news_collection['news_abstract'] = [rewrited_news_collection['news_abstract'][idx] + tmp_rewrited_news_collection['news_abstract'][idx] for idx in rewrited_news_collection['index']]


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
            "exposure": [],
            "rank": [],
            "inverse_rank": [],
            "news_title": target_test['news_title'],
            "news_abstract": target_test['news_abstract']
        },
        "rewrited": {
            "index": target_test["index"],
            "news_ID": target_test["news_ID"],
            "exposure": [[] for _ in rewrited_news_collection['news_title'][0]],
            "inverse_rank": [[] for _ in rewrited_news_collection['news_title'][0]],
            "rank_gap": [[] for _ in rewrited_news_collection['news_title'][0]],
            "inverse_rank_gap": [[] for _ in rewrited_news_collection['news_title'][0]],
            "news_title": rewrited_news_collection['news_title'],
            "news_abstract": rewrited_news_collection['news_abstract']
        }
    }

    K_values = [10, 20, 50, 100]
    result["original"]["exposure"] = calculate_exposure(
            result["original"]["news_ID"], original_user2vector, original_news2vector, batch_size=512, K=K_values
        )


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
        result["rewrited"]["exposure"][i] = calculate_exposure(
            result["original"]["news_ID"], tmp_rewrited_user2vector, tmp_rewrited_news2vector, batch_size=512, K=K_values
        )
        print(f'Done with evaluation Result of Rewrited Version {i}, Top [10, 20, 50, 100].')

    return result

def extract_win_lose_K_list_largest(model_ori_win_lose_K):
    original_scores = model_ori_win_lose_K['original']['exposure'][trun]
    rewritten_scores = [model_ori_win_lose_K['rewrited']['exposure'][i][trun] for i in range(len(model_ori_win_lose_K['rewrited']['exposure']))]
    useful, chose, reject = extract_dpo_data(original_scores, rewritten_scores)

        # Save the result
    with open(path.join(target_dir, f'{model_name}_largest_useful_chose_reject_' + config.target_trainORtest + "_" + config.pretrained_mode + '_ft.pkl'), "wb") as f:
        pickle.dump((useful, chose, reject), f)
    print("Model useful, chose, reject information saved to " + path.join(target_dir, f'{model_name}_largest_useful_chose_reject_' + config.target_trainORtest + "_" + config.pretrained_mode + '_ft.pkl'))



    rewrited_news_win_collection = {
        "index": [],
        "top_index": [],
        "medium_index": [],
        "bottom_index": [],
        "news_ID": [],
        "news_title": [],
        "news_abstract": [],
    }
    rewrited_news_reject_collection = {
        "index": [],
        "top_index": [],
        "medium_index": [],
        "bottom_index": [],
        "news_ID": [],
        "news_title": [],
        "news_abstract": [],
    }
    for idx, uf in enumerate(useful):
        if uf:
            rewrited_news_win_collection['index'].append(idx)
            rewrited_news_win_collection['news_ID'].append(model_ori_win_lose_K['rewrited']['news_ID'][idx])
            rewrited_news_win_collection['news_title'].append([model_ori_win_lose_K['rewrited']['news_title'][idx][chose[idx]]])
            rewrited_news_win_collection['news_abstract'].append([model_ori_win_lose_K['rewrited']['news_abstract'][idx][chose[idx]]])
            rewrited_news_reject_collection['index'].append(idx)
            rewrited_news_reject_collection['news_ID'].append(model_ori_win_lose_K['rewrited']['news_ID'][idx])
            rewrited_news_reject_collection['news_title'].append([model_ori_win_lose_K['rewrited']['news_title'][idx][reject[idx]]])
            rewrited_news_reject_collection['news_abstract'].append([model_ori_win_lose_K['rewrited']['news_abstract'][idx][reject[idx]]])

    rewrited_target_path_train_win = path.join(target_dir, model_name + '_largest_train_win_rewrited_target_' + config.target_trainORtest + "_" + config.diverse_mode + "_" + config.llm_api_mode + '.pkl')
    with open(rewrited_target_path_train_win, 'wb') as f:
        pickle.dump(rewrited_news_win_collection, f)
    print(f"Rewrited target news for training saved to {rewrited_target_path_train_win}")

    rewrited_target_path_train_reject = path.join(target_dir, model_name + '_largest_train_reject_rewrited_target_' + config.target_trainORtest + "_" + config.diverse_mode + "_" + config.llm_api_mode + '.pkl')
    with open(rewrited_target_path_train_reject, 'wb') as f:
        pickle.dump(rewrited_news_reject_collection, f)
    print(f"Rewrited target news for training saved to {rewrited_target_path_train_reject}")

    print("Length of useful: ", len(rewrited_news_win_collection['news_ID']))
    print("Length of reject: ", len(rewrited_news_reject_collection['news_ID']))


def extract_dpo_data(original_score, rewrited_score):
    """
    Extract data for comparison between original and rewritten news exposure scores.

    Args:
        original_score: List of scores for original news.
        rewrited_score: List of lists of scores for rewritten news (each sublist corresponds to a version).

    Returns:
        tuple:
            list: indicate_useful, whether the rewritten news has a bigger exposure score than the original one.
            list: indicate_index_chose, the index of the rewritten news with the biggest improvement (0~14).
            list: indicate_index_reject, the index of the rewritten news with the biggest decline (0~14).
    """
    num_news = len(original_score)
    num_versions = len(rewrited_score)

    indicate_useful = []
    indicate_index_chose = []
    indicate_index_reject = []

    for i in range(num_news):
        orig_score = original_score[i]
        rewritten_scores = [rewrited_score[j][i] for j in range(num_versions)]

        # Determine if any rewritten version has a bigger score
        useful = any(score > orig_score for score in rewritten_scores)
        indicate_useful.append(useful)

        if useful:
        # Find the index of the biggest improvement
            improvements = [score - orig_score for score in rewritten_scores]
            max_improvement = max(improvements)
            max_indices = [idx for idx, imp in enumerate(improvements) if imp == max_improvement]
            index_chose = random.choice(max_indices)
            indicate_index_chose.append(index_chose)

            # Find the index of the biggest decline
            min_improvement = min(improvements)
            min_indices = [idx for idx, imp in enumerate(improvements) if imp == min_improvement]
            index_reject = random.choice(min_indices)
            indicate_index_reject.append(index_reject)
        else:
            indicate_index_chose.append(0)
            indicate_index_reject.append(0)
    return indicate_useful, indicate_index_chose, indicate_index_reject

def extract_multi_largest_dpo_data(original_score, rewrited_score):
    """
    Extract data for comparison between original and rewritten news exposure scores.

    Args:
        original_score: List of scores for original news.
        rewrited_score: List of lists of scores for rewritten news (each sublist corresponds to a version).

    Returns:
        tuple:
            list: indicate_useful, whether the rewritten news has a bigger exposure score than the original one.
            list: indicate_index_chose, the index of the rewritten news with the biggest improvement (0~14).
            list: indicate_index_reject, the index of the rewritten news with the biggest decline (0~14).
    """
    num_news = len(original_score)
    num_versions = len(rewrited_score)

    indicate_useful = []
    indicate_index_chose = []
    indicate_index_reject = []

    for i in range(num_news):
        orig_score = original_score[i]
        rewritten_scores = [rewrited_score[j][i] for j in range(num_versions)]

        # Determine if any rewritten version has a bigger score
        useful = any(score > orig_score for score in rewritten_scores)
        indicate_useful.append(useful)

        if useful:
            # Find the index of the biggest improvement
            improvements = [score - orig_score for score in rewritten_scores]
            max_improvement = max(improvements)
            max_indices = [idx for idx, imp in enumerate(improvements) if imp == max_improvement]

            # Find the index of the biggest decline
            min_improvement = min(improvements)
            min_indices = [idx for idx, imp in enumerate(improvements) if imp == min_improvement]
            if min_improvement == 0:
                min_indices = [random.choice(min_indices)]
            
            index_chose = []
            index_reject = []
            for c_idx in max_indices:
                for r_idx in min_indices:
                    index_chose.append(c_idx)
                    index_reject.append(r_idx)
            indicate_index_chose.append(index_chose)
            indicate_index_reject.append(index_reject)
        else:
            indicate_index_chose.append([0])
            indicate_index_reject.append([0])

    return indicate_useful, indicate_index_chose, indicate_index_reject

def extract_hard_sample_dpo_data(original_score, rewrited_score, min_gap=20, max_gap=100):
    """
    Extract data for comparison between original and rewritten news exposure scores.

    Args:
        original_score: List of scores for original news.
        rewrited_score: List of lists of scores for rewritten news (each sublist corresponds to a version).

    Returns:
        tuple:
            list: indicate_useful, whether the rewritten news has a bigger exposure score than the original one.
            list: indicate_index_chose, the index of the rewritten news with the biggest improvement (0~14).
            list: indicate_index_reject, the index of the rewritten news with the biggest decline (0~14).
    """
    num_news = len(original_score)
    num_versions = len(rewrited_score)

    indicate_useful = []
    indicate_index_chose = []
    indicate_index_reject = []

    for i in range(num_news):
        orig_score = original_score[i]
        rewritten_scores = [rewrited_score[j][i] for j in range(num_versions)]

        # # Determine if any rewritten version has a bigger score
        # useful = any(score > orig_score for score in rewritten_scores)
        # indicate_useful.append(useful)

        # Find the index of the biggest improvement
        improvements = [score - orig_score for score in rewritten_scores]
        pos_samples = [i for i in range(len(improvements)) if improvements[i] > 0]
        neg_samples = [i for i in range(len(improvements)) if improvements[i] <= 0]

        # Construct hard pairs
        index_chose = []
        index_reject = []
        for pos_idx in pos_samples:
            for neg_idx in neg_samples:
                if improvements[pos_idx] - improvements[neg_idx] >= min_gap and improvements[pos_idx] - improvements[neg_idx] <= max_gap:
                    index_chose.append(pos_idx)
                    index_reject.append(neg_idx)
        
        for i in range(len(pos_samples)):
            for j in range(i+1, len(pos_samples)):
                if abs(improvements[pos_samples[i]] - improvements[pos_samples[j]]) >= min_gap and abs(improvements[pos_samples[i]] - improvements[pos_samples[j]]) <= max_gap:
                    index_chose.append(pos_samples[i])
                    index_reject.append(pos_samples[j])
        
        indicate_index_chose.append(index_chose)
        indicate_index_reject.append(index_reject)

        useful = True if len(index_chose) > 0 else False
        indicate_useful.append(useful)

    return indicate_useful, indicate_index_chose, indicate_index_reject

if __name__ == '__main__':
    print('Using device:', device)
    print(f'Evaluating model {model_name}')
    # Don't need to load pretrained word/entity/context embedding
    # since it will be loaded from checkpoint later
    # model = Model(config).to(device)
    # from train import latest_checkpoint  # Avoid circular imports
    # checkpoint_path = latest_checkpoint(os.path.join(config.current_data_path + '/checkpoint', config.pretrained_mode, model_name))
    # # checkpoint_path = "/data/XXXXXXXX/ndata/data/checkpoint/bert/NAML/ckpt-18000.pth"
    # if checkpoint_path is None:
    #     print('No checkpoint file found!')
    #     exit()
    # print(f"Load saved parameters in {checkpoint_path}")
    # checkpoint = torch.load(checkpoint_path)
    # model.load_state_dict(checkpoint['model_state_dict'])
    # model.eval()
    
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

    diverse_mode = ["writing_style", "sentiment", "persona"]

    test_directory = path.join(config.original_data_path, config.target_trainORtest)
    target_path = path.join(target_dir, train_or_test[config.target_trainORtest])
    rewrited_target_path = path.join(target_dir, f'rewrited_target_' + config.target_trainORtest + "_" + config.diverse_mode + '.pkl')
    
    # model_ori_win_lose_K = compute_rank_metrics(
    #     target_original_path=target_path,
    #     target_dir=target_dir,
    #     test_directory=test_directory,
    #     model=model,
    #     diverse_mode=diverse_mode,
    #     target_trainORtest=config.target_trainORtest
    # )

    # with open(path.join(target_dir, f'{model_name}_win_lose_K_list' + config.target_trainORtest + "_" + config.pretrained_mode + '_ft.pkl'), "wb") as f:
    #     pickle.dump(model_ori_win_lose_K, f)
    # print("Model win/lose information saved to " + path.join(target_dir, f'{model_name}_win_lose_K_list' + config.target_trainORtest + "_" + config.pretrained_mode + '_ft.pkl'))
    
    path_to_eval = path.join(target_dir, f'{model_name}_win_lose_K_list' + config.target_trainORtest + "_" + config.pretrained_mode + '_ft.pkl')
    with open(path_to_eval, "rb") as f:
        model_ori_win_lose_K = pickle.load(f)
    # path = '/var/scratch/XXXXXXXX/data_attack/target_gpt/target_train_output.pkl'
    # with open(path_to_eval, "rb") as f:
    #     taret_data = pickle.load(f)
    trun = 50
    extract_win_lose_K_list_largest(model_ori_win_lose_K)