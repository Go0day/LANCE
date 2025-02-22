# This is a function to get the average rank of target item among all users
# including **ALL versions** of rewrited news. 

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
def compute_average_rank(news_id, user2vector, news2vector, batch_size=512):
    """
    Compute the average rank and inverse rank for a single news ID across all users.
    Args:
        news_id: ID of the news article.
        user2vector: Dictionary mapping users to their vectors (on GPU).
        news2vector: Dictionary mapping news IDs to their vectors (on GPU).
        batch_size: Number of users to process in a single batch.
    Returns:
        rank and inverse rank for the news_id item among all the users.
    """
    # Convert news2vector to tensor for efficient computation
    news_ids = list(news2vector.keys())
    target_news_index = torch.tensor([news_ids.index(n_id) for n_id in news_id]).to(device)
    news2vector = {k: v.to(device) for k, v in news2vector.items()}
    user2vector = {k: v.to(device) for k, v in user2vector.items()}

    news_vectors = torch.stack(list(news2vector.values()))  # Shape: (num_news, dim)

    # Convert user2vector to tensor for batch processing
    user_vectors = torch.stack(list(user2vector.values()))  # Shape: (num_users, dim)
    num_users = user_vectors.shape[0]

    # Initialize lists to store ranks and inverse ranks
    ranks = []
    inverse_ranks = []

    # Process users in batches
    # for start in range(0, num_users, batch_size):
    for start in tqdm(range(0, num_users, batch_size), desc="Processing Users", unit="batch"):
        end = min(start + batch_size, num_users)
        batch_user_vectors = user_vectors[start:end]  # Shape: (batch_size, dim)

        # Compute scores for all news for the batch of users
        scores = torch.matmul(batch_user_vectors, news_vectors.T)  # Shape: (batch_size, num_news)

        # Get the scores for the target news
        target_scores = scores[:, target_news_index]  # Shape: (batch_size, 1, target_news_num)

        # Calculate the rank of the target news for each user
        # rank_batch = (scores.unsqueeze(2) >= target_scores.unsqueeze(1)).sum(dim=1).cpu().numpy()  # Shape: (batch_size,)
        rank_batch = []
        chunk_size = 500
        # Process target_news in chunks
        for start_idx in range(0, target_news_index.shape[0], chunk_size):
            end_idx = min(start_idx + chunk_size, target_news_index.shape[0])
            target_indices_batch = target_news_index[start_idx:end_idx]  # Batch of target indices
            target_scores_batch = target_scores[:, start_idx:end_idx]  # Shape: (batch_size, batch_target_size)

            # Compute ranks for this batch of target indices
            batch_ranks = []
            for target_idx in range(target_scores_batch.shape[1]):
                target_score = target_scores_batch[:, target_idx].unsqueeze(1)  # Shape: (batch_size, 1)
                rank = (scores >= target_score).sum(dim=1).cpu().numpy()  # Shape: (batch_size,)
                batch_ranks.append(rank)

            rank_batch.append(np.stack(batch_ranks, axis=1))

        # Concatenate all rank batches
        rank_batch = np.concatenate(rank_batch, axis=1)
        inverse_rank_batch = 1 / rank_batch  # Shape: (batch_size,)

        ranks.extend(rank_batch)
        inverse_ranks.extend(inverse_rank_batch)

    # Compute the mean rank and inverse rank
    avg_rank = np.mean(ranks, axis=0)
    avg_inverse_rank = np.mean(inverse_ranks, axis=0)

    return avg_rank, avg_inverse_rank

def plot_rank_distribution(original_rank, rewrited_rank, target_test, group="Overall", rewrite_mode="writing_style"):
    """
    Plot the distribution of ranks for original and rewritten news items.
    Args:
        original_rank: Array of ranks for original news.
        rewrited_rank: Array of ranks for rewritten news (shape: num_items x num_rewrites).
        target_test: Dictionary containing group indices.
        group: The group to plot ('Overall', 'top', 'medium', 'bottom').
        rewrite_mode: 'writing_style', 'sentiment', 'persona'.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # Select indices based on the group
    if group == "Overall":
        indices = range(len(target_test["index"]))
    elif group == "top":
        indices = target_test["top_index"]
    elif group == "medium":
        indices = target_test["medium_index"]
    elif group == "bottom":
        indices = target_test["bottom_index"]
    else:
        raise ValueError("Invalid group. Choose from 'Overall', 'top', 'medium', 'bottom'.")

    # Filter ranks for the selected group
    original_rank_group = original_rank[indices]
    rewrited_rank_group = rewrited_rank[indices, :]

    # Sample indices to reduce clutter
    sampled_indices = np.arange(0, len(indices), max(1, len(indices) // 100))

    # Plot original rank
    plt.figure(figsize=(10, 6))
    plt.plot(
        sampled_indices,
        original_rank_group[sampled_indices],
        label="Original Rank",
        color="blue",
        marker="o",
        markersize=4,
        linestyle="-",
        alpha=0.8,
    )

    # Plot rewritten ranks
    num_rewrite = rewrited_rank_group.shape[1]
    for i in range(num_rewrite):
        plt.plot(
            sampled_indices,
            rewrited_rank_group[sampled_indices, i],
            label=f"Rewritten Version {i + 1}",
            linestyle="--",
            alpha=0.7,
        )

    # Add labels, title, and legend
    plt.xlabel("News Index (Sampled)")
    plt.ylabel("Rank")
    plt.title(f"Rank Distribution for {group.capitalize()} Group, Mode: {rewrite_mode}")
    plt.gca().invert_yaxis()  # Lower ranks (better) should be on top
    plt.legend()
    plt.grid(alpha=0.5)
    plt.tight_layout()

    # Save or show the plot
    plt.savefig(f'{model_name}_fig_rewritted_distribution_{rewrite_mode}_{group}.pdf', bbox_inches='tight', pad_inches=0.1)

    
def plot_rank_histogram(original_rank, rewrited_rank, target_test, group="Overall", rewrite_mode="writing_style", bins=50):
    """
    Plot the histogram of ranks for original and rewritten news items with no overlap.
    Args:
        original_rank: Array of ranks for original news.
        rewrited_rank: Array of ranks for rewritten news (shape: num_items x num_rewrites).
        target_test: Dictionary containing group indices.
        group: The group to plot ('Overall', 'top', 'medium', 'bottom').
        rewrite_mode: 'writing_style', 'sentiment', 'persona'.
        bins: Number of bins for the histogram.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # Select indices based on the group
    if group == "Overall":
        indices = range(len(target_test["index"]))
    elif group == "top":
        indices = target_test["top_index"]
    elif group == "medium":
        indices = target_test["medium_index"]
    elif group == "bottom":
        indices = target_test["bottom_index"]
    else:
        raise ValueError("Invalid group. Choose from 'Overall', 'top', 'medium', 'bottom'.")

    # Filter ranks for the selected group
    original_rank_group = original_rank[indices]
    rewrited_rank_group = rewrited_rank[indices, :]

    # Set up the bins
    bin_edges = np.linspace(
        min(original_rank_group.min(), rewrited_rank_group.min()),
        max(original_rank_group.max(), rewrited_rank_group.max()),
        bins + 1
    )
    bin_width = bin_edges[1] - bin_edges[0]
    bar_width = bin_width / (rewrited_rank_group.shape[1] + 2)  # Ensure non-overlapping bars

    # Plot histogram
    plt.figure(figsize=(10, 6))

    # Plot histogram for original rank
    plt.bar(
        bin_edges[:-1] + bar_width / 2,  # Shift for the first bar
        np.histogram(original_rank_group, bins=bin_edges)[0],
        width=bar_width,
        color="blue",
        alpha=0.8,
        label="Original Rank",
        edgecolor="black",
    )

    # Plot histograms for rewritten ranks
    num_rewrite = rewrited_rank_group.shape[1]
    for i in range(num_rewrite):
        plt.bar(
            bin_edges[:-1] + bar_width * (i + 1.5),  # Shift for subsequent bars
            np.histogram(rewrited_rank_group[:, i], bins=bin_edges)[0],
            width=bar_width,
            alpha=0.6,  # More transparency for rewritten versions
            label=f"Rewritten Version {i + 1}",
            edgecolor="black",
            color=f"C{i + 1}"  # Use Matplotlib's default color cycle
        )

    # Add labels, title, and legend
    plt.xlabel("Rank")
    plt.ylabel("Frequency")
    plt.title(f"Rank Distribution Histogram for {group.capitalize()} Group, Mode: {rewrite_mode}")
    plt.gca().invert_xaxis()  # Lower ranks (better) should be on the left
    plt.legend()
    plt.grid(alpha=0.5)
    plt.tight_layout()
    plt.savefig(f'{model_name}_fig_histogram_{rewrite_mode}_{group}.pdf', bbox_inches='tight', pad_inches=0.1)



def show_gap_result(original_rank, rewrited_rank, gap_list, target_test, mode):
    num_rewrite = rewrited_rank.shape[1]

    def calculate_and_print_ranks(group_indices, group_name="Overall"):
        print("===========")
        # statis_mode = "Highest"
        # # Max statistics
        # max_original = np.min(original_rank[group_indices])
        # max_rewritten = [np.min(rewrited_rank[group_indices, i]) for i in range(num_rewrite)]
        # print(f"  {group_name} Max: Original: {max_original}, Rewritten: {max_rewritten}")


        statis_mode = "Mean"
        # Mean statistics
        mean_original = np.mean(original_rank[group_indices])
        mean_rewritten = [np.mean(rewrited_rank[group_indices, i]) for i in range(num_rewrite)]
        print(f"  {group_name} Mean: Original: {mean_original}, Rewritten: {mean_rewritten}")


    def print_group_stats(group_indices, group_name="Overall", dim=0):
        print("===========")
        if group_name == "Overall":

            overall_success = (gap_list[group_indices] > 0).any(axis=1).sum()
            per_version_success = [(gap_list[group_indices, i] > 0).sum() for i in range(num_rewrite)]
            print(f"  Success rate: Overall: {overall_success}/{len(group_indices)}, Per rewrited version: {per_version_success} \n")

            rank_intervals = [(10000, 20000), (10000, 30000), (10000, 40000), (10000, 50000)]
            for low, high in rank_intervals:
                in_interval = ((original_rank[group_indices] >= low) &
                            (original_rank[group_indices] < high)).sum()
                improved_to_top_100 = ((original_rank[group_indices] >= low) &
                                    (original_rank[group_indices] < high) &
                                    (rewrited_rank[group_indices, :] <= 10000).any(axis=1)[:, np.newaxis]).sum()
                print(f"  Initial ranks in ({low}, {high}): {in_interval}, Improved to Top 10000: {improved_to_top_100}")
        elif group_name == "Version":
            print(f"Version {dim} -- \n")
            # Initial rank distributions and improvements
            rank_intervals = [(10000, 20000), (10000, 30000), (10000, 40000), (10000, 50000)]
            for low, high in rank_intervals:
                in_interval = ((original_rank[group_indices] >= low) &
                            (original_rank[group_indices] < high)).sum()
                improved_to_top_100 = ((original_rank[group_indices] >= low) &
                                    (original_rank[group_indices] < high) &
                                    (rewrited_rank[group_indices, dim] <= 10000)[:, np.newaxis]).sum()
                print(f"\t Initial ranks in ({low}, {high}): {in_interval}, Improved to Top 10000: {improved_to_top_100}")
    # Overall statistics
    calculate_and_print_ranks(range(len(target_test["index"])), "Overall")
    calculate_and_print_ranks(target_test["top_index"], "Top")
    calculate_and_print_ranks(target_test["medium_index"], "Medium")
    calculate_and_print_ranks(target_test["bottom_index"], "Bottom")

    print_group_stats(range(len(target_test["index"])), "Overall")
    # Group-wise statistics
    for i in range(num_rewrite):
        print_group_stats(range(len(target_test["index"])), "Version", dim=i)



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
            "news_abstract": target_test['news_abstract']
        },
        "rewrited": {
            "index": target_test["index"],
            "news_ID": target_test["news_ID"],
            "rank": [[] for _ in rewrited_news_collection['news_title'][0]],
            "inverse_rank": [[] for _ in rewrited_news_collection['news_title'][0]],
            "rank_gap": [[] for _ in rewrited_news_collection['news_title'][0]],
            "inverse_rank_gap": [[] for _ in rewrited_news_collection['news_title'][0]],
            "news_title": rewrited_news_collection['news_title'],
            "news_abstract": rewrited_news_collection['news_abstract']
        }
    }

    result["original"]["rank"], result["original"]["inverse_rank"] = compute_average_rank(
        result["original"]["news_ID"], original_user2vector, original_news2vector
    )


    # Step 3: Process Rewritten News and Get Their Vectors
    print("Calculating vectors for rewritten news...")
    rewritten_news2vector = {}
    num_rewrite = len(rewrited_news_collection["news_title"][0])

    del news_dataset.news_parsed

    for i in range(num_rewrite):
        tmp_news_dataset = copy.deepcopy(news_dataset)
        tmp_rewrited_news_dataset = replace_rewrite_news(tmp_news_dataset, rewrited_news_collection, position=i)
        tmp_rewrited_news2vector = get_news2vector(tmp_rewrited_news_dataset, model)
        tmp_rewrited_user2vector = get_user2vector(user_dataset, tmp_rewrited_news2vector, model)
        # Step 4: compute average rank of rewrited target news among all users, and get the gap value
        result["rewrited"]["rank"][i], result["rewrited"]["inverse_rank"][i] = compute_average_rank(
            result["rewrited"]["news_ID"], tmp_rewrited_user2vector, tmp_rewrited_news2vector
        )

    # Here: concate the list in result["rewrited"]["rank"] and result["rewrited"]["inverse_rank"] as numpy.
    result["rewrited"]["rank"] = np.stack(result["rewrited"]["rank"], axis=-1)
    result["rewrited"]["inverse_rank"] = np.stack(result["rewrited"]["inverse_rank"], axis=-1)
    # step 5: Calculate rank_gap and inverse_rank_gap, 
    # and get the max rank_gap and inverse_rank_gap, along with the max rank_gap and inverse_rank_gap in each popularity group

    result["rewrited"]["rank_gap"] = - (result["rewrited"]["rank"] - result["original"]["rank"][:, np.newaxis])
    result["rewrited"]["inverse_rank_gap"] = result["rewrited"]["inverse_rank"] - result["original"]["inverse_rank"][:, np.newaxis]
    
    # show_gap_result(result["original"]["rank"][:, np.newaxis], result["rewrited"]["rank"], result["rewrited"]["rank_gap"], target_test, mode="rank")

    return result



if __name__ == '__main__':
    print('Using device:', device)
    print(f'Evaluating model {model_name}')
    # Don't need to load pretrained word/entity/context embedding
    # since it will be loaded from checkpoint later
    model = Model(config).to(device)
    from train import latest_checkpoint  # Avoid circular imports
    checkpoint_path = latest_checkpoint(os.path.join(config.current_data_path + '/checkpoint', config.pretrained_mode, model_name))
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
    rewrited_target_path = path.join(target_dir, f'{model_name}_rewrited_target_' + config.target_trainORtest + "_" + config.diverse_mode + '.pkl')
    
    diverse_version = ['writing_style', 'sentiment', 'persona']
    result = 0
    for diverse in diverse_version:
        if not result:
            with open(path.join(target_dir, f'{model_name}_rewrited_target_' + config.target_trainORtest + "_" + diverse + '_metrics_result.pkl'), "rb") as f:
                result = pickle.load(f)
        else:
            with open(path.join(target_dir, f'{model_name}_rewrited_target_' + config.target_trainORtest + "_" + diverse + '_metrics_result.pkl'), "rb") as f:
                result_tmp = pickle.load(f)
            result["rewrited"]["rank"] = np.concatenate((result["rewrited"]["rank"], result_tmp["rewrited"]["rank"]), axis=1)
            result["rewrited"]["rank_gap"] = np.concatenate((result["rewrited"]["rank_gap"], result_tmp["rewrited"]["rank_gap"]), axis=1)

    with open(target_path, 'rb') as f:
        target_test = pickle.load(f)
        
    show_gap_result(result["original"]["rank"][:, np.newaxis], result["rewrited"]["rank"], result["rewrited"]["rank_gap"], target_test, mode="ALL")

    # plot_rank_distribution(result["original"]["rank"][:, np.newaxis], result["rewrited"]["rank"], target_test, group="Overall", rewrite_mode=config.diverse_mode)
    # plot_rank_histogram(result["original"]["rank"][:, np.newaxis], result["rewrited"]["rank"], target_test, group="Overall", rewrite_mode=config.diverse_mode)