import os
import scipy.io
import numpy as np
import random
import pandas as pd
import torch
from Utils import * 
from sklearn.model_selection import train_test_split
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import dense_to_sparse
import torch.nn.functional as func
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.nn import GATConv, ChebConv
import torch.nn as nn
from collections import Counter
import os.path as osp
import pingouin as pg
import csv
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
import numpy as np
from scipy.signal import coherence, welch
import numpy as np
import numpy as np
from scipy.signal import cwt, morlet2
from scipy.stats import spearmanr
from joblib import Parallel, delayed
import numpy as np
from scipy.signal import csd, welch
from augmentations import *
from graph_learning import * 
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


seed=89


# Function to set seed
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

# Environment variables for reproducibility
os.environ['PYTHONHASHSEED'] = str(seed)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Set your seed
set_seed(seed)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(0)



def normalize(matrix):
    scaler = StandardScaler()
    normalized_matrix = scaler.fit_transform(matrix)
    return normalized_matrix



def normalize_to_0_1(matrix):
    scaler = MinMaxScaler()  # This scales data between 0 and 1
    normalized_matrix = scaler.fit_transform(matrix)
    return normalized_matrix


def load_BOLD_and_labels(sourcedata_path, dataset_name, atlas_name):
    X_new = np.load(sourcedata_path)
    BOLD = [X_new[key] for key in X_new.files]
    BOLD = [normalize(matrix) for matrix in BOLD]
    label = np.load(f'../{dataset_name}/{atlas_name}/Y.npy')
    
    return BOLD, label

def generate_masks(demographic_data, train_site, val_site, test_site):
    site_info = demographic_data[['Site']].values
    train_mask = (site_info == train_site)
    # val_mask = (site_info == val_site)
    val_mask = np.isin(site_info, val_site) 
    test_mask = (site_info == test_site)
    return train_mask, val_mask, test_mask



def filter_data_by_mask(data_list, mask):
    return [data_list[i] for i, mask in enumerate(mask) if mask]

def to_tensor__(numpy_array):
    return torch.tensor(numpy_array, dtype=torch.float32)
    
#####################################################################

fs = 0.5    
nperseg = 200

######### site for ABIDE ########
train_site = 'NYU'
val_site = ['LEUVEN_1', 'LEUVEN_2']
test_site = 'LEUVEN'
#################################

######### site for MDD ########
# train_site = 20
# val_site = 21
# test_site = 2
####################################

atlas_name= "AAL"
dataset_name = "ABIDE"
demographic_data = pd.read_csv(f'../{dataset_name}/{atlas_name}/demographics_data.csv')

#####################################################################

def train_get_generator(sourcedata_path, dataset_name, atlas_name,training_mode):
    BOLD, labels = load_BOLD_and_labels(sourcedata_path, dataset_name, atlas_name)
    demographic_data = pd.read_csv(f'../{dataset_name}/{atlas_name}/demographics_data.csv')    
    demographics_sex= demographic_data[['Sex']].values
    train_mask, _, _ = generate_masks(demographic_data, train_site, val_site, test_site )  # Adjust site numbers as needed

    BOLD_filtered = filter_data_by_mask(BOLD, train_mask)
    label_filtered = filter_data_by_mask(labels, train_mask)
    filtered_sex = filter_data_by_mask(demographics_sex, train_mask)

    clean_feat_time, clean_adj_time, labels = dataset_time(BOLD_filtered, label_filtered, pert=False)
    clean_dataset_time = to_tensor(clean_feat_time, clean_adj_time, labels,filtered_sex)

    pert_feat_time, pert_adj_time, labels = dataset_time(BOLD_filtered, label_filtered, pert=True)
    pert_dataset_time = to_tensor(pert_feat_time, pert_adj_time, labels,filtered_sex)
        
    clean_feat_freq, clean_adj_freq, labels = dataset_freq(BOLD_filtered, label_filtered, fs, nperseg, pert=False)  
    clean_dataset_freq = to_tensor(clean_feat_freq, clean_adj_freq, labels,filtered_sex) 
       
    pert_feat_freq, pert_adj_freq, labels = dataset_freq(BOLD_filtered, label_filtered, fs, nperseg, pert=True)
    pert_dataset_freq = to_tensor(pert_feat_freq, pert_adj_freq, labels,filtered_sex)
    
    
    return clean_dataset_time,pert_dataset_time,clean_dataset_freq,pert_dataset_freq, clean_adj_time,clean_adj_freq


def finetune_test_get_generator(sourcedata_path, dataset_name, atlas_name,training_mode):
    BOLD, labels = load_BOLD_and_labels(sourcedata_path, dataset_name, atlas_name)
    demographic_data = pd.read_csv(f'../{dataset_name}/{atlas_name}/demographics_data.csv')
    demographi_sex = demographics_sex= demographic_data[['Sex']].values
    _, val_mask, _ = generate_masks(demographic_data, train_site, val_site, test_site)  # Adjust site numbers as needed
    #print(val_mask)
    filtered_sex = filter_data_by_mask(demographics_sex, val_mask)

    # Process only train data
    BOLD_filtered = filter_data_by_mask(BOLD, val_mask)
    label_filtered = filter_data_by_mask(labels, val_mask)

    # Process time and frequency graphs only for the filtered data
    clean_feat_time, clean_adj_time, labels = dataset_time(BOLD_filtered, label_filtered, pert=False)
    clean_dataset_time = to_tensor(clean_feat_time, clean_adj_time, labels,filtered_sex)

    clean_feat_freq, clean_adj_freq, labels = dataset_freq(BOLD_filtered, label_filtered, fs, nperseg, pert=False)
    clean_dataset_freq = to_tensor(clean_feat_freq, clean_adj_freq, labels,filtered_sex)
    
    
    return clean_dataset_time,clean_dataset_freq, clean_adj_time,clean_adj_freq





