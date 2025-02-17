import os
import scipy.io
import numpy as np
import random
import pandas as pd
import torch
from Utils import *
from Utils_CB import * 
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


seed=89
atlas_name= "AAL"
dataset_name = "MDDvHC"


if atlas_name == "AAL":
    start = 0
    end = 116
elif atlas_name == "Craddock":
    start = 228
    end = 428
elif atlas_name == "Dosenbach":
    start = 1408
    end = 1568
else:
    exit()




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


from sklearn.preprocessing import MinMaxScaler

def normalize_to_0_1(matrix):
    scaler = MinMaxScaler()  # This scales data between 0 and 1
    normalized_matrix = scaler.fit_transform(matrix)
    return normalized_matrix



def get_atlas_data(X_new):
    X_loaded = [X_new[key] for key in X_new.files]
    X_loaded = [normalize(matrix) for matrix in X_loaded]
    start = 0
    end = X_loaded[0].shape[0]
    Y_loaded = np.load(f'/home/vipul/{dataset_name}/{atlas_name}/Y.npy')
    # print('size of each BOLD',X_loaded[0].shape)
    # print('Labels are:',Y_loaded)
    # print('length of X',len(X_loaded))
    # print('length of Y',len(Y_loaded))
    return X_loaded, Y_loaded

