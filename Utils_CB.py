import numpy as np
from scipy.sparse import coo_matrix
import torch
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import InMemoryDataset, Data
import pingouin as pg
import pandas as pd



def fisher_z_transform(correlation_matrix, epsilon=1e-5):
    return 0.5 * np.log((1 + correlation_matrix) / (1 - correlation_matrix + epsilon))


  
def create_graph(X, Y, start, end, region=True):
    X_adjgraph=[]
    X_featgraph = []

    for i in range(len(Y)):
        if region == True:
            bold_matrix = X[i][start:end,:] #RxR ........?
        else:
            bold_matrix = np.transpose(X[i][start:end,:]) #TxT
        
        window_data1 = np.corrcoef(bold_matrix)
        correlation_matrix_fisher = fisher_z_transform(window_data1)
        correlation_matrix_fisher = np.around(correlation_matrix_fisher, 8)
        knn_graph = compute_KNN_graph(correlation_matrix_fisher)

        if region == True:
            X_featgraph.append(correlation_matrix_fisher)
        else:
            X_featgraph.append(bold_matrix)
            
        X_adjgraph.append(knn_graph)

    return X_featgraph, X_adjgraph, Y



def to_tensor(X_featgraph, X_adjgraph, Y):
    datalist = []
    
    for i in range(len(Y)):
        ty = Y[i]

        y = torch.tensor([ty]).long()

        adjacency = X_adjgraph[i]
        feature = X_featgraph[i]

        x = torch.from_numpy(feature).float()
        adj= adjacency
        adj = torch.from_numpy(adj).float()
        edge_index, edge_attr = dense_to_sparse(adj)
        
        datalist.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y))

    return datalist



def compute_KNN_graph(matrix, k_degree=10):
    """ Calculate the adjacency matrix from the connectivity matrix."""

    matrix = np.abs(matrix)
    idx = np.argsort(-matrix)[:, 0:k_degree]
    matrix.sort()
    matrix = matrix[:, ::-1]
    matrix = matrix[:, 0:k_degree]

    A = adjacency(matrix, idx).astype(np.float32)

    return A


def adjacency(dist, idx):

    m, k = dist.shape
    assert m, k == idx.shape
    assert dist.min() >= 0

    # Weight matrix.
    I = np.arange(0, m).repeat(k)
    J = idx.reshape(m * k)
    V = dist.reshape(m * k)
    W = coo_matrix((V, (I, J)), shape=(m, m))

    # No self-connections.
    W.setdiag(0)

    # Non-directed graph.
    bigger = W.T > W
    W = W - W.multiply(bigger) + W.T.multiply(bigger)

    return W.todense()

