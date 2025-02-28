import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class GCN(nn.Module):
    def __init__(self, configs):
        super(GCN, self).__init__()

        # GCN layers for the time domain
        self.gcn1_t = GCNConv(116, 128)
        self.gcn2_t = GCNConv(128, 64)
        self.gcn3_t = GCNConv(64, 32)  # Additional layer

        # GCN layers for the frequency domain
        self.gcn1_f = GCNConv(101, 128)
        self.gcn2_f = GCNConv(128, 64)
        self.gcn3_f = GCNConv(64, 32)  # Additional layer

        # Batch normalization layers for stability
        self.batch_norm1_t = nn.BatchNorm1d(128)
        self.batch_norm2_t = nn.BatchNorm1d(64)
        self.batch_norm3_t = nn.BatchNorm1d(32)  
        self.batch_norm1_f = nn.BatchNorm1d(128)
        self.batch_norm2_f = nn.BatchNorm1d(64)
        self.batch_norm3_f = nn.BatchNorm1d(32) 

        # Projectors for creating embeddings from GCN outputs
        self.projector_t = nn.Sequential(
            nn.Linear(32, 64),  # Adjust to 32, as output of last GCN layer
            nn.ReLU(),
            nn.Linear(64, 32)
        )

        self.projector_f = nn.Sequential(
            nn.Linear(32, 64),  # Adjust to 32, as output of last GCN layer
            nn.ReLU(),
            nn.Linear(64, 32)
        )

    def forward(self, data_t, data_f):
        # Process time-domain graph data with edge weights
        x_t = self.gcn1_t(data_t.x, data_t.edge_index, edge_weight=data_t.edge_attr)
        x_t = self.batch_norm1_t(x_t)
        x_t = F.relu(x_t)
        x_t = self.gcn2_t(x_t, data_t.edge_index, edge_weight=data_t.edge_attr)
        x_t = self.batch_norm2_t(x_t)
        x_t = F.relu(x_t)
        x_t = self.gcn3_t(x_t, data_t.edge_index, edge_weight=data_t.edge_attr)
        x_t = self.batch_norm3_t(x_t)
        x_t = F.relu(x_t)

        x_t_pool = global_mean_pool(x_t, data_t.batch)
        h_time = x_t_pool
        z_time = self.projector_t(h_time)

        # Process frequency-domain graph data with edge weights
        x_f = self.gcn1_f(data_f.x, data_f.edge_index, edge_weight=data_f.edge_attr)
        x_f = self.batch_norm1_f(x_f)
        x_f = F.relu(x_f)
        x_f = self.gcn2_f(x_f, data_f.edge_index, edge_weight=data_f.edge_attr)
        x_f = self.batch_norm2_f(x_f)
        x_f = F.relu(x_f)
        x_f = self.gcn3_f(x_f, data_f.edge_index, edge_weight=data_f.edge_attr)
        x_f = self.batch_norm3_f(x_f)
        x_f = F.relu(x_f)

        x_f_pool = global_mean_pool(x_f, data_f.batch)
        h_freq = x_f_pool
        z_freq = self.projector_f(h_freq)

        return h_time, z_time, h_freq, z_freq, x_t, x_f


"""Downstream classifier only used in finetuning"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class TargetClassifier(nn.Module):
    def __init__(self, configs):
        super(TargetClassifier, self).__init__()
        self.fc1 = nn.Linear(2*32, 64)  
        self.fc2 = nn.Linear(64, 32)    
        self.logits = nn.Linear(32, configs.num_classes_target)  

    def forward(self, emb):
        emb = torch.sigmoid(self.fc1(emb)) 
        emb = F.relu(self.fc2(emb))      
        pred = self.logits(emb)            
        return pred



