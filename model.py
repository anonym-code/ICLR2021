import torch
import utils as u
import torch.nn as nn
# from torch_geometric.nn import MassagePassing
# from torch_geometric.utils import degree
from torch_geometric.nn import GCNConv

class EGNNConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, out_dim, bias=False)
        self.ELU = nn.ELU()

    def forward(self, X, E):
        X = self.lin1(X)
        X = torch.einsum('ijp,jk->ipk', E, X)
        X = X.view(X.size(0), -1)
        newX = self.ELU(X)
        return newX


class EGNNC(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, edge_dim, num_node, normalize=False, dropout=0.1):
        super().__init__()
        self.Embed = nn.Embedding(num_node, in_dim)
        self.Conv1 = EGNNConv(in_dim=in_dim, out_dim=hidden_dim)
        self.Dropout = nn.Dropout(dropout)
        self.Conv2 = EGNNConv(in_dim=hidden_dim*edge_dim, out_dim=hidden_dim)
        self.lin = nn.Linear(hidden_dim*edge_dim, out_dim)
        self.normalize = normalize

    def forward(self, E, node_ids, node_feature=None):
        E = E.cuda()
        if node_feature != None:
            X = node_feature
        else:
            X = self.Embed(node_ids)
        if self.normalize:
            E = u.DS_normalize(E)
        X = self.Dropout(self.Conv1(X, E))
        X = self.Dropout(self.Conv2(X, E))
        return self.lin(X)


class Classifier(torch.nn.Module):
    def __init__(self,args, in_features, out_features=2):
        super(Classifier,self).__init__()
        activation = torch.nn.ReLU()
        self.mlp = torch.nn.Sequential(torch.nn.Linear(in_features = in_features,
                                                       out_features =args.gcn_parameters['cls_feats']),
                                       activation,
                                       torch.nn.Linear(in_features = args.gcn_parameters['cls_feats'],
                                                       out_features = out_features))

    def forward(self,x):
        return self.mlp(x)


class GC2N(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.1):
        super().__init__()
        self.Conv1 = GCNConv(in_channels=in_dim, out_channels=hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.Conv2 = GCNConv(in_channels=hidden_dim, out_channels=out_dim)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index, edge_weight):
        x = self.relu(self.Conv1(x, edge_index, edge_weight=edge_weight))
        x = self.dropout(x)
        x = self.relu(self.Conv2(x, edge_index, edge_weight=edge_weight))
        return x

class EGNNCSp(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, edge_dim, num_node):
        super().__init__()
        self.embed = nn.Embedding(num_node, in_dim)
        self.Conv = GC2N(in_dim, hidden_dim, out_dim)
        self.num_node = num_node
        self.lin = nn.Linear(out_dim*edge_dim, out_dim)

    def forward(self, edge_index_list,node_ids, node_feature=None):
        if node_feature:
            x = node_feature
        else:
            x = self.embed(node_ids)
        results = []
        for index in edge_index_list:
            edge_index = index._indices()[1:].cuda()
            edge_weight = index._values().float().cuda()
            
            n_x = self.Conv(x, edge_index, edge_weight=edge_weight)
            # newx = torch.zeros(num_node, out_dim)
            # newx[edge_index[1]] = n_x
            results.append(n_x)        

        nn_x = torch.cat(results, dim=1)
        return self.lin(nn_x)