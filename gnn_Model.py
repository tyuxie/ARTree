import torch
import torch.nn as nn
import math
import numpy as np 
from gnnModels_slim import GNNStack, GatedGraphConv, IDConv, GraphPooling, EdgePooling, MeanStdPooling, TimeEmbedding

class GNN_GRUFourierModel(nn.Module):
    def __init__(self, ntips, hidden_dim=100, num_layers=1, gnn_type='edge', aggr='sum', edge_aggr='max', project=False, bias=True, norm_type='', device=torch.device('cpu'), **kwargs):
        super().__init__()
        self.ntips = ntips
        self.leaf_features = torch.eye(self.ntips, device=device)
        self.hidden_dim = hidden_dim

        if gnn_type == 'identity':
            self.gnn = IDConv()
        elif gnn_type != 'ggnn':
            self.gnn = GNNStack(self.ntips, hidden_dim, num_layers=num_layers, bias=bias, gnn_type=gnn_type, aggr=aggr, project=project, norm_type=norm_type, device=device)
        else:
            self.gnn = GatedGraphConv(hidden_dim, num_layers=num_layers, bias=bias, device=device)

        self.gru = nn.GRUCell(hidden_dim, hidden_dim)

        if gnn_type == 'identity':
            self.pooling_net = EdgePooling(self.ntips, hidden_dim, bias=bias, aggr=edge_aggr, norm_type=norm_type, device=device)
        else:
            self.pooling_net = EdgePooling(hidden_dim, hidden_dim, bias=bias, aggr=edge_aggr, norm_type=norm_type, device=device)
        self.time_embedding = TimeEmbedding(hidden_dim, bias=bias, device=device)
        self.device = device

    def node_embedding(self, tree, level):
        name_dict = {}
        j = level
        rel_pos = np.arange(max(4,2*level-4))
        for node in tree.traverse('postorder'):
            if node.is_leaf():
                node.c = 0
                node.d = self.leaf_features[node.name]
            else:
                child_c, child_d = 0., 0.
                for child in node.children:
                    child_c += child.c
                    child_d += child.d
                node.c = 1./(3. - child_c) 
                node.d = node.c * child_d
                if node.name != '':
                    rel_pos[node.name] = j
                node.name, j = j, j+1
            name_dict[node.name] = node
        
        node_features, node_idx_list, edge_index = [], [], []            
        for node in tree.traverse('preorder'):
            neigh_idx_list = []
            if not node.is_root():
                node.d = node.c * node.up.d + node.d
                # parent_idx_list.append(node.up.name)
                neigh_idx_list.append(node.up.name)
                
                if not node.is_leaf():
                    neigh_idx_list.extend([child.name for child in node.children])
                else:
                    neigh_idx_list.extend([-1, -1])              
            else:
                neigh_idx_list.extend([child.name for child in node.children])
            
            edge_index.append(neigh_idx_list)                
            node_features.append(node.d)
            node_idx_list.append(node.name)
        
        branch_idx_map = torch.sort(torch.tensor(node_idx_list,device=self.device).long(), dim=0, descending=False)[1]
        # parent_idxes = torch.LongTensor(parent_idx_list)
        edge_index = torch.tensor(edge_index,device=self.device).long() 
        # pdb.set_trace()
        
        return torch.index_select(torch.stack(node_features), 0, branch_idx_map), edge_index[branch_idx_map], torch.from_numpy(rel_pos).to(self.device), name_dict

    def init_hidden(self):
        self.h = torch.zeros(size=(4, self.hidden_dim), device=self.device)

    def forward(self, tree, t, level=None):
        temb = self.time_embedding(t)
        node_features, edge_index, rel_pos, name_dict = self.node_embedding(tree, level)

        nnodes = 2*level-2
        if level == 3:
            self.init_hidden()
        else:
            next_hidden = torch.zeros(size=(nnodes, self.hidden_dim), device=self.device)
            next_hidden[rel_pos] += self.h
            self.h = next_hidden
        node_features = self.gnn(node_features, edge_index)
        self.h = self.gru(node_features, self.h)
        return self.pooling_net(self.h, edge_index[:-1,0], temb), name_dict
     
    def mp_forward(self, node_features, edge_index, rel_pos, t):
        temb = self.time_embedding(t)
        batch_size, nnodes = edge_index.shape[0], edge_index.shape[1]
        compact_node_features = node_features.view(-1, node_features.shape[-1])
        compact_edge_index = torch.where(edge_index>-1, edge_index + torch.arange(0, batch_size, device=self.device)[:,None,None]*nnodes, -1)
        compact_parent_index = compact_edge_index[:,:-1,0].contiguous().view(-1)
        compact_edge_index = compact_edge_index.view(-1, compact_edge_index.shape[-1])

        compact_node_features = self.gnn(compact_node_features, compact_edge_index)
        
        if nnodes==4:
            self.h = torch.zeros(size=(batch_size*4, self.hidden_dim), device=self.device)
        else:
            next_hidden = torch.zeros(size=(batch_size*nnodes, self.hidden_dim), device=self.device)
            compact_rel_pos = (rel_pos+torch.arange(0, batch_size, device=self.device)[:,None]*nnodes).view(-1)
            next_hidden[compact_rel_pos] += self.h
            self.h = next_hidden
        self.h = self.gru(compact_node_features, self.h)
        return self.pooling_net(self.h.view(batch_size, nnodes, self.h.shape[-1]), compact_parent_index, temb)


class GNN_BranchModel(nn.Module):          
    def __init__(self, ntips, hidden_dim=100, num_layers=1, gnn_type='gcn', aggr='sum', project=False, bias=True, device=torch.device('cpu'), **kwargs):
        super().__init__()
        self.ntips = ntips
        self.leaf_features = torch.eye(self.ntips, device=device)
        
        if gnn_type == 'identity':
            self.gnn = IDConv()
        elif gnn_type != 'ggnn':
            self.gnn = GNNStack(self.ntips, hidden_dim, num_layers=num_layers, bias=bias, gnn_type=gnn_type, aggr=aggr, project=project, device=device)
        else:
            self.gnn = GatedGraphConv(hidden_dim, num_layers=num_layers, bias=bias, device=device)
            
        if gnn_type == 'identity':
            self.mean_std_net = MeanStdPooling(self.ntips, hidden_dim, bias=bias, device=device)
        else:
            self.mean_std_net = MeanStdPooling(hidden_dim, hidden_dim, bias=bias, device=device)
        
        self.device = device
    
    def node_embedding(self, tree):
        for node in tree.traverse('postorder'):
            if node.is_leaf():
                node.c = 0
                node.d = self.leaf_features[node.name]
            else:
                child_c, child_d = 0., 0.
                for child in node.children:
                    child_c += child.c
                    child_d += child.d
                node.c = 1./(3. - child_c)
                node.d = node.c * child_d
        
        node_features, node_idx_list, edge_index = [], [], []            
        for node in tree.traverse('preorder'):
            neigh_idx_list = []
            if not node.is_root():
                node.d = node.c * node.up.d + node.d
                neigh_idx_list.append(node.up.name)
                
                if not node.is_leaf():
                    neigh_idx_list.extend([child.name for child in node.children])
                else:
                    neigh_idx_list.extend([-1, -1])              
            else:
                neigh_idx_list.extend([child.name for child in node.children])
            
            edge_index.append(neigh_idx_list)                
            node_features.append(node.d)
            node_idx_list.append(node.name)
        
        branch_idx_map = torch.sort(torch.tensor(node_idx_list, device=self.device).long(), dim=0, descending=False)[1]
        # parent_idxes = torch.LongTensor(parent_idx_list)
        edge_index = torch.tensor(edge_index, device=self.device).long()
        # pdb.set_trace()
        
        return torch.index_select(torch.stack(node_features), 0, branch_idx_map), edge_index[branch_idx_map]
    
    
    def mean_std(self, tree, **kwargs):
        node_features, edge_index = self.node_embedding(tree)
        node_features = self.gnn(node_features, edge_index)

        return self.mean_std_net(node_features, edge_index[:-1, 0])
            
    
    def sample_branch_base(self, n_particles):
        samp_log_branch = torch.randn(size=(n_particles, 2*self.ntips-3), device=self.device)
        return samp_log_branch, torch.sum(-0.5*math.log(2*math.pi) - 0.5*samp_log_branch**2, -1)
    
    
    def forward(self, tree_list):
        mean, std = zip(*map(lambda x: self.mean_std(x), tree_list)) 
        mean, std = torch.stack(mean, dim=0), torch.stack(std, dim=0)
        samp_log_branch, logq_branch = self.sample_branch_base(len(tree_list))
        samp_log_branch, logq_branch = samp_log_branch * std.exp() + mean - 2.0, logq_branch - torch.sum(std, -1)
        return samp_log_branch, logq_branch 