import torch
import torch.nn as nn
import torch.nn.functional as F
import math


from torch import Tensor


norm_funcs = {'id': nn.Identity, 'batch': nn.BatchNorm1d, 'layer': nn.LayerNorm}
def uniform(size, value):
    if isinstance(value, Tensor):
        bound = 1. / math.sqrt(size)
        value.data.uniform_(-bound, bound)


        
class IDConv(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, *args):
        return x
        

class GCNConv(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False, device=torch.device('cpu'), **kwargs):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.transform = nn.Linear(self.in_channels, self.out_channels, bias=False)
        self.device = device

    def forward(self, x, edge_index, *args):
        node_degree = torch.sum(edge_index >= 0, dim=-1, keepdim=True, dtype=torch.float) + 1.0
        x = self.transform(x) / node_degree**0.5
        
        node_feature_padded = torch.cat((x, torch.zeros(size=(1, self.out_channels), device=self.device)), dim=0)
        neigh_feature = node_feature_padded[edge_index] 
        node_and_neigh_feature = torch.cat((neigh_feature, x.view(-1, 1, self.out_channels)), dim=1)
        
        return torch.sum(node_and_neigh_feature, dim=1) / node_degree**0.5
        

class SAGEConv(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True, aggr='mean', project=False, device=torch.device('cpu'), **kwargs):
        super().__init__()
        self.aggr = aggr
        self.in_channels, self.out_channels = in_channels, out_channels        
        self.proj = project
        if project:
            self.mlp = nn.Sequential(nn.Linear(self.in_channels, self.out_channels, bias=bias),
                                     nn.ELU(),)
            self.transform = nn.Linear(self.in_channels + self.out_channels, self.out_channels, bias=False)
        else:
            self.transform = nn.Linear(2*self.in_channels, self.out_channels, bias=bias)
        
        self.device = device
        
    def forward(self, x, edge_index, *args):
        if self.proj:
            node_feature_padded = torch.cat((self.mlp(x), torch.zeros(size=(1, self.out_channels), device=self.device)), dim=0)
        else:
            node_feature_padded = torch.cat((x, torch.zeros(size=(1, self.in_channels), device=self.device)), dim=0)
        neigh_feature = node_feature_padded[edge_index] 
        
        if self.aggr == 'mean':
            node_degree = torch.sum(edge_index >= 0, dim=-1, keepdim=True)
            neigh_feature_agg = torch.sum(neigh_feature, dim=1) / node_degree
        elif self.aggr == 'sum':
            neigh_feature_agg = torch.sum(neigh_feature, dim=1)
        elif self.aggr == 'max':
            neigh_feature_agg = torch.max(neigh_feature, dim=1)[0]
        else:
            raise NotImplementedError
        
        node_and_neigh_feature = torch.cat((x, neigh_feature_agg), dim=1)
        return self.transform(node_and_neigh_feature) 
        
        
class GINConv(nn.Module):
    def __init__(self, in_channels, out_channels, eps=0., train_eps=True, bias=True, norm_type='', device=torch.device('cpu'), **kwargs):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        if train_eps:
            self.eps = nn.Parameter(torch.tensor([eps]))
        else:
            self.register_buffer('eps', torch.tensor([eps], device=self.device))
        
        self.mlp = nn.Sequential(nn.Linear(self.in_channels, self.out_channels, bias=bias),
                                norm_funcs[norm_type](self.out_channels)
                                 )
        self.device = device

    def forward(self, x, edge_index, *args):
        node_feature_padded = torch.cat((x, torch.zeros(size=(1, self.in_channels), device=self.device)), dim=0)
        neigh_feature = node_feature_padded[edge_index]
        
        return self.mlp((1+self.eps)*x + neigh_feature.sum(1))


class EdgeConv(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True, norm_type='', device=torch.device('cpu'), **kwargs):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.mlp = nn.Sequential(nn.Linear(2*self.in_channels, self.out_channels, bias=bias),
                                 norm_funcs[norm_type](self.out_channels),
                                 nn.ELU())
        self.device = device
    def forward(self, x, edge_index, *args):
        node_feature_padded = torch.cat((x, torch.zeros(size=(1, self.in_channels),device=self.device)), dim=0)
        neigh_feature = node_feature_padded[edge_index] 
        
        x_ = x.repeat(1, edge_index.shape[-1]).view(-1, self.in_channels) 
        node_and_neigh_feature = torch.cat((x_, neigh_feature.view(-1, self.in_channels)-x_), dim=-1) 
        output = self.mlp(node_and_neigh_feature)
        output = torch.where(edge_index.flatten().view(-1, 1)!=-1, output, torch.tensor(-math.inf,device=self.device))        
        return torch.max(output.view(-1, edge_index.shape[-1], self.out_channels), dim=1)[0] 
        
        
        
class GNNStack(nn.Module):
    gnnModels = {'gcn': GCNConv,
                 'sage': SAGEConv,
                 'gin': GINConv,
                 'edge': EdgeConv,
             }
    def __init__(self, in_channels, out_channels, num_layers=1, bias=True, aggr='sum', gnn_type='gcn', project=False, norm_type='id', device=torch.device('cpu'), **kwargs):
        super().__init__()
        self.device=device
        self.in_channels, self.out_channels = in_channels, out_channels
        self.num_layers = num_layers
        self.norm_type = norm_type
        self.gconvs = nn.ModuleList()
        self.gconvs.append(self.gnnModels[gnn_type](self.in_channels, self.out_channels, bias=bias, aggr=aggr, project=project, norm_type=norm_type, device=device))
        for i in range(self.num_layers-1):
            self.gconvs.append(self.gnnModels[gnn_type](self.out_channels, self.out_channels, bias=bias, aggr=aggr, project=project,  norm_type=norm_type, device=device))
        
    
    def forward(self, x, edge_index, temb=0.0):
        for i in range(self.num_layers):
            if i > 1:
                x = self.gconvs[i](x + temb, edge_index)
            else:
                x = self.gconvs[i](x, edge_index)
            x = F.elu(x)
        
        return x
        
class MeanStdPooling(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True, **kwargs):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        
        self.net = nn.Sequential(nn.Linear(self.in_channels, self.out_channels, bias=bias),
                                 nn.ELU(),
                                 nn.Linear(self.out_channels, self.out_channels, bias=bias),
                                 nn.ELU(),)
        
        self.readout = nn.Sequential(nn.Linear(self.out_channels, self.out_channels, bias=bias),
                                     nn.ELU(),
                                     nn.Linear(self.out_channels, 2, bias=bias),)
                                     
    
    def forward(self, x, parent_index):
        mean_std = self.net(x)           
        mean_std = torch.max(mean_std[:-1], mean_std[parent_index])                
        mean_std = self.readout(mean_std)
        
        return mean_std[:, 0], mean_std[:, 1]

class EdgePooling(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True, aggr='max', norm_type='id', **kwargs):
        super().__init__()  
        self.in_channels, self.out_channels = in_channels, out_channels

        self.net = nn.Sequential(
            nn.Linear(self.in_channels, self.out_channels, bias=bias),
            norm_funcs[norm_type](self.out_channels),
            nn.ELU(),
            nn.Linear(self.out_channels, self.out_channels, bias=bias),
            norm_funcs[norm_type](self.out_channels),
            nn.ELU()
        )
        self.readout = nn.Sequential(nn.Linear(self.out_channels, self.out_channels, bias=bias),
                                    norm_funcs[norm_type](self.out_channels),
                                     nn.ELU(),
                                     nn.Linear(self.out_channels, 1, bias=bias))

        self.aggr = aggr

    def forward(self, x, parent_index, temb=0.0):
        if len(x.shape) == 3:
            batch, nnodes, nfeatures = x.shape
            edge_info = self.net(x.view(-1, nfeatures) + temb).view(batch, nnodes, -1)
            child_edge_info = edge_info[:,:-1].contiguous().view(-1, edge_info.shape[-1])
            parent_edge_info = edge_info.view(-1, edge_info.shape[-1])[parent_index]
            if self.aggr == 'max':
                edge_info = torch.max(child_edge_info, parent_edge_info)
            elif self.aggr == 'mean':
                edge_info = (child_edge_info+parent_edge_info)/2
            elif self.aggr == 'sum':
                edge_info = child_edge_info+parent_edge_info
            
            edge_info = self.readout(edge_info+temb)
            out = torch.squeeze(edge_info).reshape(-1, nnodes-1)
            return out - torch.logsumexp(out, dim=1, keepdim=True)
        else:
            edge_info = self.net(x + temb)      
            if self.aggr == 'max':
                edge_info = torch.max(edge_info[:-1], edge_info[parent_index])   
            elif self.aggr == 'mean':     
                edge_info = (edge_info[:-1]+ edge_info[parent_index])/2
            elif self.aggr == 'sum':
                edge_info = edge_info[:-1]+ edge_info[parent_index]
            edge_info = self.readout(edge_info + temb)
            out = torch.squeeze(edge_info)
            return out - torch.logsumexp(out, dim=0)
        

class GraphPooling(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True, aggr='mean', **kwargs):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.aggr = aggr
        
        self.net = nn.Sequential(nn.Linear(self.in_channels, self.out_channels, bias=bias),
                                 nn.ELU(),
                                 nn.Linear(self.out_channels, self.out_channels, bias=bias),
                                 nn.ELU(),)
        
        self.readout = nn.Sequential(nn.Linear(self.out_channels, self.out_channels, bias=bias),
                                     nn.ELU(),
                                     nn.Linear(self.out_channels, 1, bias=bias),)
    
    def forward(self, x):
        output = self.net(x)
        if self.aggr == 'mean':
            output = torch.mean(output, dim=0, keepdim=True)
        elif self.aggr == 'sum':
            output = torch.sum(output, dim=0, keepdim=True)
        elif self.aggr == 'max':
            output = torch.max(output, dim=0, keepdim=True)
        else:
            raise NotImplementedError
        
        return self.readout(output)            

class TimeEmbedding(nn.Module):
    def __init__(self, out_channels, bias=True, device=torch.device('cpu')):
        super().__init__()
        self.out_channels = out_channels

        self.coef = torch.arange(0, out_channels, step=2, device=device) / out_channels * math.log(10000)
        self.embedding = nn.Sequential(
            nn.Linear(out_channels, out_channels, bias=bias),
            nn.ELU(),
            nn.Linear(out_channels, out_channels, bias=bias)
        )

    def forward(self, t):
        temb = torch.cat([torch.sin(self.coef * t), torch.cos(self.coef * t)], dim=-1)
        # print(temb.shape)
        assert temb.shape == (self.out_channels,)
        temb = self.embedding(temb.unsqueeze(0))
        return temb

class GRU(nn.Module):
    def __init__(self, hidden_dim, device=torch.device('cpu')):
        super().__init__()

        self.hr_net = nn.Linear(hidden_dim, hidden_dim)
        self.xr_net = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.hz_net = nn.Linear(hidden_dim, hidden_dim)
        self.xz_net = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.hn_net = nn.Linear(hidden_dim, hidden_dim)
        self.xn_net = nn.Linear(hidden_dim, hidden_dim)

    
    def forward(self, x, h):
        r = self.hr_net(h) + self.xr_net(x)
        r = torch.sigmoid(r)

        z = self.hz_net(h) + self.xz_net(x)
        z = torch.sigmoid(z)

        n = self.xn_net(x) + r * self.hn_net(x)
        n = torch.tanh(n)

        return (1-z) * n + z * h