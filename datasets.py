import torch
import numpy as np
import os
from torch.utils.data import Dataset
from utils import mcmc_treeprob, namenum, remove, renamenum_backward, summary
import pickle
import tarfile
import dill

def node_embedding(tree, ntips):
    leaf_features = torch.eye(ntips)
    for node in tree.traverse('postorder'):
        if node.is_leaf():
            node.c = 0
            node.d = leaf_features[node.name]
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
    
    branch_idx_map = torch.sort(torch.tensor(node_idx_list).long(), dim=0, descending=False)[1]
    edge_index = torch.tensor(edge_index).long() 
    

    return torch.index_select(torch.stack(node_features), 0, branch_idx_map), edge_index[branch_idx_map]

def node_embedding_forward(tree, ntips, level):
    leaf_features = torch.eye(ntips)
    name_dict = {}
    j = level
    rel_pos = np.arange(max(2*level-4, 4))
    for node in tree.traverse('postorder'):
        if node.is_leaf():
            node.c = 0
            node.d = leaf_features[node.name]
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
    
    branch_idx_map = torch.sort(torch.tensor(node_idx_list).long(), dim=0, descending=False)[1]
    edge_index = torch.tensor(edge_index).long() 
    
    return torch.index_select(torch.stack(node_features), 0, branch_idx_map), edge_index[branch_idx_map], torch.from_numpy(rel_pos), name_dict


def process_data(dataset, repo):
    emp_tree_freq = mcmc_treeprob('data/short_run_data_DS1-8/' + str(dataset) + '/rep_{}/'.format(repo) + str(dataset) + '.trprobs', 'nexus')
    taxa = sorted(list(emp_tree_freq.keys())[0].get_leaf_names())
    ntips = len(taxa)
    trees, wts = zip(*emp_tree_freq.items())
    wts = np.array(wts) / np.sum(wts)
    path = os.path.join('embed_data',dataset,'repo{}'.format(repo))
    os.makedirs(path, exist_ok=True)

    np.save(os.path.join(path, 'wts.npy'), wts)
    np.save(os.path.join(path, 'taxa.npy'), taxa)
    i = 0
    for tree in trees:
        stat = []
        namenum(tree, taxa)
        subtree = tree.copy()
        for taxon in range(ntips-1,2,-1):
            subtree, pos = remove(subtree, taxon, return_pos=True)
            pos, rel_pos = renamenum_backward(subtree, level=taxon, target=pos, return_rel_pos=True)
            node_features, edge_index = node_embedding(subtree, ntips)
            stat.append(torch.tensor(pos))
            stat.append(edge_index)
            stat.append(node_features)
            if taxon < ntips-1:
                stat.insert(-6,torch.LongTensor(rel_pos))
        stat.insert(-3, torch.LongTensor([-1]))
        stat.reverse()
        with open(os.path.join(path, 'tree_{}.pkl'.format(i)), 'wb') as f:
            pickle.dump(tuple(stat), f)

        tar = tarfile.open(os.path.join(path, 'tree_{}.tar'.format(i)), 'w:gz')
        tar.add(os.path.join(path, 'tree_{}.pkl'.format(i)))
        tar.close()

        os.remove(os.path.join(path, 'tree_{}.pkl'.format(i)))
        i += 1
    return

def process_empFreq(dataset):
    ground_truth_path, samp_size = 'data/raw_data_DS1-8/', 750001
    tree_dict_total, tree_names_total, tree_wts_total = summary(dataset, ground_truth_path, samp_size=samp_size)
    emp_tree_freq = {tree_dict_total[tree_name]:tree_wts_total[i] for i, tree_name in enumerate(tree_names_total)}
    wts = list(emp_tree_freq.values())
    taxa = sorted(list(emp_tree_freq.keys())[0].get_leaf_names())
    ntips = len(taxa)
    path = os.path.join('embed_data',dataset,'emp_tree_freq')
    os.makedirs(path, exist_ok=True)
    np.save(os.path.join(path, 'wts.npy'), wts)
    np.save(os.path.join(path, 'taxa.npy'), taxa)
    i = 0
    for tree in emp_tree_freq.keys():
        stat = []
        namenum(tree, taxa)
        subtree = tree.copy()
        for taxon in range(ntips-1,2,-1):
            subtree, pos = remove(subtree, taxon, return_pos=True)
            pos, rel_pos = renamenum_backward(subtree, level=taxon, target=pos, return_rel_pos=True)
            node_features, edge_index = node_embedding(subtree, ntips)
            stat.append(torch.tensor(pos))
            stat.append(edge_index)
            stat.append(node_features)
            if taxon < ntips-1:
                stat.insert(-6,torch.LongTensor(rel_pos))
        stat.insert(-3, torch.LongTensor([-1]))
        stat.reverse()
        with open(os.path.join(path, 'tree_{}.pkl'.format(i)), 'wb') as f:
            pickle.dump(tuple(stat), f)

        tar = tarfile.open(os.path.join(path, 'tree_{}.tar'.format(i)), 'w:gz')
        tar.add(os.path.join(path, 'tree_{}.pkl'.format(i)))
        tar.close()

        os.remove(os.path.join(path, 'tree_{}.pkl'.format(i)))

        i += 1
    return

class EmbedData(Dataset):
    def __init__(self, dataset, wts, repo='emp', folder=None) -> None:
        super().__init__()
        self.wts = wts
        self.length = len(wts)
    
        if repo == 'emp':
            self.path = os.path.join('..', 'embed_data',dataset,'emp_tree_freq')
            self.folder = folder
            self.folder_path = os.path.join(folder, 'embed_data',dataset,'emp_tree_freq')
        else:
            self.path = os.path.join('..', 'embed_data',dataset,'repo{}'.format(repo))
            self.folder = folder
            self.folder_path = os.path.join(folder, 'embed_data',dataset,'repo{}'.format(repo))

    def __getitem__(self, index):
        tar = tarfile.open(os.path.join(self.path, 'tree_{}.tar'.format(index)) , 'r:gz')
        tar.extractall(path=self.folder)
        tar.close()
        with open(os.path.join(self.folder_path, 'tree_{}.pkl'.format(index)), 'rb') as f:
            stat = pickle.load(f)
        os.remove(os.path.join(self.folder_path, 'tree_{}.pkl'.format(index)))
        return stat
    
    def __len__(self):
        return self.length

class EmbedDataLoader(object):
    def __init__(self, dataset, batch_size, wts=None) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.wts = wts
        self.random = True if isinstance(self.wts, np.ndarray) else False
        self.length = self.dataset.__len__()
        self.position = 0
    
    def nonrandomize(self):
        self.random = False

    def randomize(self):
        self.random = True
    def initialize(self):
        self.position = 0

    def next(self):
        if self.random:
            indexes = np.random.choice(self.length, size=self.batch_size, replace=True, p=self.wts)
        else:
            if self.position + self.batch_size <= self.length:
                indexes = list(range(self.position, self.position+self.batch_size))
                self.position += self.batch_size
            elif self.position < self.length:
                indexes = list(range(self.position, self.length))
                self.position = self.length
            else:
                raise StopIteration
        return self.fetch(indexes)

    def fetch(self, indexes):
        samples = []
        for i in indexes:
            samples.append(self.dataset.__getitem__(i))
        smp_len = len(samples[0])
        
        samples_tuple = []
        for k in range(smp_len):
            samples_tuple.append(torch.stack([samples[j][k] for j in range(len(indexes))]))
        return tuple(samples_tuple)


def get_dataloader(dataset, repo, batch_size=10, maxiter=200000, folder=None):
    path = os.path.join('..', 'embed_data',dataset,'repo{}'.format(repo))
    wts = np.load(os.path.join(path, 'wts.npy'))
    data = EmbedData(dataset, wts, repo, folder=folder)
    dataloader = EmbedDataLoader(data, batch_size=batch_size, wts=wts)
    return dataloader

def get_empdataloader(dataset, batch_size=10, folder=None):
    path = os.path.join('..', 'embed_data',dataset,'emp_tree_freq')
    wts = np.load(os.path.join(path, 'wts.npy'))
    data = EmbedData(dataset, wts, folder=folder)
    empdataloader = EmbedDataLoader(data, batch_size=batch_size, wts=None)
    return empdataloader