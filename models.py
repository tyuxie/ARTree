import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import time
import math
import numpy as np
from ete3 import TreeNode
from phyloModel import PHY
from gnn_Model import GNN_BranchModel, GNN_GRUFourierModel
from utils import remove, renamenum, add, renamenum_backward, mp_node_embedding, mp_add, mp_renamenum, node_embedding
import gc


class TDE(nn.Module):
    EPS = np.finfo(float).eps
    def __init__(self, dataloader, ntips, hidden_dim=100, num_layers=2, gnn_type='gcn', aggr='sum', edge_aggr='max', project=False, norm_type='', empdataloader=None) -> None:
        super().__init__()
        self.ntips = ntips
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tree_model = GNN_GRUFourierModel(self.ntips, hidden_dim, num_layers=num_layers, gnn_type=gnn_type, aggr=aggr, edge_aggr=edge_aggr, project=project, norm_type=norm_type, device=self.device)
        self.tree_model = self.tree_model.to(self.device)
        self.dataloader = dataloader
        self.empdataloader = empdataloader
        torch.set_num_threads(1)

    
    def nll(self, batch, wts):
        '''Calculate the negative log-likelihood.'''

        assert len(batch) == (self.ntips-3) * 4
        loss = 0.0
        for i in range(3, self.ntips):
            node_features, edge_index, pos, rel_pos = batch[4*(i-3):4*(i-2)]
            node_features = node_features.to(self.device)
            edge_index = edge_index.to(self.device)
            pos = pos.to(self.device)
            rel_pos = rel_pos.to(self.device)
            logits = self.tree_model.mp_forward(node_features, edge_index, rel_pos, i)
            loss += torch.sum(torch.gather(logits, dim=1, index=pos[:,None]).squeeze() * wts)
        return -loss

    def tree_prob(self, batch):
        logprob = 0.0
        assert len(batch) == (self.ntips-3) * 4
        with torch.no_grad():
            for i in range(3, self.ntips):
                node_features, edge_index, pos, rel_pos = batch[4*(i-3):4*(i-2)]
                node_features = node_features.to(self.device)
                edge_index = edge_index.to(self.device)
                pos = pos.to(self.device)
                rel_pos = rel_pos.to(self.device)
                logits = self.tree_model.mp_forward(node_features, edge_index, rel_pos, i)
                logprob += torch.gather(logits, dim=1, index=pos[:,None]).squeeze()
        return logprob.exp().tolist()
        
    def kl_div(self):
        kl_div = 0.0
        probs = []

        self.negDataEnt = np.sum(self.empdataloader.dataset.wts * np.log(np.maximum(self.empdataloader.dataset.wts, self.EPS)))
        self.empdataloader.initialize()
        for i in range(0, self.empdataloader.dataset.length, self.empdataloader.batch_size):
            batch = self.empdataloader.next()
            tree_prob = self.tree_prob(batch) 
            if isinstance(tree_prob, list):
                probs.extend(tree_prob)
            elif isinstance(tree_prob, float):
                probs.append(tree_prob)
            else:
                raise TypeError
        kl_div = self.negDataEnt - np.sum(self.empdataloader.dataset.wts * np.log(np.maximum(probs, self.EPS)))
        return kl_div, probs

    def emp_ll(self):
        probs = []
        self.dataloader.nonrandomize()
        self.dataloader.initialize()
        for i in range(0, self.dataloader.dataset.length, self.dataloader.batch_size):
            batch = self.dataloader.next()
            tree_prob = self.tree_prob(batch)
            if isinstance(tree_prob, list):
                probs.extend(tree_prob)
            elif isinstance(tree_prob, float):
                probs.append(tree_prob)
            else:
                raise TypeError
        self.dataloader.randomize()
        emp_ll =  np.sum(self.dataloader.dataset.wts * np.log(np.maximum(probs, self.EPS)))
        return emp_ll

    def train(self, lr, maxiter=200000, test_freq=1000, anneal_freq=10000, anneal_rate=0.75, save_to_path=None, logger=None, klf=5000):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        run_time = -time.time()
        nlls_track, kldivs, nlls, ells = [], [], [], []

        wts = torch.ones(size=(self.dataloader.batch_size,), device=self.device) / self.dataloader.batch_size
        self.tree_model.train()
        for it in range(1, maxiter+1):
            batch=self.dataloader.next()
            loss = self.nll(batch, wts)
            nlls.append(loss.item())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            gc.collect()
            if it % test_freq == 0:
                run_time += time.time()
                gc.collect()
                logger.info('{} Iter {}:({:.1f}s) NLL Loss {:.4f}'.format(time.asctime(time.localtime(time.time())), it, run_time, np.mean(nlls)))               
                nlls_track.append(np.mean(nlls))
                run_time = -time.time()
                nlls = []           
            if self.empdataloader and it % klf == 0:
                self.tree_model.eval()
                kldiv, pred_prob = self.kl_div()
                ell = self.emp_ll()
                ells.append(ell)
                kldivs.append(kldiv)
                self.tree_model.train()
                run_time += time.time()
                logger.info('>>> Iter {}:({:.1f}s) KL {:.4f} LL {:.4f}'.format(it, run_time, kldivs[-1], ells[-1]))
                run_time = -time.time()

            if it % anneal_freq == 0:
                for g in optimizer.param_groups:
                    g['lr'] *= anneal_rate
                
        if save_to_path is not None:
            torch.save(self.tree_model.state_dict(), save_to_path)

        if self.empdataloader:
            return nlls_track, kldivs, ells
        else:
            return nlls_track

class VBPI(nn.Module):
    EPS = np.finfo(float).eps
    def __init__(self, taxa, data, pden, subModel, emp_tree_freq=None,
                 scale=0.1, hidden_dim_tree=100, hidden_dim_branch=100, num_layers_tree=2, num_layers_branch=2, gnn_type='edge', aggr='sum', edge_aggr='max', project=False, norm_type=''):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.taxa, self.emp_tree_freq = taxa, emp_tree_freq
        self.ntips = len(data)
        self.scale = scale
        self.phylo_model = PHY(data, taxa, pden, subModel, scale=scale, device=self.device)  
        self.log_p_tau = - np.sum(np.log(np.arange(3, 2*self.ntips-3, 2)))

        self.tree_model = GNN_GRUFourierModel(self.ntips, hidden_dim_tree, num_layers=num_layers_tree, gnn_type=gnn_type, aggr=aggr, edge_aggr=edge_aggr, project=project, norm_type=norm_type, device=self.device).to(self.device)
        self.branch_model = GNN_BranchModel(self.ntips, hidden_dim_branch, num_layers=num_layers_branch, gnn_type=gnn_type, aggr=aggr, project=project, device=self.device).to(device=self.device)
        torch.set_num_threads(1)


    def init_tree(self):
        tree = TreeNode(name=3)
        for i in [0,1,2]:
            node = TreeNode(name=i)
            tree.add_child(node)
        return tree

    def sample_tree(self):
        '''sample one tree from the model'''
        tree = self.init_tree()
        logq_tree = 0.0
        for taxon in range(3, self.ntips):
            logprob, name_dict = self.tree_model(tree, taxon, level=taxon)
            pos = torch.multinomial(input=logprob.exp(), num_samples=1)[0].item()
            add(tree, name=taxon, pos=name_dict[pos])
            del name_dict
            logq_tree += logprob[pos]
        renamenum(tree, level=self.ntips)
        return tree, logq_tree

    def process_outputs(self, outputs):
        trees, node_features, edge_index, rel_pos, name_dicts = zip(*outputs)
        node_features = torch.stack(node_features)
        edge_index = torch.stack(edge_index)
        rel_pos =  torch.stack(rel_pos)
        return trees, node_features, edge_index, rel_pos, name_dicts

    def mp_sample_tree(self, n_particles):
        trees = [self.init_tree() for _ in range(n_particles)]
        logq_tree = 0.0
        for taxon in range(3, self.ntips):
            ntips = self.ntips
            trees, node_features, edge_index, rel_pos, name_dicts = self.process_outputs(map(mp_node_embedding, list(zip(trees, [ntips]*n_particles, [taxon]*n_particles))))
            logits = self.tree_model.mp_forward(node_features, edge_index, rel_pos, taxon)
            pos = torch.multinomial(input=logits.exp(), num_samples=1)
            logq_tree += torch.gather(logits, dim=1, index=pos).squeeze(-1)
            pos = pos.squeeze(-1)
            anchor_nodes = [name_dicts[i][pos[i].item()] for i in range(len(name_dicts))]
            trees = map(mp_add, list(zip(trees, [taxon]*n_particles, anchor_nodes)))
        trees = map(mp_renamenum, list(zip(trees, [self.ntips]*n_particles)))
        return list(trees), logq_tree

    def tree_prob(self, batch):
        logprob = 0.0
        assert len(batch) == (self.ntips-3) * 4
        with torch.no_grad():
            for i in range(3, self.ntips):
                node_features, edge_index, pos, rel_pos = batch[4*(i-3):4*(i-2)]
                node_features = node_features.to(self.device)
                edge_index = edge_index.to(self.device)
                pos = pos.to(self.device)
                rel_pos = rel_pos.to(self.device)
                logits = self.tree_model.mp_forward(node_features, edge_index, rel_pos, i)
                logprob += torch.gather(logits, dim=1, index=pos[:,None]).squeeze()
        return logprob.exp().tolist()

    def lower_bound(self, n_particles=1, n_runs=1000):
        lower_bounds = []
        with torch.no_grad():
            for run in range(n_runs):
                samp_trees, logq_tree = zip(*[self.sample_tree() for particle in range(n_particles)])
                samp_log_branch, logq_branch = self.branch_model(samp_trees)
                logll = torch.stack([self.phylo_model.loglikelihood(log_branch, tree) for log_branch, tree in zip(*[samp_log_branch, samp_trees])])
                logp_prior = self.phylo_model.logprior(samp_log_branch)
                logq_tree = torch.stack(logq_tree)       
                lower_bounds.append(torch.logsumexp(logll + logp_prior - logq_tree - logq_branch + self.log_p_tau - math.log(n_particles), 0))            

            lower_bound = torch.stack(lower_bounds).mean()
            
        return lower_bound.item()

    def mp_lower_bound(self, n_runs=1000, step=50):
        def cal_lower_bound(ELBOs, n_particles, n_runs):
            ELBOs = ELBOs.reshape(n_runs, n_particles)
            lower_bounds = torch.logsumexp(ELBOs - math.log(n_particles), -1)
            return torch.mean(lower_bounds).item()
        
        logll_tensor, logq_tree_tensor, logp_prior_tensor, logq_branch_tensor = torch.empty(0), torch.empty(0), torch.empty(0), torch.empty(0)
        
        for i in range(0, n_runs, step):
            n_samples = min(n_runs - i, step)

            with torch.no_grad():
                samp_trees, logq_tree = self.mp_sample_tree(n_samples)
                samp_log_branch, logq_branch = self.branch_model(samp_trees)
                logll = torch.stack([self.phylo_model.loglikelihood(log_branch, tree) for log_branch, tree in zip(*[samp_log_branch, samp_trees])])
                logp_prior = self.phylo_model.logprior(samp_log_branch)
                    
                logll_tensor = torch.cat([logll_tensor,logll], dim=0)
                logq_tree_tensor = torch.cat([logq_tree_tensor,logq_tree], dim=0)
                logp_prior_tensor = torch.cat([logp_prior_tensor,logp_prior], dim=0)
                logq_branch_tensor = torch.cat([logq_branch_tensor, logq_branch], dim=0)

        ELBOs = logll_tensor + logp_prior_tensor - logq_tree_tensor - logq_branch_tensor + self.log_p_tau
        return cal_lower_bound(ELBOs, 1, n_runs//1), cal_lower_bound(ELBOs, 10, n_runs//10), cal_lower_bound(ELBOs, 1000, n_runs//1000)

    def mp_lower_bound_sbnsupport(self, sbn_model, n_runs=1000, step=50):
        def cal_lower_bound(ELBOs, n_particles, n_runs):
            ELBOs = ELBOs.reshape(n_runs, n_particles)
            lower_bounds = torch.logsumexp(ELBOs - math.log(n_particles), -1)
            return torch.mean(lower_bounds).item()
        
        logll_tensor, logq_tree_tensor, logp_prior_tensor, logq_branch_tensor, is_in_support_tensor = torch.empty(0), torch.empty(0), torch.empty(0), torch.empty(0), torch.empty(0, dtype=torch.bool)
        i = 0
        n_runs_more = n_runs
        while True:
            if i >= n_runs_more:
                break
            n_samples = min(n_runs_more - i, step)
            with torch.no_grad():
                samp_trees, logq_tree = self.mp_sample_tree(n_samples)
                is_in_support = torch.tensor([sbn_model.is_in_support(tree) for tree in samp_trees])
                samp_log_branch, logq_branch = self.branch_model(samp_trees)
                logll = torch.stack([self.phylo_model.loglikelihood(log_branch, tree) for log_branch, tree in zip(*[samp_log_branch, samp_trees])])
                logp_prior = self.phylo_model.logprior(samp_log_branch)
                    
                logll_tensor = torch.cat([logll_tensor,logll], dim=0)
                logq_tree_tensor = torch.cat([logq_tree_tensor,logq_tree], dim=0)
                logp_prior_tensor = torch.cat([logp_prior_tensor,logp_prior], dim=0)
                logq_branch_tensor = torch.cat([logq_branch_tensor, logq_branch], dim=0)
                is_in_support_tensor = torch.cat([is_in_support_tensor, is_in_support], dim=0)
            
            n_runs_more += len(is_in_support) - torch.sum(is_in_support)
            i += n_samples
        ELBOs = logll_tensor + logp_prior_tensor - logq_tree_tensor - logq_branch_tensor + self.log_p_tau
        ELBOs = ELBOs[is_in_support_tensor]

        assert len(ELBOs) == n_runs
        return cal_lower_bound(ELBOs, 1, n_runs//1), cal_lower_bound(ELBOs, 10, n_runs//10), cal_lower_bound(ELBOs, 1000, n_runs//1000), n_runs_more-n_runs

    def kl_div(self):
        kl_div = 0.0
        probs = []

        if isinstance(self.emp_tree_freq, dict):
            wts = np.array(list(self.emp_tree_freq.values()))
            negDataEnt = np.sum(wts * np.log(np.maximum(wts, self.EPS)))
            for tree in self.emp_tree_freq.keys():
                stat = []
                subtree = tree.copy()
                for taxon in range(self.ntips-1,2,-1):
                    subtree, pos = remove(subtree, taxon, return_pos=True)
                    pos, rel_pos = renamenum_backward(subtree, level=taxon, target=pos, return_rel_pos=True)
                    node_features, edge_index = node_embedding(subtree, self.ntips)
                    stat.append(torch.tensor(pos).unsqueeze(0))
                    stat.append(edge_index.unsqueeze(0))
                    stat.append(node_features.unsqueeze(0))
                    if taxon < self.ntips-1:
                        stat.insert(-6,torch.LongTensor(rel_pos).unsqueeze(0))
                stat.insert(-3, torch.LongTensor([-1]))
                stat.reverse()

                probs.append(self.tree_prob(stat))
            kl_div = negDataEnt - np.sum(wts * np.log(np.maximum(probs, self.EPS)))
        else:
            negDataEnt = np.sum(self.emp_tree_freq.dataset.wts * np.log(np.maximum(self.emp_tree_freq.dataset.wts, self.EPS)))
            self.emp_tree_freq.initialize()
            for i in range(0, self.emp_tree_freq.dataset.length, self.emp_tree_freq.batch_size):
                batch = self.emp_tree_freq.next()
                tree_prob = self.tree_prob(batch)
                if isinstance(tree_prob, list):
                    probs.extend(tree_prob)
                elif isinstance(tree_prob, float):
                    probs.append(tree_prob)
                else:
                    raise TypeError
            kl_div = negDataEnt - np.sum(self.emp_tree_freq.dataset.wts * np.log(np.maximum(probs, self.EPS)))
        return kl_div, probs

    def rws_lower_bound(self, inverse_temp=1.0, n_particles=10):
        samp_trees, logq_tree = self.mp_sample_tree(n_particles)

        samp_log_branch, logq_branch = self.branch_model(samp_trees)
        logll = torch.stack([self.phylo_model.loglikelihood(log_branch, tree) for log_branch, tree in zip(*[samp_log_branch, samp_trees])])
        logp_prior = self.phylo_model.logprior(samp_log_branch)
        logp_joint = inverse_temp * logll + logp_prior
        lower_bound = torch.logsumexp(logll + logp_prior - logq_tree - logq_branch + self.log_p_tau - math.log(n_particles), 0)
        
        l_signal = logp_joint - logq_tree.detach() - logq_branch
        temp_lower_bound = torch.logsumexp(l_signal - math.log(n_particles), dim=0)
        snis_wts = torch.softmax(l_signal, dim=0)
        rws_fake_term = torch.sum(snis_wts.detach() * logq_tree, dim=0)

        return temp_lower_bound, rws_fake_term, lower_bound, torch.max(logll)

    def vimco_lower_bound(self, inverse_temp=1.0, n_particles=10):
        samp_trees, logq_tree = self.mp_sample_tree(n_particles)
        samp_log_branch, logq_branch = self.branch_model(samp_trees)

        logll = torch.stack([self.phylo_model.loglikelihood(log_branch, tree) for log_branch, tree in zip(*[samp_log_branch, samp_trees])])
        logp_prior = self.phylo_model.logprior(samp_log_branch)
        logp_joint = inverse_temp * logll + logp_prior
        lower_bound = torch.logsumexp(logll + logp_prior - logq_tree - logq_branch + self.log_p_tau - math.log(n_particles), 0)
        
        l_signal = logp_joint - logq_tree - logq_branch
        mean_exclude_signal = (torch.sum(l_signal) - l_signal) / (n_particles-1.)
        control_variates = torch.logsumexp(l_signal.view(-1,1).repeat(1, n_particles) - l_signal.diag() + mean_exclude_signal.diag() - math.log(n_particles), dim=0)
        temp_lower_bound = torch.logsumexp(l_signal - math.log(n_particles), dim=0)
        vimco_fake_term = torch.sum((temp_lower_bound - control_variates).detach() * logq_tree, dim=0)

        return temp_lower_bound, vimco_fake_term, lower_bound, torch.max(logll)

    def learn(self, stepsz, maxiter=200000, test_freq=1000, lb_test_freq=5000, anneal_freq_tree=20000, anneal_freq_branch=20000, anneal_freq_tree_warm=20000, anneal_freq_branch_warm=20000, anneal_rate_tree=0.75, anneal_rate_branch=0.75, n_particles=10,
              init_inverse_temp=0.001, warm_start_interval=100000, method='vimco',  save_to_path=None, logger=None):
        lbs, lls = [], []
        test_kl_div, test_lb = [], []
        if not isinstance(stepsz, dict):
            stepsz = {'tree': stepsz, 'branch': stepsz}
        optimizer_tree = torch.optim.Adam([
                    {'params': self.tree_model.parameters(), 'lr':stepsz['tree']},
                ])
        optimizer_branch = torch.optim.Adam([
                    {'params': self.branch_model.parameters(), 'lr':stepsz['branch']},
                ])
        run_time = -time.time()
        self.tree_model.train()
        for it in range(1, maxiter+1):
            inverse_temp = min(1., init_inverse_temp + it * 1.0/warm_start_interval)
            if method == 'vimco':
                temp_lower_bound, vimco_fake_term, lower_bound, logll  = self.vimco_lower_bound(inverse_temp, n_particles)
                loss = - temp_lower_bound - vimco_fake_term
            elif method == 'rws':
                temp_lower_bound, rws_fake_term, lower_bound, logll = self.rws_lower_bound(inverse_temp, n_particles)
                loss = - temp_lower_bound - rws_fake_term
            else:
                raise NotImplementedError

            lbs.append(lower_bound.item())
            lls.append(logll.item())
            
            optimizer_tree.zero_grad()
            optimizer_branch.zero_grad()
            loss.backward()
            optimizer_tree.step()
            optimizer_branch.step()

            gc.collect()
            if it % test_freq == 0:
                run_time += time.time()
                logger.info('{} Iter {}:({:.1f}s) Lower Bound: {:.4f} | Loglikelihood: {:.4f}'.format(time.asctime(time.localtime(time.time())), it, run_time, np.mean(lbs), np.max(lls)))
                if it % lb_test_freq == 0:
                    self.tree_model.eval()
                    run_time = -time.time()
                    test_lb.append(self.lower_bound(n_particles=1))
                    run_time += time.time()
                    if self.emp_tree_freq:
                        kldiv, pred_probs = self.kl_div()
                        test_kl_div.append(kldiv)
                        logger.info('>>> Iter {}:({:.1f}s) Test Lower Bound: {:.4f} Test KL: {:.4f}'.format(it, run_time, test_lb[-1], test_kl_div[-1]))
                    else:
                        logger.info('>>> Iter {}:({:.1f}s) Test Lower Bound: {:.4f}'.format(it, run_time, test_lb[-1]))
                    self.tree_model.train()
                    gc.collect()
                run_time = -time.time()
                lbs, lls = [], []
            if it > warm_start_interval:
                if (it-warm_start_interval) % anneal_freq_tree == 0:
                    for g in optimizer_tree.param_groups:
                        g['lr'] *= anneal_rate_tree
                if (it-warm_start_interval) % anneal_freq_branch == 0:
                    for g in optimizer_branch.param_groups:
                        g['lr'] *= anneal_rate_branch
            else:
                if it % anneal_freq_tree_warm == 0:
                    for g in optimizer_tree.param_groups:
                        g['lr'] *= anneal_rate_tree
                if it % anneal_freq_branch_warm == 0:
                    for g in optimizer_branch.param_groups:
                        g['lr'] *= anneal_rate_branch
        if save_to_path is not None:
            torch.save(self.state_dict(), save_to_path)
            
        return test_lb, test_kl_div
