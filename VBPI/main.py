import argparse
from copy import deepcopy
import os
import logging
import sys
import torch
import numpy as np
import pickle
import datetime
import time
sys.path.append("..")
from utils import loadData, summary, namenum
from datasets import get_empdataloader
from models import VBPI

def main(args):
    data_path = '../data/hohna_datasets_fasta/'
    ###### Load Data
    data, taxa = loadData(data_path + args.dataset + '.fasta', 'fasta')
    args.ntips = len(data)
    data, taxa = data, taxa

    if args.order_idx > -1:
        assert args.dataset == 'DS1'
        orders = np.load('../data/DS1ablation/100_random_orders.npy')
        data = [data[i] for i in orders[args.order_idx]]
        taxa = [taxa[i] for i in orders[args.order_idx]]
        name = "order" + str(args.order_idx) + '_' + args.gradMethod + '_hL_' + str(args.hLTree) + str(args.hLBranch) + '_norm_' + args.norm_type + '_edgeaggr_' + args.edge_aggr
    else:
        name = args.gradMethod + '_hL_' + str(args.hLTree) + str(args.hLBranch) + '_norm_' + args.norm_type + '_edgeaggr_' + args.edge_aggr
    if args.proj:
        name = name + '_proj'
    name = name + '_' + args.date
    args.folder = os.path.join(args.workdir, args.dataset, name)
    os.makedirs(args.folder, exist_ok=False)

    args.save_to_path = os.path.join(args.folder, 'final.pt')
    args.logpath = os.path.join(args.folder, 'final.log')

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    filehandler = logging.FileHandler(args.logpath)
    filehandler.setLevel(logging.INFO)
    logger.addHandler(filehandler)

    logger.info('Training with the following settings:')
    for name, value in vars(args).items():
        logger.info('{} : {}'.format(name, value))

    if args.empFreq:
        if args.order_idx > -1:
            ground_truth_path, samp_size = '../data/raw_data_DS1-8/', 750001
            logger.info('\nLoading empirical posterior estimates ......')
            run_time = -time.time()
            tree_dict_total, tree_names_total, tree_wts_total = summary(args.dataset, ground_truth_path, samp_size=samp_size)
            emp_tree_freq = {tree_dict_total[tree_name]:tree_wts_total[i] for i, tree_name in enumerate(tree_names_total)}
            for tree in emp_tree_freq.keys():
                namenum(tree, taxa)
            run_time += time.time()
        else:
            emp_tree_freq = get_empdataloader(args.dataset, batch_size=200, folder=args.folder)
        logger.info('Empirical estimates from MrBayes loaded')
    else:
        emp_tree_freq = None
    
    model = VBPI(taxa, data, pden=np.ones(4)/4., subModel=('JC', 1.0), emp_tree_freq=emp_tree_freq, hidden_dim_tree=args.hdimTree, hidden_dim_branch=args.hdimBranch, 
                 num_layers_tree=args.hLTree, num_layers_branch=args.hLBranch, gnn_type=args.gnn_type, aggr=args.aggr, edge_aggr=args.edge_aggr, project=args.proj, norm_type=args.norm_type)

    logger.info('Running on device: {}'.format(model.device))  
    logger.info('Parameter Info:')
    for param in model.parameters():
        logger.info(param.dtype)
        logger.info(param.size())

    logger.info('\nVBPI running, results will be saved to: {}\n'.format(args.save_to_path))
    test_lb, test_kl_div = model.learn({'tree':args.stepszTree,'branch':args.stepszBranch}, args.maxIter, test_freq=args.tf, 
    lb_test_freq=args.lbf, n_particles=args.nParticle, anneal_freq_tree=args.afTree, anneal_freq_branch=args.afBranch, anneal_freq_tree_warm=args.afTreewarm, anneal_freq_branch_warm=args.afBranchwarm, anneal_rate_tree=args.arTree, anneal_rate_branch=args.arBranch,init_inverse_temp=args.invT0, 
    warm_start_interval=args.nwarmStart, method=args.gradMethod, save_to_path=args.save_to_path, logger=logger)
                
    np.save(args.save_to_path.replace('.pt', '_test_lb.npy'), test_lb)
    if args.empFreq:
        np.save(args.save_to_path.replace('.pt', '_kl_div.npy'), test_kl_div)

def test(args):
    data_path = '../data/hohna_datasets_fasta/'
    ###### Load Data
    data, taxa = loadData(data_path + args.dataset + '.fasta', 'fasta')
    args.ntips = len(data)
    data, taxa = data[:args.ntips], taxa[:args.ntips]


    name = args.gradMethod + '_hL_' + str(args.hLTree) + str(args.hLBranch) + '_norm_' + args.norm_type + '_edgeaggr_' + args.edge_aggr
    if args.proj:
        name = name + '_proj'
    name = name + '_' + args.date
    args.folder = os.path.join(args.workdir, args.dataset, name)
    args.logpath = os.path.join(args.folder, 'final.log')

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    filehandler = logging.FileHandler(args.logpath)
    filehandler.setLevel(logging.INFO)
    logger.addHandler(filehandler)

    logger.info('Start evaluation:')
    empdataloader = get_empdataloader(args.dataset, batch_size=200, folder=args.folder)
    logger.info('Empirical estimates from MrBayes loaded')


    model = VBPI(taxa, data, pden=np.ones(4)/4., subModel=('JC', 1.0), emp_tree_freq=empdataloader, hidden_dim_tree=args.hdimTree, hidden_dim_branch=args.hdimBranch, 
                 num_layers_tree=args.hLTree, num_layers_branch=args.hLBranch, gnn_type=args.gnn_type, aggr=args.aggr, edge_aggr=args.edge_aggr, project=args.proj, norm_type=args.norm_type)
    with torch.no_grad():
        model.load_state_dict(torch.load(os.path.join(args.folder, 'final.pt')))
    model.tree_model.eval()
    model.branch_model.eval()

    if os.path.exists(os.path.join(args.folder, 'final_kl_div.npy')):
        kldiv = np.load(os.path.join(args.folder, 'final_kl_div.npy'))[-1]
    else:
        kldiv, pred_probs = model.kl_div(logger=logger)    
        np.save(os.path.join(args.folder, 'final_kl_div.npy'), np.array([kldiv]))
        np.save(os.path.join(args.folder, 'final_pred_probs.npy'), pred_probs)
    logger.info('KL: {:.6f}'.format(kldiv))

    if not args.sbn_support:
        lb1s, lb10s, lb1000s = [], [], []
        for i in range(1000):
            lb1, lb10, lb1000 = model.mp_lower_bound(n_runs=1000)
            lb1s.append(lb1)
            lb10s.append(lb10)
            lb1000s.append(lb1000)
            if (i+1) % 10 == 0:
                logger.info('{} {} rounds calculated'.format(time.asctime(time.localtime(time.time())), i+1))

        lb10s = np.array(lb10s).reshape(10,100)
        lb10s = np.mean(lb10s, axis=0)
        lb1s = np.array(lb1s).reshape(10,100)
        lb1smean10000 = np.mean(lb1s, axis=0)

        np.save(os.path.join(args.folder, 'lb1s.npy'), lb1s)
        np.save(os.path.join(args.folder, 'lb10s.npy'), lb10s)
        np.save(os.path.join(args.folder, 'lb1000s.npy'), lb1000s)
        logger.info('LB1: mean: {:.4f} std: {:.4f}'.format(np.mean(lb1s), np.std(lb1s)))
        logger.info('LB10: mean: {:.4f} std: {:.4f}'.format(np.mean(lb10s), np.std(lb10s)))
        logger.info('LB1000: mean: {:.4f} std: {:.4f}'.format(np.mean(lb1000s), np.std(lb1000s)))
    else:
        from vector_sbnModel import SBN
        from utils import summary_raw, get_support_from_mcmc
        ufboot_support_path = '../data/ufboot_data_DS1-11/'
        logger.info('Loading ufboot support')
        tree_dict_support, tree_names_support = summary_raw(args.dataset, ufboot_support_path)
        rootsplit_supp_dict, subsplit_supp_dict = get_support_from_mcmc(taxa, tree_dict_support, tree_names_support)
        logger.info('ufboot support loaded')
        sbn_model = SBN(taxa, rootsplit_supp_dict, subsplit_supp_dict)

        lb1s, lb10s, lb1000s = [], [], []
        n_outliers = 0
        for i in range(1000):
            lb1, lb10, lb1000, n_out = model.mp_lower_bound_sbnsupport(sbn_model, n_runs=1000)
            n_outliers += n_out
            lb1s.append(lb1)
            lb10s.append(lb10)
            lb1000s.append(lb1000)
            if (i+1) % 10 == 0:
                logger.info('{} {} rounds calculated | Num of Outliers {}'.format(time.asctime(time.localtime(time.time())), i+1, n_outliers))

        lb10s = np.array(lb10s).reshape(10,100)
        lb10s = np.mean(lb10s, axis=0) 
        lb1s = np.array(lb1s).reshape(10,100)
        lb1s = np.mean(lb1s, axis=0)

        np.save(os.path.join(args.folder, 'sbnsupp_lb1s.npy'), lb1s)
        np.save(os.path.join(args.folder, 'sbnsupp_lb10s.npy'), lb10s)
        np.save(os.path.join(args.folder, 'sbnsupp_lb1000s.npy'), lb1000s)
        logger.info('SBN support LB1: mean: {:.4f} std: {:.4f}'.format(np.mean(lb1s), np.std(lb1s)))
        logger.info('SBN support LB10: mean: {:.4f} std: {:.4f}'.format(np.mean(lb10s), np.std(lb10s)))
        logger.info('SBN support LB1000: mean: {:.4f} std: {:.4f}'.format(np.mean(lb1000s), np.std(lb1000s)))

    return kldiv, lb1s, lb1smean10000, lb10s, lb1000s

def parse_args():
    parser = argparse.ArgumentParser()

    ######### Data arguments
    parser.add_argument('--dataset', default='DS1', help=' DS1 | DS2 | DS3 | DS4 | DS5 | DS6 | DS7 | DS8 ')
    parser.add_argument('--empFreq', default=False, action='store_true', help='emprical frequence for KL computation')
    ######### Model arguments
    parser.add_argument('--nf', type=int, default=2, help=' branch length feature embedding dimension ')
    parser.add_argument('--hdimTree', type=int, default=100, help='hidden dimension for node embedding net')
    parser.add_argument('--hdimBranch', type=int, default=100, help='hidden dimension for node embedding net')
    parser.add_argument('--hLTree', type=int, default=2, help='number of hidden layers for node embedding net of tree model')
    parser.add_argument('--hLBranch',  type=int, default=2, help='number of hidden layers for node embedding net of branch model')
    parser.add_argument('--gnn_type', type=str, default='edge', help='gcn | sage | gin | ggnn')
    parser.add_argument('--norm_type', type=str, default='layer', help='normalization type for tree model. ')
    parser.add_argument('--aggr', type=str, default='sum', help='sum | mean | max')
    parser.add_argument('--edge_aggr', type=str, default='max', help='sum | mean | max for EdgePooling')
    parser.add_argument('--proj', default=False, action='store_true', help='use projection first in SAGEConv')

    ######### Optimizer arguments
    parser.add_argument('--stepszTree', type=float, default=0.0001, help=' step size for tree topology parameters ')
    parser.add_argument('--stepszBranch', type=float, default=0.001, help=' stepsz for branch length parameters ')
    parser.add_argument('--maxIter', type=int, default=400000, help=' number of iterations for training, default=400000')
    parser.add_argument('--invT0', type=float, default=0.001, help=' initial inverse temperature for annealing schedule, default=0.001')
    parser.add_argument('--nwarmStart', type=float, default=100000, help=' number of warm start iterations, default=100000')
    parser.add_argument('--nParticle', type=int, default=10, help='number of particles for variational objectives, default=10')
    parser.add_argument('--arTree', type=float, default=0.75, help='step size anneal rate for tree model, default=0.75')
    parser.add_argument('--arBranch', type=float, default=0.75, help='step size anneal rate for branch model, default=0.75')
    parser.add_argument('--afTreewarm', type=int, default=20000, help='step size anneal frequency for tree model during annealing, default=20000')
    parser.add_argument('--afBranchwarm', type=int, default=20000, help='step size anneal frequency for branch model during annealing, default=20000')
    parser.add_argument('--afTree', type=int, default=20000, help='step size anneal frequency for tree model, default=20000')
    parser.add_argument('--afBranch', type=int, default=20000, help='step size anneal frequency for branch model, default=20000')
    parser.add_argument('--tf', type=int, default=1000, help='monitor frequency during training, default=1000')
    parser.add_argument('--lbf', type=int, default=5000, help='lower bound test frequency, default=5000')
    parser.add_argument('--gradMethod', type=str, default='vimco', help=' vimco | rws ')

    parser.add_argument('--workdir', default='results', type=str)
    parser.add_argument('--date', default='2023-03-16', type=str)
    parser.add_argument('--eval', default=False, action='store_true')
    parser.add_argument('--sbn_support', default=False, action='store_true')

    parser.add_argument('--order_idx', default=-1, type=int, help='-1 refers to the lexicographical order')
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()
    if not args.eval:
        main(args)
    else:
        test(args)
    sys.exit()