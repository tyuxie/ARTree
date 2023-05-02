import argparse
import os
import sys
import numpy as np
import datetime
import logging
sys.path.append("..")
from datasets import get_dataloader, get_empdataloader
from models import TDE

def main(args):
    name = 'nParticle_' + str(args.batch_size) + '_norm_' + args.norm_type + '_edgeaggr_' + args.edge_aggr + '_' + str(datetime.datetime.now()).replace(' ','_')
    args.save_folder = 'results/{}/repo{}/{}'.format(args.dataset, args.repo, name)
    os.makedirs(args.save_folder, exist_ok=False)
    args.save_to_path = args.save_folder + '/final.pt'
    args.logpath = args.save_folder + '/final.log'

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    filehandler = logging.FileHandler(args.logpath)
    filehandler.setLevel(logging.INFO)
    logger.addHandler(filehandler)

    logger.info('Training with the following settings:')
    for name, value in vars(args).items():
        logger.info('{} : {}'.format(name, value))

    dataloader = get_dataloader(args.dataset, args.repo, args.batch_size, args.maxIter, folder=args.save_folder)
    wts = np.load(os.path.join('..','embed_data', args.dataset, 'repo{}'.format(args.repo), 'wts.npy'))
    taxa = np.load(os.path.join('..','embed_data', args.dataset, 'repo{}'.format(args.repo), 'taxa.npy'))
    wts = np.array(wts) / np.sum(wts)
        
    if args.empFreq:
        empdataloader = get_empdataloader(args.dataset, batch_size=10, folder=args.save_folder)
        logger.info('Empirical estimates from MrBayes loaded')
    else:
        empdataloader = None

    model = TDE(dataloader, ntips=len(taxa), hidden_dim=args.hdim, num_layers=args.hL, gnn_type=args.gnn_type, aggr=args.aggr, edge_aggr=args.edge_aggr, project=args.project, norm_type=args.norm_type, empdataloader=empdataloader)
    logger.info('Optimal NLL: {}'.format(-np.sum(wts * np.log(wts))))
    logger.info('Parameter Info:')
    for param in model.parameters():
        logger.info(param.dtype)
        logger.info(param.size())
    

    logger.info('\n TDE Model training, results will be saved to: {}\n'.format(args.save_to_path))
    nlls = model.train(args.stepsz, maxiter=args.maxIter, test_freq=args.tf,
                                    anneal_freq=args.af, anneal_rate=args.ar, save_to_path=args.save_to_path, 
                                    logger=logger, klf=args.klf)

    if args.empFreq:
        np.save(args.logpath.replace('final.log', 'nll.npy'), nlls[0])
        np.save(args.logpath.replace('final.log', 'kldivs.npy'), nlls[1])
        np.save(args.logpath.replace('final.log', 'ells.npy'), nlls[2])
    else:
        np.save(args.logpath.replace('final.log', 'nll.npy'), nlls)

def parse_args():
    parser = argparse.ArgumentParser()

    ######### Model arguments
    parser.add_argument('--dataset', type=str, default='DS1', help='DS1 | DS2 | ...')
    parser.add_argument('--norm_type', default='layer', type=str)
    parser.add_argument('--repo', type=int, default=1, help='simulation | DS1 | DS2 | ...')
    parser.add_argument('--hdim', type=int, default=100, help='hidden dimension for node embedding net')
    parser.add_argument('--hL', type=int, default=2, help='number of hidden layers for node embedding net')
    parser.add_argument('--gnn_type', type=str, default='edge', help='gcn | sage | gin | ggnn')
    parser.add_argument('--project', default=False, action='store_true')
    parser.add_argument('--aggr', type=str, default='sum', help='sum | mean | max')
    parser.add_argument('--edge_aggr', type=str, default='max', help='sum | mean | max for EdgePooling')
    parser.add_argument('--empFreq', default=False, action='store_true')
    ######### Optimizer arguments
    parser.add_argument('--stepsz', type=float, default=0.0001, help=' stepsz parameters ')
    parser.add_argument('--maxIter', type=int, default=200000, help=' number of iterations for training, default=200000')
    parser.add_argument('--batch_size', type=int, default=10, help='batch_size for gradient based optimization, default=10')
    parser.add_argument('--ar', type=float, default=0.75, help='step size anneal rate, default=0.75')
    parser.add_argument('--af', type=int, default=20000, help='step size anneal frequency, default=20000')
    parser.add_argument('--tf', type=int, default=1000, help='monitor frequency during training, default=1000')
    parser.add_argument('--klf', type=int, default=5000, help='KL monitor frequency during training, default=5000')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    main(args)