import argparse
import os

import copy
import numpy as np
import torch
from tqdm import tqdm

from model.dpmil import HDP_Cluster_EM, DP_Cluster_VI, DP_Classifier, BClassifier
from collections import Counter
import random
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ['CUDA_VISIBLE_DEVICES']='3'


def get_feats(train_list, eta_cluster, feat_dim, dataset):
    total_loss = 0
    for i,feat_pth in tqdm(zip(range(len(train_list)),train_list)):
        if dataset == 'Camelyon':
            bag_feats = np.load(feat_pth)
            bag_feats = torch.tensor(bag_feats).to(device)
            feat_pth = feat_pth[20:-4]
        else:
            bag_feats = torch.load(feat_pth+'/features.pt')
            bag_feats = bag_feats.to(device)
        dp_cluster = HDP_Cluster_EM(n_dps=10, trunc=10, eta=eta_cluster, batch_size=1, epoch=20, dim=feat_dim).to(
            device)
        logits = dp_cluster(bag_feats)
        assignments = torch.argmax(logits, dim=1)
        # num_cluster = len(torch.unique(assignments))
        centroids = [torch.mean(bag_feats[assignments == i], dim=0) for i in torch.unique(assignments)]
        centroids = torch.stack(centroids) # [num_cluster, dim]
        # centroids = np.array([np.mean(bag_feats[assignments == i], axis=0) for i in range(args.num_prototypes)])
        # abort invalid features
        torch.save(centroids,'/home/r20user8/Documents/HDPMIL/datasets/'+dataset+'/HDP_feats/'+feat_pth+'.pt')
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='base dictionary construction')
    parser.add_argument('--dataset', type=str, default='Camelyon')
    parser.add_argument('--num_prototypes', type=int, default=8)
    parser.add_argument('--num_shift_vectors', type=int, default=200)
    parser.add_argument('--lr', default=0.0002, type=float, help='Initial learning rate [0.0002]')
    parser.add_argument('--weight_decay', default=5e-3, type=float, help='Weight decay [5e-3]')
    parser.add_argument('--num_epochs', default=50, type=int, help='Number of total training epochs')
    parser.add_argument('--num_classes', default=2, type=int, help='Number of total classes in classification task')
    parser.add_argument('--feat_dim', default=512, type=int, help='feature dimension')
    parser.add_argument('--task', default='binary', help='binary cancer/normal classification or staging')
    args = parser.parse_args()
    if args.dataset == 'Camelyon' and args.task == 'binary':
        train_list = f'datasets/{args.dataset}16/remix_processed/train_list.txt'
        train_list = open(train_list, 'r').readlines()
        train_list = [x.split(',')[0] for x in train_list]  # file names
        test_list = f'datasets/{args.dataset}16/remix_processed/test_list.txt'
        test_list = open(test_list,'r').readlines()
        test_list = [x.split(',')[0] for x in test_list]
    else:
        train_list = f'datasets/{args.dataset}/{args.task}_{args.dataset}_train.txt'
        train_list = open(train_list, 'r').readlines()
        train_list = [x.split('\n')[0] for x in train_list]
        test_list = f'datasets/{args.dataset}/{args.task}_{args.dataset}_testval.txt'
        test_list = open(test_list, 'r').readlines()
        test_list = [x.split('\n')[0] for x in test_list]

    eta_cluster = 7
    shuffled_test_idxs = np.random.permutation(len(test_list))
    test_list = [test_list[index] for index in shuffled_test_idxs]
    get_feats(test_list,  eta_cluster, args.feat_dim, args.dataset)

