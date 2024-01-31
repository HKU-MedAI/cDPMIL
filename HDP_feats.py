import argparse
import glob
import os

import copy
import numpy as np
import torch
from tqdm import tqdm

from model.dpmil import HDP_Cluster_EM, DP_Cluster_VI, DP_Classifier, BClassifier
from collections import Counter
import random
import math
from sklearn.mixture import BayesianGaussianMixture
from sklearn.cluster import KMeans
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ['CUDA_VISIBLE_DEVICES']='3'


def get_feats(train_list, eta_cluster, feat_dim, dataset):
    for i,feat_pth in tqdm(zip(range(len(train_list)),train_list)):
        if dataset == 'Camelyon':
            # bag_feats = np.load(feat_pth)
            # feat_pth = feat_pth[20:-4]
            feat_pth = feat_pth.split('/')[-1].split('.')[0]
            if 'test' not in feat_pth:
                bag_feats = torch.load(f'/data1/WSI/Patches/Features/Camelyon16/simclr_files_256_v2/training/{feat_pth}/faetures.pt')
            else:
                bag_feats = torch.load(f'/data1/WSI/Patches/Features/Camelyon16/simclr_files_256_v2/testing/{feat_pth}/features.pt')
            bag_feats = bag_feats.cpu().numpy()
        else:
            bag_feats = torch.load(feat_pth+'/features.pt')
            bag_feats = bag_feats.cpu().numpy()
            feat_pth = feat_pth[-23:]
            # bag_feats = bag_feats.to(device)
        # dp_cluster = HDP_Cluster_EM(n_dps=10, trunc=10, eta=eta_cluster, batch_size=1, epoch=20, dim=feat_dim).to(
        #     device)
        # logits = dp_cluster(bag_feats)
        # assignments = torch.argmax(logits, dim=1)
        # # num_cluster = len(torch.unique(assignments))
        # centroids = [torch.mean(bag_feats[assignments == i], dim=0) for i in torch.unique(assignments)]
        # centroids = torch.stack(centroids) # [num_cluster, dim]
        for n_comp in [10]:
            dp_cluster = BayesianGaussianMixture(n_components=n_comp, random_state=0, max_iter=30)
            dp_cluster.fit(bag_feats)
            assignments = dp_cluster.predict(bag_feats)
            centroids = np.array([np.mean(bag_feats[assignments == i], axis=0) for i in np.unique(assignments)])
            if 'test' not in feat_pth:
                coor_pth = f'/data1/WSI/Patches/Features/Camelyon16/simclr_files_256_v2/traning/{feat_pth}/c_idx.txt'
            else:
                coor_pth = f'/data1/WSI/Patches/Features/Camelyon16/simclr_files_256_v2/testing/{feat_pth}/c_idx.txt'
            with open(coor_pth) as f:
                coor = f.readlines()
            X = []
            Y = []
            for item in coor:
                X.append(int(item.split('\t')[0]) * 256)
                Y.append(int(item.split('\t')[1]) * 256)
            pred = dp_cluster.score_samples(bag_feats)
            coor_prob_info = {'X': X, 'Y': Y, 'logit': pred}
            coor_prob_info = pd.DataFrame(coor_prob_info)
            coor_prob_info.to_csv(
                f'/data1/WSI/Patches/Features/Camelyon16/simclr_files_256_v2/testing/{feat_pth}/coor_logit_DP.csv')
            # centroids = torch.from_numpy(centroids).to(device)
            # centroids = np.array([np.mean(bag_feats[assignments == i], axis=0) for i in range(args.num_prototypes)])
            # abort invalid features
            # if i==0:
            #     os.mkdir(f'/home/r20user8/Documents/HDPMIL/datasets/{dataset}/DP_EM_feats/Concentration_{concentration}')
            np.save(f'/home/r20user8/Documents/HDPMIL/datasets/{dataset}/DP_EM_feats_DSMIL_v2/{feat_pth}.npy', centroids)
    return

def get_feats_revised(train_list, eta_cluster, feat_dim, dataset, method):
    for i,feat_pth in tqdm(zip(range(len(train_list)),train_list)):
        if dataset == 'Camelyon':
            feat_pth = feat_pth.split('\n')[0]
            # bag_feats = torch.load(feat_pth+'/features.pt')
            # bag_feats = bag_feats.cpu().numpy()
            bag_feats = np.load(feat_pth)
            # bag_feats = torch.tensor(bag_feats).to(device)
            # slide_name = feat_pth.split('/')[-1]
            slide_name = feat_pth.split('/')[-1].split('.')[0]
        else:
            bag_feats = torch.load(feat_pth+'/features.pt')
            bag_feats = bag_feats.cpu().numpy()
            slide_name = feat_pth[-23:]
            # bag_feats = bag_feats.to(device)
        # dp_cluster = HDP_Cluster_EM(n_dps=10, trunc=10, eta=eta_cluster, batch_size=1, epoch=20, dim=feat_dim).to(
        #     device)
        # logits = dp_cluster(bag_feats)
        # assignments = torch.argmax(logits, dim=1)
        # # num_cluster = len(torch.unique(assignments))
        # centroids = [torch.mean(bag_feats[assignments == i], dim=0) for i in torch.unique(assignments)]
        # centroids = torch.stack(centroids) # [num_cluster, dim]
        if method == 'DP':
            # for concentration in [0.1]:
            dp_cluster = BayesianGaussianMixture(n_components=10, random_state=0, max_iter=30)
            dp_cluster.fit(bag_feats)
            assignments = dp_cluster.predict(bag_feats)
            weights = np.exp(dp_cluster._estimate_log_weights())
            centroids = [np.mean(bag_feats[assignments == i], axis=0) for i in range(10)]
            for i in range(10):
                centroids[i] = centroids[i]*1
        # elif method == 'mean':
        #     slide_feats = np.mean(bag_feats,axis=0)
        # elif method == 'max':
        #     slide_feats = np.max(bag_feats,axis=0)
        # elif method == 'kmeans':
        #     kmeans = KMeans(n_clusters=10, random_state=0).fit(bag_feats)
        #     assignments = kmeans.labels_
        #     centroids = np.array([np.mean(bag_feats[assignments == i], axis=0) for i in range(10)])
        #     slide_feats = np.mean(centroids,axis=0)
        # else:
        #     print('Method Not Supported!')
        np.save(f'/home/r20user8/Documents/HDPMIL/datasets/{dataset}/{method}_weighted/{slide_name}.npy', centroids)
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
    parser.add_argument('--split',default=0,help='which split of dataset you want to deal with')
    parser.add_argument('--method',default='DP',help='pooling methods: DP, max, mean, kmeans')
    args = parser.parse_args()
    if args.dataset == 'Camelyon' and args.task == 'binary':
        # train_list = f'/home/r20user8/Documents/HDPMIL/datasets/{args.dataset}/binary_Camelyon_train.txt'
        train_list = f'/home/r20user8/Documents/HDPMIL/datasets/Camelyon16/remix_processed/train_list.txt'
        train_list = open(train_list, 'r').readlines()
        train_list = [x.split(',')[0] for x in train_list]  # file names
        # test_list = f'/home/r20user8/Documents/HDPMIL/datasets/{args.dataset}/binary_Camelyon_testval.txt'
        test_list = f'/home/r20user8/Documents/HDPMIL/datasets/Camelyon16/remix_processed/test_list.txt'
        test_list = open(test_list,'r').readlines()
        test_list = [x.split(',')[0] for x in test_list]
        all_list = test_list + train_list
        exist_list = glob.glob('/home/r20user8/Documents/HDPMIL/datasets/Camelyon/DP_EM_feats_DSMIL_v2/*')
        # pending_list = []
        # for item in all_list:
        #     flag = 1
        #     for i in exist_list:
        #         if item.split('/')[-1][:-1] == i.split('/')[-1].split('.')[0]:
        #             flag = 0
        #             break
        #     if flag:
        #         pending_list.append(item)
        pending_list = train_list+test_list
    else:
        # all_list = np.load(f'pending_list_{args.split}.npy')
        # all_list = all_list.tolist()
        all_list = glob.glob(f'/data1/WSI/Patches/Features/{args.dataset}/{args.dataset}_Tissue_Kimia_20x/*')
        # exist_list = glob.glob('/home/r20user8/Documents/HDPMIL/datasets/'+args.dataset+'/DP_EM_feats/*')
        # pending_list = []
        # for item in all_list:
        #     flag = 1
        #     for i in exist_list:
        #         if item[-23:] in i:
        #             flag = 0
        #             break
        #     if flag:
        #         pending_list.append(item)
        pending_list = all_list


    print(f'Need to perform DP feat extraction on {len(pending_list)} slides.')
    eta_cluster = 7
    shuffled_idxs = np.random.permutation(len(pending_list))
    pending_list = [pending_list[index] for index in shuffled_idxs]
    get_feats_revised(pending_list,  eta_cluster, args.feat_dim, args.dataset,args.method)
