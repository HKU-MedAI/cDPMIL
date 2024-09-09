import argparse
import glob
import os

import numpy as np
import torch
from tqdm import tqdm

from sklearn.mixture import BayesianGaussianMixture
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_feats(train_list, eta_cluster, args):
    for i,feat_pth in tqdm(zip(range(len(train_list)),train_list)):
        bag_feats = torch.load(feat_pth+'/features.pt')
        bag_feats = bag_feats.cpu().numpy()
        slide_name = feat_pth.split('/')[-1]
        dp_cluster = BayesianGaussianMixture(n_components=eta_cluster, random_state=0, max_iter=30, weight_concentration_prior=args.concentration)
        dp_cluster.fit(bag_feats)
        assignments = dp_cluster.predict(bag_feats)
        centroids = np.array([np.mean(bag_feats[assignments == i], axis=0) for i in np.unique(assignments)])
        os.makedirs(f'{args.save_dir}/{args.dataset}/DP_EM_feats_concentration{args.concentration}',exist_ok=True)
        np.save(f'{args.save_dir}/{args.dataset}/DP_EM_feats_concentration{args.concentration}/{slide_name}.npy', centroids)
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='base dictionary construction')
    parser.add_argument('--dataset', type=str, default='LUAD')
    parser.add_argument('--feat_dir', type=str)
    parser.add_argument('--num_clusters', type=int, default=10)
    parser.add_argument('--concentration', type=float, default=0.1)
    parser.add_argument('--save_dir', type=str, default='./')
    args = parser.parse_args()
    # features directory of all slides including train/test/val
    all_list = glob.glob(f'{args.feat_dir}/*')
    # pending list is the list of slides that need to be processed
    pending_list = all_list

    print(f'Need to perform DP feat extraction on {len(pending_list)} slides.')
    eta_cluster = args.num_clusters
    shuffled_idxs = np.random.permutation(len(pending_list))
    pending_list = [pending_list[index] for index in shuffled_idxs]
    start_time = time.time()
    get_feats(pending_list,  eta_cluster,  args)
    end_time = time.time()
    duration = end_time - start_time
    print(f'Time elapsed: {duration // 3600} hours {(duration % 3600) // 60} mins {duration % 60}  seconds.')
