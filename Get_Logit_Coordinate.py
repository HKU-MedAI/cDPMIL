import argparse
import copy
import logging
import sys
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.spatial.distance import cdist
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from torch.autograd import Variable

from model import abmil, dsmil
from tools.utils import setup_logger
from train_remix import get_bag_feats_v1

i_classifier = dsmil.FCLayer(in_size=512, out_size=1).cuda()
b_classifier = dsmil.BClassifier(input_size=512, output_class=1, dropout_v=0).cuda()
milnet = dsmil.MILNet(i_classifier, b_classifier).cuda()
state_dict_weights = torch.load('DSMIL_v2.pth')
milnet.load_state_dict(state_dict_weights)
print('loading from DSMIL_v2.pth')

test_feats = open(f'datasets/Camelyon16/remix_processed/test_list.txt', 'r').readlines()
test_feats = np.array(test_feats)

with torch.no_grad():
    for i in range(len(test_feats)):
        _, bag_feats = get_bag_feats_v1(test_feats[i], 1)
        ins_prediction, bag_prediction, _, _ = milnet(bag_feats)

        pred = (ins_prediction.view(-1)).cpu().numpy()
        slide_name = test_feats[i].split(',')[0].split('/')[-1].split('.')[0]
        if 'test' not in slide_name:
            coor_pth = f'/data1/WSI/Patches/Features/Camelyon16/simclr_files_256_v2/traning/{slide_name}/c_idx.txt'
        else:
            coor_pth = f'/data1/WSI/Patches/Features/Camelyon16/simclr_files_256_v2/testing/{slide_name}/c_idx.txt'
        with open(coor_pth) as f:
            coor = f.readlines()
        X = []
        Y = []
        for item in coor:
            X.append(int(item.split('\t')[0])*256)
            Y.append(int(item.split('\t')[1])*256)
        coor_prob_info = {'X':X,'Y':Y,'logit':pred}
        coor_prob_info = pd.DataFrame(coor_prob_info)
        coor_prob_info.to_csv(f'/data1/WSI/Patches/Features/Camelyon16/simclr_files_256_v2/testing/{slide_name}/coor_logit.csv')

