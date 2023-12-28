from math import floor
from collections import OrderedDict
from random import shuffle
import argparse

import glob

import numpy as np
import pandas as pd


def randomize_files(file_list):
    shuffle(file_list)

def get_training_and_testing_sets(file_list, split):
    split_index = floor(len(file_list) * split)
    train_files = file_list[:split_index]
    test_files = file_list[split_index:]
    return train_files, test_files

def Split_Dataset(dataset,task):
    if task == 'staging':
        label = pd.read_csv('/data1/WSI/Patches/'+dataset+'/BioData/'+dataset+'_Stage_Label.csv')
        # ignore slides w/o stage labels
        label = label.dropna()
        graph_indices = np.arange(len(label))


        randomize_files(graph_indices)

        train_indices, testval_indices = get_training_and_testing_sets(graph_indices, 0.8)

        train_list = list(label.iloc[train_indices,0])
        train_label = np.array(label.iloc[train_indices,1],dtype=int)
        testval_list = list(label.iloc[testval_indices,0])
        testval_label = np.array(label.iloc[testval_indices,1],dtype=int)

        with open('/home/r20user8/Documents/HDPMIL/datasets/'+dataset+'/'+task+'_'+dataset+"_train.txt", "w") as f:
            f.write('\n'.join(train_list))

        with open('/home/r20user8/Documents/HDPMIL/datasets/'+dataset+'/'+task+'_'+dataset+"_testval.txt", "w") as f:
            f.write('\n'.join(testval_list))

        np.save('/home/r20user8/Documents/HDPMIL/datasets/'+dataset+'/'+task+'_'+dataset+'_train_label.npy',train_label)
        np.save('/home/r20user8/Documents/HDPMIL/datasets/'+dataset+'/'+task+'_'+dataset+'_testval_label.npy', testval_label)

    elif task == 'binary' and dataset in ['BRCA','COAD']:
        normal_list = glob.glob(f'/data1/WSI/Patches/{dataset}_Solid_Tissue_Normal_Kimia_20x/*')
        cancer_list = glob.glob(f'/data1/WSI/Patches/{dataset}_Kimia_20x/*')
        randomize_files(normal_list)
        randomize_files(cancer_list)

        train_cancer_list, testval_cancer_list = get_training_and_testing_sets(cancer_list, 0.8)
        # test_list, val_list = get_training_and_testing_sets(testval_list, 0.5)
        train_normal_list, testval_normal_list = get_training_and_testing_sets(normal_list, 0.8)
        # test_normal_list, val_normal_list = get_training_and_testing_sets(testval_normal_list, 0.5)
        train_list = train_cancer_list + train_normal_list
        testval_list = testval_cancer_list + testval_normal_list
        with open('/home/r20user8/Documents/HDPMIL/datasets/'+dataset+'/'+task+'_'+dataset+"_train.txt", "w") as f:
            f.write('\n'.join(train_list))
        with open('/home/r20user8/Documents/HDPMIL/datasets/'+dataset+'/'+task+'_'+dataset+"_testval.txt", "w") as f:
            f.write('\n'.join(testval_list))
        # val_list = val_list + val_normal_list
        train_label = np.array([1]*len(train_cancer_list)+[0]*len(train_normal_list))
        testval_label = np.array([1] * len(testval_cancer_list) + [0] * len(testval_normal_list))

        np.save('/home/r20user8/Documents/HDPMIL/datasets/'+dataset+'/'+task+'_'+dataset+'_train_label.npy',train_label)
        np.save('/home/r20user8/Documents/HDPMIL/datasets/'+dataset+'/'+task+'_'+dataset+'_testval_label.npy', testval_label)
    else:
        print('Do not support such task or dataset!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='base dictionary construction')
    parser.add_argument('--dataset', type=str, default='COAD')
    parser.add_argument('--task',default='binary')

    args = parser.parse_args()
    Split_Dataset(args.dataset,args.task)
