import argparse
import os

import copy
import numpy as np
import torch
from tqdm import tqdm

from tools.clustering import Kmeans
from model.dpmil import DirichletProcess,HDP_binary_classifier
from collections import Counter
import random
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             roc_auc_score, roc_curve)

os.environ['CUDA_VISIBLE_DEVICES']='0'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def setup_seed(seed):
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True

def train(train_list, train_labels, model, criterion, optimizer):
    model.train()
    total_loss = 0
    for i,feat_pth,label in tqdm(zip(range(len(train_list)),train_list,train_labels)):
        optimizer.zero_grad()
        bag_feats = np.load(feat_pth)
        bag_feats = torch.tensor(bag_feats).to(device)
        # abort invalid features
        if torch.isnan(bag_feats).sum() > 0:
            continue
        bag_prediction, neg_likelyhood = model(bag_feats)
        MC_num = bag_prediction.shape[0]
        bag_label = torch.tensor([label]*MC_num).to(device)
        bag_loss = criterion(bag_prediction,bag_label)
        loss = bag_loss + neg_likelyhood

        loss.backward()
        optimizer.step()
        total_loss = total_loss + loss.item()
        print('\r Training bag [%d/%d] bag loss: %.4f' % (i, len(train_list), loss.item()))
    return total_loss/len(train_list)

def test(test_list, test_labels, model, criterion):
    model.eval()
    total_loss = 0
    test_predictions = []
    with torch.no_grad():
        for i,(feat_pth,label) in enumerate(zip(test_list,test_labels)):
            bag_feats = np.load(feat_pth)
            bag_feats = torch.tensor(bag_feats).to(device)
            bag_prediction, neg_likelyhood = model(bag_feats)
            bag_prediction = torch.mean(bag_prediction, axis=0).unsqueeze(0)
            bag_label = torch.tensor([label]).to(device)
            loss = criterion(bag_prediction,bag_label)
            total_loss = total_loss + loss.item()
            print('\r Testing bag [%d/%d] bag loss: %.4f' % (i, len(test_list), loss.item()))
            prob = torch.nn.Softmax(dim=1)(bag_prediction)
            test_predictions.append(prob.squeeze().cpu().numpy())
    test_predictions = np.array(test_predictions)
    test_predictions = test_predictions[:,1]
    _, _, thresholds_optimal = multi_label_roc(test_labels, test_predictions, 1)
    class_prediction_bag = copy.deepcopy(test_predictions)
    class_prediction_bag[test_predictions >= thresholds_optimal[0]] = 1
    class_prediction_bag[test_predictions < thresholds_optimal[0]] = 0
    test_predictions = class_prediction_bag
    test_labels = np.squeeze(test_labels)
    y_pred, y_true = inverse_convert_label(test_predictions), inverse_convert_label(test_labels)
    p = precision_score(y_true, y_pred, average='macro')
    r = recall_score(y_true, y_pred, average='macro')
    acc = accuracy_score(y_true, y_pred)
    avg = np.mean([p, r, acc])
    c_auc = roc_auc_score(y_pred, y_true)
    return p, r, acc, avg, c_auc

def inverse_convert_label(labels):
    # one-hot decoding
    if len(np.shape(labels)) == 1:
        return labels
    else:
        converted_labels = np.zeros(len(labels))
        for ix in range(len(labels)):
            converted_labels[ix] = np.argmax(labels[ix])
        return converted_labels

def multi_label_roc(labels, predictions, num_classes):
    thresholds, thresholds_optimal, aucs = [], [], []
    if len(predictions.shape) == 1:
        predictions = predictions[:, None]
    if len(labels.shape) == 1:
        labels = labels[:, None]
    for c in range(0, num_classes):
        label = labels[:, c]
        prediction = predictions[:, c]
        fpr, tpr, threshold = roc_curve(label, prediction)
        fpr_optimal, tpr_optimal, threshold_optimal = optimal_thresh(fpr, tpr, threshold)
        c_auc = roc_auc_score(label, prediction)
        aucs.append(c_auc)
        thresholds.append(threshold)
        thresholds_optimal.append(threshold_optimal)
    return aucs, thresholds, thresholds_optimal

def optimal_thresh(fpr, tpr, thresholds, p=0):
    loss = (fpr - tpr) - p * tpr / (fpr + tpr + 1)
    idx = np.argmin(loss, axis=0)
    return fpr[idx], tpr[idx], thresholds[idx]

def reduce(args, train_list, train_label):
    setup_seed(1)
    prototypes = []
    semantic_shifts = []
    real_train_list = []
    for feat_pth,label in tqdm(zip(train_list,train_label)):
        # try:
        feats = np.load(feat_pth)
        # feats = np.ascontiguousarray(feats, dtype=np.float32)
        feats = (feats-np.mean(feats, axis=0))/(np.std(feats, axis=0))
        # # feats = (feats - np.min(feats, axis=1, keepdims=True))
        # # (np.max(feats, axis=1, keepdims=True) - np.min(feats, axis=1, keepdims=True))
        #
        feats = torch.tensor(feats).to(device)
        D_P_cluster = DirichletProcess(concentration=0.1,trunc=args.num_prototypes,eta=1,batch_size=1,dim=512).to(device)
        neg_likelyhood = -D_P_cluster.likelihood(feats)


        # dp_cluster = DP_Cluster(concentration=10,trunc=args.num_prototypes,eta=1,batch_size=1,epoch=20, dim=512).to(device)
        # logits = dp_cluster(feats)
        num_MC = 100
        betas = D_P_cluster.sample_beta(num_MC)
        weights = D_P_cluster.mix_weights(betas)[:, :-1]
        normalize_factor = torch.transpose(torch.vstack((torch.sum(weights, axis=1), torch.sum(weights, axis=1))), 0, 1)
        weights = weights/normalize_factor
        MC_samples = []
        for j in range(num_MC):
            prob = weights[j,:].cpu().numpy()
            draw = np.random.choice(args.num_prototypes,size=1,p=prob)[0]
            x = D_P_cluster.gaussians[draw].sample(1)[0,:]
            MC_samples.append(x)
        MC_samples = torch.stack(MC_samples)
        D_P_clssfy = DirichletProcess(concentration=0.1,trunc=args.num_prototypes,eta=1,batch_size=1,dim=512).to(device)
        logits = D_P_clssfy.infer(MC_samples)
        bag_label = torch.tensor([label] * num_MC)


        # assignments = torch.argmax(logits, dim=1)
        # logits = logits.cpu().numpy()
        feats = np.load(feat_pth)
        # feats = (feats-np.mean(feats,axis=0))/(np.std(feats,axis=0))
        # # feats = (feats - np.min(feats, axis=1, keepdims=True))
        # # (np.max(feats, axis=1, keepdims=True) - np.min(feats, axis=1, keepdims=True))
        # kmeans = Kmeans(k=args.num_prototypes, pca_dim=-1)
        # kmeans.cluster(feats, seed=66)  # for reproducibility
        # assignments = kmeans.labels.astype(np.int64)
        # # a = Counter(assignments)
        # # compute the centroids for each cluster
        # # centroids = torch.zeros(args.num_prototypes,feats.shape[1])
        # # for i in range(args.num_prototypes):
        # #     centroids[i] = torch.mean(feats[assignments == i], axis=0)
        centroids = np.array([np.mean(feats[assignments.cpu() == i], axis=0)
                            for i in range(args.num_prototypes)])
        # centroids = np.array([np.mean(feats[assignmentsyi == i], axis=0)
        #                     for i in range(args.num_prototypes)])
        # centroids = torch.tensor([torch.mean(feats[assignments == i], axis=0)
        #                     for i in range(args.num_prototypes)])

        # compute covariance matrix for each cluster
        # a = []
        # for i in range(args.num_prototypes):
        #     # a.append(feats[assignments.cpu() == i].T)
        #     a.append(feats[assignments == i].T)
        covariance = np.array([np.cov(feats[assignments.cpu() == i].T)
            # covariance = np.array([np.cov(feats[assignments == i].T)
                            for i in range(args.num_prototypes)])

        # the semantic shift vectors are enough.
        semantic_shift_vectors = []
        for cov in covariance:
            semantic_shift_vectors.append(
                # sample shift vector from zero-mean multivariate Gaussian distritbuion N(0, cov)
                np.random.multivariate_normal(np.zeros(cov.shape[0]), cov,
                                            size=args.num_shift_vectors))

        semantic_shift_vectors = np.array(semantic_shift_vectors)
        prototypes.append(centroids)
        semantic_shifts.append(semantic_shift_vectors)
        del feats
        real_train_list.append(feat_pth)
        # except:
        #     print("failed to reduce train id: " + feat_pth)

    # prototypes = torch.stack(prototypes) # shape = [num_samples, num_clusters, feat_dim]
    # prototypes = prototypes.numpy()
    prototypes = np.array(prototypes,dtype=np.float64)
    # semantic_shifts = np.array(semantic_shifts)
    os.makedirs(f'datasets/{args.dataset}/dirichlet_clustered', exist_ok=True)
    np.save(f'datasets/{args.dataset}/dirichlet_clustered/train_bag_feats_proto_{args.num_prototypes}_DP.npy', prototypes)
    np.save(f'datasets/{args.dataset}/dirichlet_clustered/train_bag_feats_shift_{args.num_prototypes}_DP.npy', semantic_shifts)

    # gt_labels_path = 'datasets/COAD/COAD_patient_label.pkl'
    # coad_wsi_list = np.load(gt_labels_path, allow_pickle=True)
    # train_list = [os.path.basename(x).split('.')[0] for x in real_train_list]
    # train_labels = []
    # train_list_txt = open(f'datasets/{args.dataset}/remix_processed/train_list.txt', 'w')
    # for i in train_list:
    #     classname = '0-normal-npy' if coad_wsi_list[i] == 0 else '1-tumor-npy'
    #     train_list_txt.write(os.path.join(f'datasets/{args.dataset}', classname, i+'.npy')+','+str(coad_wsi_list[i])+'\n')
    #     train_labels.append(coad_wsi_list[i])
    # np.save(f'datasets/{args.dataset}/remix_processed/train_bag_labels_v2.npy', train_labels)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='base dictionary construction')
    parser.add_argument('--dataset', type=str, default='Camelyon')
    parser.add_argument('--num_prototypes', type=int, default=8)
    parser.add_argument('--num_shift_vectors', type=int, default=200)
    parser.add_argument('--lr', default=0.0002, type=float, help='Initial learning rate [0.0002]')
    parser.add_argument('--weight_decay', default=5e-3, type=float, help='Weight decay [5e-3]')
    parser.add_argument('--num_epochs', default=50, type=int, help='Number of total training epochs')
    args = parser.parse_args()
    train_list = f'datasets/{args.dataset}16/remix_processed/train_list.txt'
    train_list = open(train_list, 'r').readlines()
    train_list = [x.split(',')[0] for x in train_list]  # file names
    train_labels_pth = f'datasets/{args.dataset}16/remix_processed/train_bag_labels.npy'
    train_label = np.load(train_labels_pth)
    test_list = f'datasets/{args.dataset}16/remix_processed/test_list.txt'
    test_list = open(test_list,'r').readlines()
    test_list = [x.split(',')[0] for x in test_list]
    test_labels_pth = f'datasets/{args.dataset}16/remix_processed/test_bag_labels.npy'
    test_label = np.load(test_labels_pth)

    model = HDP_binary_classifier(concentration=0.1,trunc=2, eta=1, batch_size=1, MC_num=100, dim=512, n_sample=1).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.9), weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epochs, 0.000005)

    for epoch in range(args.num_epochs):
        shuffled_train_idxs = np.random.permutation(len(train_label))
        train_list, train_label = [train_list[index] for index in shuffled_train_idxs], train_label[shuffled_train_idxs]
        train_loss_bag = train(train_list, train_label, model, criterion, optimizer)
        print('Epoch [%d/%d] train loss: %.4f' % (epoch, args.num_epochs, train_loss_bag))
        scheduler.step()
    # train(train_list, train_label, model, criterion, optimizer)
    precision, recall, accuracy, avg, auc = test(test_list, test_label, model, criterion)
    print(f'Precision, Recall, Accuracy, Avg, AUC')
    print((f'{precision*100:.2f} {recall*100:.2f} {accuracy*100:.2f} {avg*100:.2f} {auc*100:.2f}'))


