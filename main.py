import argparse
import os

import copy
import numpy as np
import torch
from tqdm import tqdm

from tools.clustering import Kmeans
from model.dpmil import DP_Cluster_EM, DP_Cluster_VI, DP_Classifier, BClassifier
from collections import Counter
import random
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             roc_auc_score, roc_curve)
import math
import wandb
from pyhealth.metrics import binary_metrics_fn, multiclass_metrics_fn
os.environ['CUDA_VISIBLE_DEVICES']='3'

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

def train(train_list, train_labels, model, criterion, optimizer, eta_cluster, feat_dim, dataset, mode='fix'):

    model.train()
    total_loss = 0
    for i,feat_pth,label in tqdm(zip(range(len(train_list)),train_list,train_labels)):
        if mode == 'fix':
            optimizer.zero_grad()
            if dataset == 'Camelyon':
                feat_pth = feat_pth[20:-4]+'.pt'
                bag_feats = torch.load('/home/r20user8/Documents/HDPMIL/datasets/'+dataset+'/HDP_feats/'+feat_pth)
                bag_feats = bag_feats.to(device)
            bag_prediction = model(bag_feats.float())
            bag_prediction = torch.mean(bag_prediction, axis=0).unsqueeze(0)
            # MC_num = bag_prediction.shape[0]
            bag_label = torch.tensor([label]).to(device)
            bag_loss = criterion(bag_prediction, bag_label)
            loss = bag_loss

            loss.backward()
            optimizer.step()
            total_loss = total_loss + loss.item()
            print('\r Training bag [%d/%d] bag loss: %.4f' % (i, len(train_list), loss.item()))

        else:
            optimizer.zero_grad()
            if dataset == 'Camelyon':
                bag_feats = np.load(feat_pth)
                bag_feats = torch.tensor(bag_feats).to(device)
            else:
                bag_feats = torch.load(feat_pth+'/features.pt')
                bag_feats = bag_feats.to(device)
            dp_cluster = DP_Cluster_EM(trunc=10, eta=eta_cluster, batch_size=1, epoch=20, dim=feat_dim).to(
                device)
            logits = dp_cluster(bag_feats)
            assignments = torch.argmax(logits, dim=1)
            # num_cluster = len(torch.unique(assignments))
            centroids = [torch.mean(bag_feats[assignments == i], dim=0) for i in torch.unique(assignments)]
            centroids = torch.stack(centroids) # [num_cluster, dim]
            # centroids = np.array([np.mean(bag_feats[assignments == i], axis=0) for i in range(args.num_prototypes)])
            # abort invalid features
            if torch.isnan(bag_feats).sum() > 0:
                continue
            bag_prediction = model(centroids.float())
            bag_prediction = torch.mean(bag_prediction, axis=0).unsqueeze(0)
            # MC_num = bag_prediction.shape[0]
            bag_label = torch.tensor([label]).to(device)
            bag_loss = criterion(bag_prediction,bag_label)
            loss = bag_loss

            loss.backward()
            optimizer.step()
            total_loss = total_loss + loss.item()
            print('\r Training bag [%d/%d] bag loss: %.4f' % (i, len(train_list), loss.item()))
    return total_loss/len(train_list)

def test_binary(test_list, test_labels, model, criterion, eta_cluster, feat_dim, dataset, mode='fix'):
    model.eval()
    total_loss = 0
    test_predictions = []

    for i,(feat_pth,label) in enumerate(zip(test_list,test_labels)):
        if mode == 'fix':
            if dataset == 'Camelyon':
                feat_pth = feat_pth[20:-4]+'.pt'
                bag_feats = torch.load('/home/r20user8/Documents/HDPMIL/datasets/'+dataset+'/HDP_feats/'+feat_pth)
                bag_feats = bag_feats.to(device)
                with torch.no_grad():
                    bag_prediction = model(bag_feats.float())
                    bag_prediction = torch.mean(bag_prediction, axis=0).unsqueeze(0)
                    bag_label = torch.tensor([label]).to(device)
                    loss = criterion(bag_prediction,bag_label)
                    total_loss = total_loss + loss.item()
                    print('\r Testing bag [%d/%d] bag loss: %.4f' % (i, len(test_list), loss.item()))
                    prob = torch.nn.Softmax(dim=1)(bag_prediction)
                    test_predictions.append(prob.squeeze().cpu().numpy())
        else:
            if dataset == 'Camelyon':
                bag_feats = np.load(feat_pth)
                bag_feats = torch.tensor(bag_feats).to(device)
            else:
                bag_feats = torch.load(feat_pth+'/features.pt')
                bag_feats = bag_feats.to(device)
            dp_cluster = DP_Cluster_EM(trunc=10, eta=eta_cluster, batch_size=1, epoch=20, dim=feat_dim).to(
                device)
            logits = dp_cluster(bag_feats)
            assignments = torch.argmax(logits, dim=1)
            # num_cluster = len(torch.unique(assignments))
            centroids = [torch.mean(bag_feats[assignments == i], dim=0) for i in torch.unique(assignments)]
            centroids = torch.stack(centroids)
            with torch.no_grad():
                bag_prediction = model(centroids.float())
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

    # a = test_predictions

    # class_prediction_bag = copy.deepcopy(test_predictions)
    # class_prediction_bag[test_predictions >= thresholds_optimal[0]] = 1
    # class_prediction_bag[test_predictions < thresholds_optimal[0]] = 0
    # test_predictions = class_prediction_bag
    # test_labels = np.squeeze(test_labels)
    y_pred, y_true = inverse_convert_label(test_predictions), inverse_convert_label(test_labels)

    res = binary_metrics_fn(y_true, test_predictions, threshold=thresholds_optimal[0],
                            metrics=['accuracy', 'precision', 'recall', 'roc_auc', 'f1'])
    acc = res['accuracy']
    p = res['precision']
    r = res['recall']
    c_auc = res['roc_auc']
    f1 = res['f1']
    # p = precision_score(y_true, y_pred, average='macro')
    # r = recall_score(y_true, y_pred, average='macro')
    # acc = accuracy_score(y_true, y_pred)
    avg = np.mean([p, r, acc, f1])
    # c_auc = roc_auc_score(y_true, y_pred)


    return p, r, acc, f1, avg, c_auc

def test_multiclass(test_list, test_labels, model, criterion, eta_cluster, feat_dim, dataset):
    model.eval()
    total_loss = 0
    test_predictions = []

    for i,(feat_pth,label) in enumerate(zip(test_list,test_labels)):
        if dataset == 'Camelyon':
            bag_feats = np.load(feat_pth)
            bag_feats = torch.tensor(bag_feats).to(device)
        else:
            bag_feats = torch.load(feat_pth+'/features.pt')
            bag_feats = bag_feats.to(device)
        dp_cluster = DP_Cluster(concentration=0.1, trunc=10, eta=eta_cluster, batch_size=1, epoch=20, dim=feat_dim).to(
            device)
        logits = dp_cluster(bag_feats)
        assignments = torch.argmax(logits, dim=1)
        # num_cluster = len(torch.unique(assignments))
        centroids = [torch.mean(bag_feats[assignments == i], dim=0) for i in torch.unique(assignments)]
        centroids = torch.stack(centroids)
        with torch.no_grad():
            bag_prediction = model(centroids)
            # bag_prediction = torch.mean(bag_prediction, axis=0).unsqueeze(0)
            bag_label = torch.tensor([label]).to(device)
            loss = criterion(bag_prediction,bag_label)
            total_loss = total_loss + loss.item()
            print('\r Testing bag [%d/%d] bag loss: %.4f' % (i, len(test_list), loss.item()))
            prob = torch.nn.Softmax(dim=1)(bag_prediction)
            test_predictions.append(prob.squeeze().cpu().numpy())

    test_predictions = np.array(test_predictions)
    # test_predictions = test_predictions[:,1]

    # a = test_predictions

    # class_prediction_bag = copy.deepcopy(test_predictions)
    # class_prediction_bag[test_predictions >= thresholds_optimal[0]] = 1
    # class_prediction_bag[test_predictions < thresholds_optimal[0]] = 0
    # test_predictions = class_prediction_bag
    # test_labels = np.squeeze(test_labels)
    y_pred, y_true = inverse_convert_label(test_predictions), inverse_convert_label(test_labels)

    res = multiclass_metrics_fn(y_true, test_predictions, metrics=["roc_auc_weighted_ovo","f1_weighted","accuracy"])
    acc = res['accuracy']
    f = res['f1_weighted']
    # r = res['recall']
    c_auc = res['roc_auc_weighted_ovo']
    # p = precision_score(y_true, y_pred, average='macro')
    # r = recall_score(y_true, y_pred, average='macro')
    # acc = accuracy_score(y_true, y_pred)
    # avg = np.mean([p, r, acc])
    # c_auc = roc_auc_score(y_true, y_pred)


    return acc,f,c_auc

def inverse_convert_label(labels):
    # one-hot decoding
    if len(np.shape(labels)) == 1:
        return labels
    else:
        converted_labels = np.zeros(len(labels))
        for ix in range(len(labels)):
            converted_labels[ix] = np.argmax(labels[ix])
            converted_labels = np.array(converted_labels,dtype=int)
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

# reduce function is not used
# def reduce(args, train_list, train_label):
#     setup_seed(1)
#     prototypes = []
#     semantic_shifts = []
#     real_train_list = []
#     for feat_pth,label in tqdm(zip(train_list,train_label)):
#         # try:
#         feats = np.load(feat_pth)
#         # feats = np.ascontiguousarray(feats, dtype=np.float32)
#         feats = (feats-np.mean(feats, axis=0))/(np.std(feats, axis=0))
#         # # feats = (feats - np.min(feats, axis=1, keepdims=True))
#         # # (np.max(feats, axis=1, keepdims=True) - np.min(feats, axis=1, keepdims=True))
#         #
#         feats = torch.tensor(feats).to(device)
#         D_P_cluster = DirichletProcess(concentration=0.1,trunc=args.num_prototypes,eta=1,batch_size=1,dim=512).to(device)
#         neg_likelyhood = -D_P_cluster.likelihood(feats)
#
#
#         # dp_cluster = DP_Cluster(concentration=10,trunc=args.num_prototypes,eta=1,batch_size=1,epoch=20, dim=512).to(device)
#         # logits = dp_cluster(feats)
#         num_MC = 100
#         betas = D_P_cluster.sample_beta(num_MC)
#         weights = D_P_cluster.mix_weights(betas)[:, :-1]
#         normalize_factor = torch.transpose(torch.vstack((torch.sum(weights, axis=1), torch.sum(weights, axis=1))), 0, 1)
#         weights = weights/normalize_factor
#         MC_samples = []
#         for j in range(num_MC):
#             prob = weights[j,:].cpu().numpy()
#             draw = np.random.choice(args.num_prototypes,size=1,p=prob)[0]
#             x = D_P_cluster.gaussians[draw].sample(1)[0,:]
#             MC_samples.append(x)
#         MC_samples = torch.stack(MC_samples)
#         D_P_clssfy = DirichletProcess(concentration=0.1,trunc=args.num_prototypes,eta=1,batch_size=1,dim=512).to(device)
#         logits = D_P_clssfy.infer(MC_samples)
#         bag_label = torch.tensor([label] * num_MC)
#
#
#         # assignments = torch.argmax(logits, dim=1)
#         # logits = logits.cpu().numpy()
#         feats = np.load(feat_pth)
#         # feats = (feats-np.mean(feats,axis=0))/(np.std(feats,axis=0))
#         # # feats = (feats - np.min(feats, axis=1, keepdims=True))
#         # # (np.max(feats, axis=1, keepdims=True) - np.min(feats, axis=1, keepdims=True))
#         # kmeans = Kmeans(k=args.num_prototypes, pca_dim=-1)
#         # kmeans.cluster(feats, seed=66)  # for reproducibility
#         # assignments = kmeans.labels.astype(np.int64)
#         # # a = Counter(assignments)
#         # # compute the centroids for each cluster
#         # # centroids = torch.zeros(args.num_prototypes,feats.shape[1])
#         # # for i in range(args.num_prototypes):
#         # #     centroids[i] = torch.mean(feats[assignments == i], axis=0)
#         centroids = np.array([np.mean(feats[assignments.cpu() == i], axis=0)
#                             for i in range(args.num_prototypes)])
#         # centroids = np.array([np.mean(feats[assignmentsyi == i], axis=0)
#         #                     for i in range(args.num_prototypes)])
#         # centroids = torch.tensor([torch.mean(feats[assignments == i], axis=0)
#         #                     for i in range(args.num_prototypes)])
#
#         # compute covariance matrix for each cluster
#         # a = []
#         # for i in range(args.num_prototypes):
#         #     # a.append(feats[assignments.cpu() == i].T)
#         #     a.append(feats[assignments == i].T)
#         covariance = np.array([np.cov(feats[assignments.cpu() == i].T)
#             # covariance = np.array([np.cov(feats[assignments == i].T)
#                             for i in range(args.num_prototypes)])
#
#         # the semantic shift vectors are enough.
#         semantic_shift_vectors = []
#         for cov in covariance:
#             semantic_shift_vectors.append(
#                 # sample shift vector from zero-mean multivariate Gaussian distritbuion N(0, cov)
#                 np.random.multivariate_normal(np.zeros(cov.shape[0]), cov,
#                                             size=args.num_shift_vectors))
#
#         semantic_shift_vectors = np.array(semantic_shift_vectors)
#         prototypes.append(centroids)
#         semantic_shifts.append(semantic_shift_vectors)
#         del feats
#         real_train_list.append(feat_pth)
#         # except:
#         #     print("failed to reduce train id: " + feat_pth)
#
#     # prototypes = torch.stack(prototypes) # shape = [num_samples, num_clusters, feat_dim]
#     # prototypes = prototypes.numpy()
#     prototypes = np.array(prototypes,dtype=np.float64)
#     # semantic_shifts = np.array(semantic_shifts)
#     os.makedirs(f'datasets/{args.dataset}/dirichlet_clustered', exist_ok=True)
#     np.save(f'datasets/{args.dataset}/dirichlet_clustered/train_bag_feats_proto_{args.num_prototypes}_DP.npy', prototypes)
#     np.save(f'datasets/{args.dataset}/dirichlet_clustered/train_bag_feats_shift_{args.num_prototypes}_DP.npy', semantic_shifts)
#
#     # gt_labels_path = 'datasets/COAD/COAD_patient_label.pkl'
#     # coad_wsi_list = np.load(gt_labels_path, allow_pickle=True)
#     # train_list = [os.path.basename(x).split('.')[0] for x in real_train_list]
#     # train_labels = []
#     # train_list_txt = open(f'datasets/{args.dataset}/remix_processed/train_list.txt', 'w')
#     # for i in train_list:
#     #     classname = '0-normal-npy' if coad_wsi_list[i] == 0 else '1-tumor-npy'
#     #     train_list_txt.write(os.path.join(f'datasets/{args.dataset}', classname, i+'.npy')+','+str(coad_wsi_list[i])+'\n')
#     #     train_labels.append(coad_wsi_list[i])
#     # np.save(f'datasets/{args.dataset}/remix_processed/train_bag_labels_v2.npy', train_labels)

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
        train_labels_pth = f'datasets/{args.dataset}16/remix_processed/train_bag_labels.npy'
        train_label = np.load(train_labels_pth)
        test_list = f'datasets/{args.dataset}16/remix_processed/test_list.txt'
        test_list = open(test_list,'r').readlines()
        test_list = [x.split(',')[0] for x in test_list]
        test_labels_pth = f'datasets/{args.dataset}16/remix_processed/test_bag_labels.npy'
        test_label = np.load(test_labels_pth)
    else:
        train_list = f'datasets/{args.dataset}/{args.task}_{args.dataset}_train.txt'
        train_list = open(train_list, 'r').readlines()
        train_list = [x.split('\n')[0] for x in train_list]
        train_labels_pth = f'datasets/{args.dataset}/{args.task}_{args.dataset}_train_label.npy'
        train_label = np.load(train_labels_pth)-int(args.task=='staging')
        test_list = f'datasets/{args.dataset}/{args.task}_{args.dataset}_testval.txt'
        test_list = open(test_list, 'r').readlines()
        test_list = [x.split('\n')[0] for x in test_list]
        test_labels_pth = f'datasets/{args.dataset}/{args.task}_{args.dataset}_testval_label.npy'
        test_label = np.load(test_labels_pth)-int(args.task=='staging')

        # eta [0.1-10]
    # [ 1, 3, 5, 7, 10]
    lr = 0.1
    eta_cluster = 7
    eta_classifier = 10
    config = {"eta_cluster":1,"eta_classifier":1,"lr":lr}
    for eta_cluster in [7]:
        for eta_classifier in [10]:
    # for lr in [ # for (eta_classifier,eta_cluster)=(5,5)
    #     0.00005, 0.0001, 0.0005, 0.001, 0.01, 0.1, 1
    # ]:
        # config["eta_cluster"] = eta_cluster
        # config["eta_classifier"] = eta_classifier
        #     else:
            config["eta_cluster"] = eta_cluster
            config["eta_classifier"] = eta_classifier
            # wandb.init(name=args.dataset+'_HDPMIL_DP(EM)+DP(VI)_'+args.task,
            #            project='HDPMIL',
            #            entity='yihangc',
            #            notes='',
            #            mode='online', # disabled/online/offline
            #            config=config,
            #            tags=[])
            # model = BClassifier(input_size=args.feat_dim,num_classes=args.num_classes).to(device)
            model = DP_Classifier(concentration=0.1,trunc=args.num_classes,eta=eta_classifier,batch_size=1,dim=args.feat_dim,n_sample=1).to(device)
            # model = HDP_binary_classifier(concentration=0.1,trunc=2, eta=1, batch_size=1, MC_num=100, dim=512, n_sample=1).to(device)
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.9), weight_decay=args.weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epochs, 0.000005)
            for epoch in range(args.num_epochs):
                shuffled_train_idxs = np.random.permutation(len(train_label))
                train_list, train_label = [train_list[index] for index in shuffled_train_idxs], train_label[shuffled_train_idxs]
                train_loss_bag = train(train_list, train_label, model, criterion, optimizer, eta_cluster, args.feat_dim, args.dataset)
                if args.task == 'binary':
                    precision, recall, accuracy, f1, avg, auc = test_binary(test_list, test_label, model, criterion, eta_cluster, args.feat_dim, args.dataset)
                    wandb.log({'train_loss': train_loss_bag, 'precision': precision, 'recall': recall, 'accuracy': accuracy, 'f1':f1,
                               'avg': avg, 'auc': auc})
                elif args.task=='staging':
                    accuracy, f1, auc = test_multiclass(test_list, test_label, model, criterion, eta_cluster, args.feat_dim, args.dataset)
                    wandb.log({'accuracy': accuracy, 'f1': f1, 'auc': auc})
                print('Epoch [%d/%d] train loss: %.4f' % (epoch, args.num_epochs, train_loss_bag))
                scheduler.step()

                        # train(train_list, train_label, model, criterion, optimizer)
                        # precision, recall, accuracy, avg, auc = test(test_list, test_label, model, criterion)
                        # print(f'Precision, Recall, Accuracy, Avg, AUC')
                        # print((f'{precision*100:.2f} {recall*100:.2f} {accuracy*100:.2f} {avg*100:.2f} {auc*100:.2f}'))
            # wandb.finish()