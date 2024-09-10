import argparse
import os

import numpy as np
import torch
from tqdm import tqdm
from model.dpmil import  DP_Classifier
from sklearn.metrics import (roc_auc_score, roc_curve)
import time
import wandb
from pyhealth.metrics import binary_metrics_fn, multiclass_metrics_fn
os.environ['CUDA_VISIBLE_DEVICES']='5'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(train_list, train_labels, model, criterion, optimizer, args):

    model.train()
    total_loss = 0
    for i,slide,label in tqdm(zip(range(len(train_list)),train_list,train_labels)):
        optimizer.zero_grad()

        slide_name = slide.split('/')[-1].split('\n')[0]
        feat_pth = f'{args.aggr_feat_dir}/{slide_name}.npy'
        centroids = np.load(feat_pth)
        centroids = torch.from_numpy(centroids).to(device)

        bag_prediction = model(centroids.float())
        bag_prediction = torch.mean(bag_prediction, dim=0).unsqueeze(0)
        bag_label = torch.tensor([label]).to(device)
        bag_loss = criterion(bag_prediction, bag_label)
        loss = bag_loss

        loss.backward()
        optimizer.step()
        total_loss = total_loss + loss.item()
        print('\r Training bag [%d/%d] bag loss: %.4f' % (i, len(train_list), loss.item()))
    return total_loss/len(train_list)


def test_binary(test_list, test_labels, model, criterion,  args):
    model.eval()
    total_loss = 0
    test_predictions = []

    for i,(slide,label) in enumerate(zip(test_list,test_labels)):
        slide_name = slide.split('/')[-1].split('\n')[0]
        feat_pth = f'{args.aggr_feat_dir}/{slide_name}.npy'
        centroids = np.load(feat_pth)
        centroids = torch.from_numpy(centroids).to(device)

        with torch.no_grad():
            bag_prediction = model(centroids.float())
            bag_prediction = torch.mean(bag_prediction, dim=0).unsqueeze(0)
            bag_label = torch.tensor([label]).to(device)
            loss = criterion(bag_prediction,bag_label)
            total_loss = total_loss + loss.item()
            print('\r Testing bag [%d/%d] bag loss: %.4f' % (i, len(test_list), loss.item()))
            prob = torch.nn.Softmax(dim=1)(bag_prediction)
            test_predictions.append(prob.squeeze().cpu().numpy())

    test_predictions = np.array(test_predictions)
    test_predictions = test_predictions[:,1]
    _, _, thresholds_optimal = multi_label_roc(test_labels, test_predictions, 1)

    y_pred, y_true = inverse_convert_label(test_predictions), inverse_convert_label(test_labels)

    res = binary_metrics_fn(y_true, test_predictions,
                            metrics=['accuracy', 'precision', 'recall', 'roc_auc', 'f1'])
    acc = res['accuracy']
    p = res['precision']
    r = res['recall']
    c_auc = res['roc_auc']
    f1 = res['f1']
    avg = np.mean([p, r, acc, f1])


    return p, r, acc, f1, avg, c_auc


def test_multiclass(test_list, test_labels, model, criterion, args):
    model.eval()
    total_loss = 0
    test_predictions = []

    for i,(slide,label) in enumerate(zip(test_list,test_labels)):

        slide_name = slide.split('/')[-1].split('\n')[0]
        feat_pth = f'{args.aggr_feat_dir}/{slide_name}.npy'
        centroids = np.load(feat_pth)
        centroids = torch.from_numpy(centroids).to(device)

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
    y_pred, y_true = inverse_convert_label(test_predictions), inverse_convert_label(test_labels)

    res = multiclass_metrics_fn(y_true, test_predictions, metrics=["roc_auc_weighted_ovo","f1_weighted","accuracy"])
    acc = res['accuracy']
    f = res['f1_weighted']
    c_auc = res['roc_auc_weighted_ovo']


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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='base dictionary construction')
    # label file: .npy format
    parser.add_argument('--train_label_file',type=str)
    parser.add_argument('--test_label_file',type=str)
    # wsi names file: .txt format
    # label and wsi should match
    parser.add_argument('--train_wsi_file',type=str)
    parser.add_argument('--test_wsi_file',type=str)
    # aggregated features directory
    parser.add_argument('--aggr_feat_dir',type=str)
    parser.add_argument('--lr', default=0.0002, type=float, help='Initial learning rate [0.0002]')
    parser.add_argument('--weight_decay', default=5e-3, type=float, help='Weight decay [5e-3]')
    parser.add_argument('--num_epochs', default=50, type=int, help='Number of total training epochs')
    parser.add_argument('--num_classes', default=2, type=int, help='Number of total classes in classification task')
    parser.add_argument('--feat_dim', default=512, type=int, help='feature dimension')
    parser.add_argument('--task', default='binary', help='binary or multiclass')
    args = parser.parse_args()
    wandb.init(name=f'...',
               project='...',
               entity='...',
               notes='',
               mode='online',
               tags=[])

    train_list = open(args.train_wsi_file, 'r').readlines()
    train_list = [x.split(',')[0] for x in train_list]
    test_list = open(args.test_wsi_file, 'r').readlines()
    test_list = [x.split(',')[0] for x in test_list]
    train_label = np.load(args.train_label_file)
    test_label = np.load(args.test_label_file)
    lr = 0.01
    eta_classifier = 10
    num_cluster = 10
    config = {"num_cluster1":30,"num_cluster2":10,"eta_classifier":10,"lr":lr,"rep":0,"n_comp":10,"concentration":0.1}
    mode = 'fix'
    concentration = 0.1
    eta_classifier = 10
    n_comp = 10

    model = DP_Classifier(trunc=args.num_classes,eta=eta_classifier,batch_size=1,dim=args.feat_dim,n_sample=1).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.9), weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epochs, 0.000005)

    for epoch in range(args.num_epochs):
        start_train = time.time()
        shuffled_train_idxs = np.random.permutation(len(train_label))
        train_list, train_label = [train_list[index] for index in shuffled_train_idxs], train_label[shuffled_train_idxs]
        train_loss_bag = train(train_list, train_label, model, criterion, optimizer, args)
        train_end = time.time()

        train_duration = train_end - start_train
        print(
            f'Time elapsed: {train_duration // 3600} hours {(train_duration % 3600) // 60} mins {train_duration % 60}  seconds.')
        start_test = time.time()
        if args.task == 'binary':
            precision, recall, accuracy, f1, avg, auc = test_binary(test_list, test_label, model, criterion,  args)

            print(f"precision:{precision}, recall:{recall}, acc:{accuracy}, f1:{f1}, auc:{auc}.")
            test_end = time.time()
            test_duration = test_end - start_test
            print(
                f'Time elapsed: {test_duration // 3600} hours {(test_duration % 3600) // 60} mins {test_duration % 60}  seconds.')
            wandb.log({'train_loss': train_loss_bag, 'precision': precision, 'recall': recall, 'accuracy': accuracy, 'f1':f1,
                       'avg': avg, 'auc': auc})

        else:
            accuracy, f1, auc = test_multiclass(test_list, test_label, model, criterion,  args)
            wandb.log({'accuracy': accuracy, 'f1': f1, 'auc': auc})

        scheduler.step()


    wandb.finish()