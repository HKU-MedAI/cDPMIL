# Notice
This repository is the official implementation of the paper [cDP-MIL: Robust Multiple Instance Learning via Cascaded Dirichlet Process](https://arxiv.org/abs/2407.11448).

However, the codes need some reorganization and we will update manuscript on how to use the codes soon :). 

# [ECCV2024] cDP-MIL: Robust Multiple Instance Learning via Cascaded Dirichlet Process



<p align="center">

  <img src="Framework.png" width="1000">

</p>


# Data Download

We use two dataset projects in our paper for demonstration: 1) [Camelyon16](https://camelyon16.grand-challenge.org/) and 2) [TCGA](https://portal.gdc.cancer.gov/). 


You may follow the instructions in the websites to download the data.


# WSI Preprocessing and Feature Extraction

We follow the preprocessing steps of [DSMIL](https://github.com/binli123/dsmil-wsi) and features are extracted using pretrained [KimiaNet](https://github.com/KimiaLabMayo/KimiaNet). Of course, you can use any feature extractor you want.


# Model Training

In order to train a cDP-MIL model, you need to firstly aggregate the extracted features and then use the aggrgated features for prediction. 

## Feature Aggregation

Extracted features should be arranged as below:
```
Feature Directory
|-- WSI
|   |-- features.pt (or other format)
```
Then, run the command:
```
  $ python DP_feats_aggr.py --feat_dir Yours --save_dir Yours
```
and you will find the aggregated feature files in your designed directory.

So basically, the training module contains two step: aggregation and prediction.

## DP Aggregation


```shell

python DP_feats_aggr.py --dataset LUAD

```

## Model Training and Evaluation


```shell

python main.py --dataset LUAD --num_epochs 200 --feat_dim 1024 --rep 5 --task binary

```


# Disclaimer

Our code is based on [Remix](https://github.com/1st-Yasuo/ReMix).


# Citation

Please consider citing our paper in your publications if the project helps your research.



