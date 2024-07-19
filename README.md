# Notice
This repository is the official implementation of the paper [cDP-MIL: Robust Multiple Instance Learning via Cascaded Dirichlet Process](https://arxiv.org/abs/2207.01805).

However, the codes need some reorganization and we will update manuscript on how to use the codes soon :). 

[//]: # (# [ECCV2024] cDP-MIL: Robust Multiple Instance Learning via Cascaded Dirichlet Process)

[//]: # ()
[//]: # ([//]: # &#40;This repository holds the Pytorch implementation for the ReMix augmentation described in the paper &#41;)
[//]: # ()
[//]: # ([//]: # &#40;> [**ReMix: A General and Efficient Framework for Multiple Instance Learning based Whole Slide Image Classification**]&#40;https://arxiv.org/abs/2207.01805&#41;,  &#41;)
[//]: # ()
[//]: # ([//]: # &#40;> Jiawei Yang, Hanbo Chen, Yu Zhao, Fan Yang,  Yao Zhang, Lei He, and Jianhua Yao    &#41;)
[//]: # ()
[//]: # ([//]: # &#40;> International Conference on Medical Image Computing and Computer Assisted Intervention &#40;MICCAI&#41;, 2022 &#41;)
[//]: # ()
[//]: # ()
[//]: # ()
[//]: # (<p align="center">)

[//]: # (  <img src="Framework.png" width="1000">)

[//]: # (</p>)

[//]: # ()
[//]: # ()
[//]: # ([//]: # &#40;# Installation&#41;)
[//]: # ()
[//]: # ([//]: # &#40;&#41;)
[//]: # ([//]: # &#40;We use [Remix]&#40;https://github.com/1st-Yasuo/ReMix&#41; as the original codebase.&#41;)
[//]: # ()
[//]: # (# Data Download)

[//]: # (We use two dataset projects in our paper for demonstration: 1&#41; [Camelyon16]&#40;https://camelyon16.grand-challenge.org/&#41; and 2&#41; [TCGA]&#40;https://portal.gdc.cancer.gov/&#41;. )

[//]: # ()
[//]: # (You may follow the instructions in the websites to download the data.)

[//]: # ()
[//]: # (# Crop Slide and Feature Extraction)

[//]: # (We crop slides with magnification parameter set to 20 &#40;level 0&#41; and features are extracted using pretrained KimiaNet.)

[//]: # ()
[//]: # ([//]: # &#40;For implementation details, please refer to our previous project [WSI-HGNN]&#40;https://github.com/HKU-MedAI/WSI-HGNN&#41;.&#41;)
[//]: # ()
[//]: # (# Model Training)

[//]: # (In order to train a cDP-MIL model, you need to firstly aggregate the extracted features and then use the aggrgated features for prediction. )

[//]: # (So basically, the training module contains two step: aggregation and prediction.)

[//]: # (## DP Aggregation)

[//]: # ()
[//]: # (```shell)

[//]: # (python DP_feats_aggr.py --dataset LUAD)

[//]: # (```)

[//]: # (## Model Training and Evaluation)

[//]: # ()
[//]: # (```shell)

[//]: # (python main.py --dataset LUAD --num_epochs 200 --feat_dim 1024 --rep 5 --task binary)

[//]: # (```)

[//]: # ()
[//]: # (# Disclaimer)

[//]: # (Our code is based on [Remix]&#40;https://github.com/1st-Yasuo/ReMix&#41;.)

[//]: # ()
[//]: # (# Citation)

[//]: # (Please consider citing our paper in your publications if the project helps your research.)



