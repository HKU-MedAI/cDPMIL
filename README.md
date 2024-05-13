# [ECCV2024 Under Review] cDP-MIL: Robust Multiple Instance Learning via Cascaded Dirichlet Process

[//]: # (This repository holds the Pytorch implementation for the ReMix augmentation described in the paper )

[//]: # (> [**ReMix: A General and Efficient Framework for Multiple Instance Learning based Whole Slide Image Classification**]&#40;https://arxiv.org/abs/2207.01805&#41;,  )

[//]: # (> Jiawei Yang, Hanbo Chen, Yu Zhao, Fan Yang,  Yao Zhang, Lei He, and Jianhua Yao    )

[//]: # (> International Conference on Medical Image Computing and Computer Assisted Intervention &#40;MICCAI&#41;, 2022 )



<p align="center">
  <img src="Framework.png" width="1000">
</p>


[//]: # (# Installation)

[//]: # ()
[//]: # (We use [Remix]&#40;https://github.com/1st-Yasuo/ReMix&#41; as the original codebase.)

# Data Download
We use two dataset projects in our paper for demonstration: 1) [Camelyon16](https://camelyon16.grand-challenge.org/) and 2) [TCGA](https://portal.gdc.cancer.gov/). 

You may follow the instructions in the websites to download the data.

# Crop Slide and Feature Extraction
We crop slides with magnification parameter set to 20 (level 0) and features are extracted using pretrained KimiaNet.

[//]: # (For implementation details, please refer to our previous project [WSI-HGNN]&#40;https://github.com/HKU-MedAI/WSI-HGNN&#41;.)

# Model Training
In order to train a cDP-MIL model, you need to firstly aggregate the extracted features and then use the aggrgated features for prediction. 
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



