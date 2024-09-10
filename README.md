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
and you will find the aggregated feature files in your designated directory.

So basically, the training module contains two step: aggregation and prediction.


## Model Training and Evaluation


```shell

python main.py ----train_wsi_file Yours --test_wsi_file Yours --train_label_file Yours --test_label_file Yours --aggr_feat_dir Yours --feat_dim 1024/512...

```
WSI file refers to the file that contains the wsi names and they should match the corresponding label file. Aggregated feature directory refers to the directory that you store the features generated in the last step.
Results will be uploaded to [wandb](https://github.com/wandb/wandb).

# Disclaimer

Our code is based on [Remix](https://github.com/1st-Yasuo/ReMix).


# Citation

Please consider citing our paper in your publications if the project helps your research.
```
@article{chen2024cdp,
  title={cDP-MIL: Robust Multiple Instance Learning via Cascaded Dirichlet Process},
  author={Chen, Yihang and Chan, Tsai Hor and Yin, Guosheng and Jiang, Yuming and Yu, Lequan},
  journal={arXiv preprint arXiv:2407.11448},
  year={2024}
}
```



