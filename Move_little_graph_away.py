import glob
import os.path
import shutil

import torch


dataset = "ESCA"
wsi_path = "/data1/WSI/Patches/Features/"+dataset+"/"+dataset+"_Tissue_Kimia_20x/*"
des_path = "/data1/WSI/Patches/Features/"+dataset+"/"+dataset+"_Tissue_Kimia_20x_Little"
wsi_dirs = glob.glob(wsi_path)

little_wsi = list()
for wsi in wsi_dirs:
    feature_path = os.path.join(wsi, 'features.pt')
    features = torch.load(feature_path, map_location=lambda storage, loc: storage)
    node_num = features.shape[0]
    if node_num <= 100:
        little_wsi.append(wsi)

if not os.path.exists(des_path):
    os.mkdir(des_path)

source2des = list()
for wsi in little_wsi:
    des = os.path.join(des_path, wsi.split('/')[-1])
    source2des.append([wsi, des])

for x in source2des:
    shutil.move(x[0], x[1])
