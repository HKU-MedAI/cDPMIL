import glob
import os.path
import shutil

import numpy as np
import torch


dataset = 'LUAD'
# wsi_path = "/data1/WSI/Patches/Features/"+dataset+"/"+dataset+"_Kimia_20x/*"
wsi_path = ['/data1/WSI/Patches/Features/BRACS_WSI/BRACS_WSI_Kimia_20x/*']
# des_path = "/data1/WSI/Patches/Features/"+dataset+"/"+dataset+"_Kimia_20x_Little"
# wsi_dirs = glob.glob(wsi_path)
wsi_dirs = list()
for i in wsi_path:
    wsi_dirs.extend(glob.glob(i))

little_wsi = list()
node_nums = []
for wsi in wsi_dirs:
    feature_path = os.path.join(wsi, 'features.pt')
    features = torch.load(feature_path, map_location=lambda storage, loc: storage)
    node_num = features.shape[0]
    node_nums.append(node_num)
    if node_num <= 30:
        little_wsi.append(wsi)
print(np.mean(node_nums))

# if not os.path.exists(des_path):
#     os.mkdir(des_path)
#
# source2des = list()
# for wsi in little_wsi:
#     des = os.path.join(des_path, wsi.split('/')[-1])
#     source2des.append([wsi, des])
#
# for x in source2des:
#     shutil.move(x[0], x[1])
