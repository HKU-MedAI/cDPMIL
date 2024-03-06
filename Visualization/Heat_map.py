import glob
import os
from pathlib import Path
from collections import OrderedDict

import cv2
import numpy as np
import pandas as pd
import torch
from openslide import OpenSlide
from openslide.deepzoom import DeepZoomGenerator
from skimage.io import imsave
import matplotlib
import matplotlib.cm as cm
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from sklearn.metrics import precision_score, recall_score, f1_score, auc, roc_curve, roc_auc_score
from xml.dom import minidom

# from .evaluator import Evaluator
# from data import C16EvalDataset
# from parser import parse_gnn_model
# from explainers import GNNExplainer, GemExplainer, HetGemExplainer


def get_magnified_image(wsi_name,level):
    # print("Whole slide image name \t %s" % name)
    wsi_path = f'/data1/public/WSI/Camelyon16/CAMELYON16/{wsi_name}.tif'
    wsi = OpenSlide(wsi_path)
    print("\t Image dimensions @ level 0 \t", wsi.dimensions)
    dim = wsi.level_dimensions[level]
    print("\t Image dimensions @ level " + str(level) + "\t", dim)
    img = wsi.get_thumbnail(dim)
    return img, wsi

def color_map_color(value, cmap_name='Wistia', vmin=0, vmax=1):
    # norm = plt.Normalize(vmin, vmax)
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.get_cmap(cmap_name)  # PiYG
    color = cmap(norm(value))[:, :3]  # will return rgba, we take only first 3 so we get rgb
    return color

def visualize(node_feat_mask, wsi_name, patches_coords, poly_coords, img, level, epoch):
    output_name = os.path.join('./Graphs_DP', wsi_name + ".png")
    imsave(output_name, np.asarray(img))
    img = np.int32(img)

    colours = color_map_color(node_feat_mask)

    for idx, (bbox, cl) in enumerate(zip(patches_coords, colours)):
        cl = [c * 255 for c in cl]
        s = 256
        img = cv2.rectangle(img, (int(bbox[0] + s/2**level), bbox[1]), (bbox[0], int(bbox[1] + s/2**level)), cl, cv2.FILLED)

    for coords in poly_coords:
        mag_fac = 2 ** (level)
        coords = coords.reshape((-1, 1, 2)) / mag_fac
        img = cv2.polylines(img, np.int32([coords]), False, (255, 0, 0), thickness=4)
    os.makedirs(f'./Explain_Graphs_DP2_{epoch}',exist_ok=True)
    output_name = os.path.join( f'./Explain_Graphs_DP2_{epoch}', wsi_name + ".jpeg")
    imsave(output_name, np.uint8(img))

def visualize1(node_feat_mask, wsi_name, patches_coords, poly_coords, img, level):

    img = np.int32(img)
    for coords in poly_coords:
        mag_fac = 2 ** (level)
        coords = coords.reshape((-1, 1, 2)) / mag_fac
        img = cv2.polylines(img, np.int32([coords]), False, (255, 0, 0), thickness=4)

    output_name = os.path.join('./Graphs_Annotated', wsi_name + ".png")
    imsave(output_name, np.uint8(img))


    # colours = color_map_color(node_feat_mask)
    #
    # for idx, (bbox, cl) in enumerate(zip(patches_coords, colours)):
    #     cl = [c * 255 for c in cl]
    #     s = 256
    #     img = cv2.rectangle(img, (int(bbox[0] + s/2**level), bbox[1]), (bbox[0], int(bbox[1] + s/2**level)), cl, cv2.FILLED)
    #
    #
    #
    # output_name = os.path.join( './Explain_Graphs_DP', wsi_name + ".jpeg")
    # imsave(output_name, np.uint8(img))

def get_ground_truths(xml_path, patches_coords, level):
    """
    Ground truth path in xml
    :param gt_path:
    :return:
    """
    polygons = minidom.parse(xml_path).getElementsByTagName("Coordinates")
    polygons_out = []
    polygon_coords = []
    for p in polygons:
        coords = []
        for c in p.childNodes:
            if c.attributes:
                x_coords = c.attributes["X"].value
                y_coords = c.attributes["Y"].value
            else:
                continue
            coords.append((float(x_coords), float(y_coords)))
        coords = np.stack(coords)
        if len(coords)>=4:
            polygon_coords.append(coords)
            polygons_out.append(Polygon(coords))

    gt_labels = []
    for c in patches_coords:
        # Get center
        mag_factor = 2 ** (level)
        s = 256 * 2 // 2  # Patch size at level 0
        c = (k * mag_factor + s for k in c)
        point = Point(c)
        flag = False
        for p in polygons_out:
            if p.contains(point):
                flag = True
        if flag is True:
            gt_labels.append(1)
        else:
            gt_labels.append(0)

    return gt_labels, polygon_coords

level = 5
annotated_slides = glob.glob('/data1/public/WSI/Camelyon16/lesion_annotation/*.xml')
for epoch in [1,10,20,30,40,50,60,70,77]:
    for slide in annotated_slides:
        wsi_name = slide.split('/')[-1][:8]
        img, wsi = get_magnified_image(wsi_name,level)
        coor_feat_mask = pd.read_csv(f'/data1/WSI/Patches/Features/Camelyon16/simclr_files_256_v2/testing/{wsi_name}/DP_insclassifier_prob_{epoch}.csv')
        X = np.array(list(coor_feat_mask['X']),dtype=int)
        Y = np.array(list(coor_feat_mask['Y']),dtype=int)
        # node_feat_mask = [-np.log(1/(x)-1) for x in list(coor_feat_mask['prob'])]
        # node_feat_mask = np.array(node_feat_mask,dtype=float)
        node_feat_mask = 1-np.array(list(coor_feat_mask['prob']),dtype=float)
        node_feat_mask = (node_feat_mask-np.min(node_feat_mask))/(np.max(node_feat_mask)-np.min(node_feat_mask))
        patches_coords = [(int(X[i]/2**level),int(Y[i]/2**level)) for i in range(len(X))]
        labels, poly_coords = get_ground_truths(slide, patches_coords, level)
        visualize(node_feat_mask,wsi_name,patches_coords,poly_coords,img,level,epoch)