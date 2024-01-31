import glob
import xml.dom.minidom
import cv2
import os
import numpy as np
import torch
# import multiresolutionimageinterface as mir

def convert():
    CAMELYON_ANNOTATION_DIR = '/data1/WSI/Camelyon16/lesion_annotation'
    all_slides = glob.glob('/data1/WSI/Camelyon16/lesion_annotation/*.xml')
    shape_dict = {}
    with open('/home/r20user8/Documents/HDPMIL/Localization/image_shape.txt', 'r') as f:
        data = f.readlines()
    for item in data:
        item = item.split(',')
        shape_dict[item[0].split('.')[0]]=(int(item[1]),int(item[2]))
    for slide_id in all_slides:
        slide_id = slide_id.split('/')[-1].split('.')[0]
        annotation_file = os.path.join(CAMELYON_ANNOTATION_DIR, f"{slide_id}.xml")
        w, h = shape_dict[slide_id]
        # FIXME: change this later
        # assume we use 20x, 256 for camelyon16 dataset
        # Following camelyon16 github, we evaluate the model on level 5
        level = 5
        img_anno = np.zeros(
            (h // (2 ** level), w // (2 ** level)), np.uint8)
        if os.path.exists(annotation_file):
            polygons = get_coordinates(annotation_file)
            for polygon in polygons:
                polygon = polygon // (2**level)
                cv2.fillConvexPoly(img_anno, polygon, 255)
            # cv2.imwrite(os.path.join("./", slide_id + '.jpg'), img_anno)
            img_anno = img_anno.reshape(
                (h // (2**level), w // (2**level)))
            img_anno = img_anno // 255

            np.save(f'{CAMELYON_ANNOTATION_DIR}/{slide_id}.npy',img_anno)
        # img_anno = torch.tensor(img_anno)

    return

def get_coordinates(annotation_file):
    DOMTree = xml.dom.minidom.parse(annotation_file)
    collection = DOMTree.documentElement
    coordinatess = collection.getElementsByTagName("Coordinates")
    polygons = []
    for coordinates in coordinatess:
        coordinate = coordinates.getElementsByTagName("Coordinate")
        poly_coordinates = []
        for point in coordinate:
            x = point.getAttribute("X")
            y = point.getAttribute("Y")
            poly_coordinates.append([float(x), float(y)])
        polygons.append(np.array(poly_coordinates, dtype=int))
    return polygons

def get_coordinates1(annotation_file):
    DOMTree = xml.dom.minidom.parse(annotation_file)
    collection = DOMTree.documentElement
    coordinatess = collection.getElementsByTagName("Coordinates")
    annotationss = collection.getElementsByTagName("Annotation")
    polygons = []
    for coordinates,annotations in zip(coordinatess,annotationss):
        if annotations.getAttribute('PartOfGroup') in ['_0','_1','_2','Tumor']:
            coordinate = coordinates.getElementsByTagName("Coordinate")
            poly_coordinates = []
            for point in coordinate:
                x = point.getAttribute("X")
                y = point.getAttribute("Y")
                poly_coordinates.append([float(x), float(y)])
            polygons.append(np.array(poly_coordinates, dtype=int))
    return polygons
import multiresolutionimageinterface as mir
# def convert1():
#     annotation_lists = glob.glob('/data1/WSI/Camelyon16/lesion_annotation/*.xml')
#     img_dir = '/data1/WSI/Camelyon16/CAMELYON16/testing/images'
#     for annotation_list in annotation_lists:
#         slide_name = annotation_list.split('/')[-1].split('.')[0]
#         reader = mir.MultiResolutionImageReader()
#         mr_image = reader.open(os.path.join(img_dir,slide_name+'.tif'))
#         assert mr_image is not None
#         annotation_list = mir.AnnotationList()
#         xml_repository = mir.XmlRepository(annotation_list)
#         xml_repository.setSource(xml_file)
#         xml_repository.load()
#         annotation_mask = mir.AnnotationToMask()
#         camelyon17_type_mask = False
#         monitor=mir.CmdLineProgressMonitor()
#         annotation_mask.setProgressMonitor(monitor)
#         label_map = {'metastases': 255, 'normal': 0} if camelyon17_type_mask else {'_0': 255, '_1': 255, 'Tumor':255, '_2': 0, 'None':0,'Exclusion':0}
#         conversion_order = ['metastases', 'normal'] if camelyon17_type_mask else  ['_0', '_1', 'Tumor','_2','None','Exclusion']
#         annotation_mask.convert(annotation_list, output_path, mr_image.getDimensions(), mr_image.getSpacing(), label_map, conversion_order)

convert()