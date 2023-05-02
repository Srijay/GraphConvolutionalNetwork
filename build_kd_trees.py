import numpy as np
from pathlib import Path
import gzip
from skimage.io import imread, imsave
import matplotlib.pyplot as plt
import pandas as pd
import os
from tqdm import tqdm
import pickle
from PIL import Image
import numpy as np
from sklearn.neighbors import KDTree

def mkdirs(dirs):
    if not os.path.exists(dirs):
        os.makedirs(dirs)

preprocessed_locations_path = r'F:\Datasets\RA\dawood\data\location_cellularcounts.csv' #images
L = pd.read_csv(preprocessed_locations_path,sep=',')

kdtreekey_to_imagename = {}
slidename_to_coordinatelist = {}

L = L[['id','pixel_x','pixel_y']]

for index, row in L.iterrows():

    image_name = row['id']
    coordinates = [float(row['pixel_x']), float(row['pixel_y'])]
    slide_name = "_".join(image_name.split("_")[:-1])
    if(slide_name in slidename_to_coordinatelist):
        slidename_to_coordinatelist[slide_name].append(coordinates)
    else:
        slidename_to_coordinatelist[slide_name] = [coordinates]
    kdtreekey = str(int(float(row['pixel_x'])*100))+"_"+str(int(float(row['pixel_y'])*100))
    kdtreekey_to_imagename[kdtreekey] = image_name

slidename_to_kdtree = {}
for slide_name in slidename_to_coordinatelist:
    coordinateslist = slidename_to_coordinatelist[slide_name]
    coordinateslist_np = np.array(coordinateslist)
    kdtree = KDTree(coordinateslist_np)
    slidename_to_kdtree[slide_name] = kdtree

print(slidename_to_kdtree)