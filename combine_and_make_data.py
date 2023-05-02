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

def mkdirs(dirs):
    if not os.path.exists(dirs):
        os.makedirs(dirs)

gene_expression_path = r'F:\Datasets\RA\dawood\data\ST-cnts' #gene expressions
cell_counts_path = r'F:\Datasets\RA\dawood\major' #cellular composition dir
img_path = r'F:\Datasets\RA\dawood\data\ST-imgs' #images
spot_path = r'F:\Datasets\RA\dawood\data\ST-spotfiles' #location information
gids = np.array(['FASN','GNAS','ACTG1','DDX5','XBP1']) #Selected Genes
extracted_patches_dir = r"F:\Datasets\RA\dawood\data\patches"

LG_Combined=pd.DataFrame()
LC_Combined=pd.DataFrame()

img_files = Path(img_path)

for imgf in img_files.glob('**/*.jpg'):

    print('Processing csv file ',imgf.stem)
    sptf = os.path.join(spot_path, imgf.parent.name+'_selection.tsv')
    gexf = os.path.join(gene_expression_path, imgf.parent.name+'.tsv.gz')
    ccounts = os.path.join(cell_counts_path, imgf.parent.name+'-proportion.tsv')

    # Reading locations file
    L = pd.read_csv(sptf,sep='\t')
    L['id'] = L['x'].astype('str')+'x'+L['y'].astype('str')
    L.set_index('id',inplace=True)

    # Reading gene expressions file
    with gzip.open(gexf) as f:
        G = pd.read_csv(f,sep='\t')
        G.rename( columns={'Unnamed: 0':'id'}, inplace=True )
        G.set_index('id',inplace=True)
        G = G[gids]

    # Reading cellular composition file
    C = pd.read_csv(ccounts, sep='\t')
    C.rename(columns={'Unnamed: 0': 'id'}, inplace=True)
    C.set_index('id', inplace=True)

    LG = L.join(G, how='inner')
    # LG['id']=LG.index.astype(str)
    LG['img']=imgf.parent.name+'_'+imgf.stem
    LG.set_index(imgf.parent.name+'_'+imgf.stem+'_'+LG.index.astype(str), inplace=True)
    LG_Combined=LG_Combined.append(LG)

    LC = L.join(C, how='inner')
    # LC['id'] = LC.index.astype(str)
    LC['img'] = imgf.parent.name + '_' + imgf.stem
    LC.set_index(imgf.parent.name + '_' + imgf.stem + '_' + LC.index.astype(str), inplace=True)
    LC_Combined = LC_Combined.append(LC)


LG_Combined.to_csv("location_geneexpressions.csv")
LC_Combined.to_csv("location_cellularcounts.csv")

# Process images, extract patches and save into directory

mkdirs(extracted_patches_dir)

win = 256
hw = win//2

for imgf in img_files.glob('**/*.jpg'):
    fname = imgf.parent.name+'_'+imgf.stem
    print("Processing image file ",fname)
    LC_Combined_current = LC_Combined[LC_Combined['img']==fname]
    if LC_Combined_current.shape[0]==0:
        continue
    I = imread(imgf)
    for idx, row in LC_Combined_current.iterrows():
        patch_name = fname + "_" + str(row['x']) + 'x' + str(row['y'])+".png"
        x,y =  int(row['pixel_x']), int(row['pixel_y'])
        if (y-hw)<0 or (y+hw)>I.shape[0] or (x-hw)<0 or (x+hw)>I.shape[1]: continue
        p = I[y-hw:y+hw,x-hw:x+hw]
        imsave(os.path.join(extracted_patches_dir,patch_name), p)