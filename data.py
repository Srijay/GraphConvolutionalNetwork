import glob
import os
import sys
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as T
from scipy.spatial import distance
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import pandas as pd
from sklearn.neighbors import KDTree

sys.path.insert(0,os.getcwd())

class GenomicsDataset(Dataset):

    def __init__(self, images_dir, counts_path, columns_to_count, radius_to_fetch_neighbors, num_gcn_layers=0):
        super(Dataset, self).__init__()

        self.images_dir = images_dir
        self.image_names = os.listdir(images_dir)
        self.counts_df = pd.read_csv(counts_path)
        self.num_gcn_layers = num_gcn_layers

        self.kdtreekey_to_imagename = {}
        self.imagename_to_kdtreekey = {}
        slidename_to_coordinatelist = {}
        for index, row in self.counts_df.iterrows():
            image_name = row['id']
            coordinates = [float(row['pixel_x']), float(row['pixel_y'])]
            slide_name = "_".join(image_name.split("_")[:-1])
            if (slide_name in slidename_to_coordinatelist):
                slidename_to_coordinatelist[slide_name].append(coordinates)
            else:
                slidename_to_coordinatelist[slide_name] = [coordinates]
            kdtreekey = str(int(float(row['pixel_x']) * 100)) + "_" + str(int(float(row['pixel_y']) * 100))
            self.kdtreekey_to_imagename[kdtreekey] = image_name
            self.imagename_to_kdtreekey[image_name] = kdtreekey

        if(self.num_gcn_layers > 0):
            self.slidename_to_kdtree = {}
            for slide_name in slidename_to_coordinatelist:
                coordinateslist = slidename_to_coordinatelist[slide_name]
                coordinateslist_np = np.array(coordinateslist)
                kdtree = KDTree(coordinateslist_np)
                self.slidename_to_kdtree[slide_name] = kdtree

        self.columns_to_count = columns_to_count.split(",")
        self.radius_to_fetch_neighbors = radius_to_fetch_neighbors

    def read_image(self,img_path):
        img = Image.open(img_path)
        img_np = np.asarray(img)/255.0
        return img_np

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):

        image_name = self.image_names[index]

        image_path = os.path.join(self.images_dir,image_name)
        transform = T.Compose([T.ToTensor()])
        image = self.read_image(image_path)
        image = transform(image)

        image_name = image_name.split(".")[0]
        counts = self.counts_df.loc[self.counts_df['id'] == image_name]
        counts = counts[self.columns_to_count]
        counts=np.array(counts.values.flatten().tolist())

        if (self.num_gcn_layers > 0):
            #construct graph
            slide_name = "_".join(image_name.split("_")[:-1])
            kdtree = self.slidename_to_kdtree[slide_name]
            kdtreekey = self.imagename_to_kdtreekey[image_name]
            x_loc, y_loc = kdtreekey.split("_")
            x_loc, y_loc = int(x_loc)/100, int(y_loc)/100
            locations = kdtree.get_arrays()[0]
            neighbors_indices, neighbors_distances = kdtree.query_radius([[x_loc,y_loc]], r=400, return_distance=True)
            neighbors_indices = neighbors_indices[0]
            neighbors_distances = neighbors_distances[0]

            edge_index_source = []
            edge_index_target = []
            edge_weight = []
            images = [image]
            neighbor_index = 0
            for kindex in range(0,len(neighbors_indices)):
                if(neighbors_distances[kindex] == 0):
                    continue
                x,y = locations[neighbors_indices[kindex]]
                neighbor_kdtreekey = str(int(x*100)) + "_" + str(int(y*100))
                neighbor_image_name = self.kdtreekey_to_imagename[neighbor_kdtreekey]+".png"
                neighbor_image_path = os.path.join(self.images_dir, neighbor_image_name)
                neighbor_image = self.read_image(neighbor_image_path)
                neighbor_image = transform(neighbor_image)
                images.append(neighbor_image)
                edge_weight.append(neighbors_distances[kindex])
                neighbor_index+=1
                edge_index_source.append(neighbor_index)
                edge_index_target.append(0)

            images = torch.stack(images, dim=0)
            edge_index = torch.Tensor([edge_index_source,edge_index_target])
            edge_weight = torch.Tensor(np.array(edge_weight))
        else:
            images = image
            edge_index = []
            edge_weight = []

        return image_name, images, counts, edge_index, edge_weight

def genomics_collate_fn(batch):

    image_name_l = []
    images_l = []
    counts_l = []
    edge_index_l = []
    edge_weight_l = []

    for i, (image_name, images, counts, edge_index, edge_weight) in enumerate(batch):
        image_name_l.append(image_name)
        images_l.append(images)
        counts_l.append(counts)
        edge_index_l.append(edge_index)
        edge_weight_l.append(edge_weight)

    images_l = torch.cat(images_l)
    counts_l = torch.Tensor(np.array(counts_l))
    edge_index_l = torch.cat(edge_index_l)
    edge_weight_l = torch.cat(edge_weight_l)

    out = (image_name_l, images_l, counts_l, edge_index_l, edge_weight_l)

    return out

# dataset = GenomicsDataset(images_dir=r"F:\Datasets\RA\dawood\data\patches",
#                           counts_path=r"F:\Datasets\RA\dawood\data\location_cellularcounts.csv",
#                           columns_to_count="B-cells,CAFs,Epithelial",
#                           radius_to_fetch_neighbors=10)
#
# image_name, image, counts, edge_index = dataset[0]