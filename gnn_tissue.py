import numpy as np
import os
import torch
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.utils.data import DataLoader
import pandas as pd
from torch_geometric.data import InMemoryDataset, Data
from sklearn.model_selection import train_test_split
import torch_geometric.transforms as T
import matplotlib.pyplot as plt
from data import GenomicsDataset, genomics_collate_fn
from torchvision import datasets, models, transforms
from torchsummary import summary
import torch.nn.functional as F
from torch import optim
from scipy.stats import pearsonr,spearmanr,kendalltau
from sklearn.metrics import r2_score
import csv

def mkdir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def compute_metrics(total_real_counts, total_predicted_counts):
    #Compute pearson, spearman, kendalltau and r2
    metric_dict = {}
    i=0
    for type in genomics_data.columns_to_count:
        real_counts = total_real_counts[:,i]
        predicted_counts = total_predicted_counts[:,i]
        metric_dict[type] = [pearsonr(real_counts, predicted_counts)[0],
                             spearmanr(real_counts, predicted_counts)[0],
                             kendalltau(real_counts, predicted_counts)[0],
                             r2_score(real_counts, predicted_counts)]
        i+=1
    return metric_dict

def generate_report(report_name, metric_dict):
    output_file = os.path.join(output_dir, report_name+".csv")
    csv_file = open(output_file, "w")
    writer = csv.writer(csv_file, delimiter=',')
    writer.writerow(['type', 'pearson', 'spearman', 'kendalltau', 'r2'])
    for type in metric_dict:
        metrics = metric_dict[type]
        metrics = [str(metric) for metric in metrics]
        writer.writerow([type] + metrics)

class Net(torch.nn.Module):
    def __init__(self, num_gcn_layers=0):
        super(Net, self).__init__()

        self.resnet_model = models.resnet18(pretrained=True)
        for param in self.resnet_model.parameters():
            param.requires_grad = False #If want to freeze resnet weights

        num_resnet_features = 256
        self.resnet_model.fc = nn.Linear(self.resnet_model.fc.in_features, num_resnet_features)

        self.num_gcn_layers = num_gcn_layers
        if(num_gcn_layers>0):
            n_graph_features = 64
            self.gconv1 = GCNConv(num_resnet_features, n_graph_features, add_self_loops=False)
        else:
            n_graph_features = num_resnet_features

        # fully connected layer
        num_hidden_units = 32 #32 for gcn experiment, 128 for first experiment without GCN
        self.hidden_layer = nn.Linear(n_graph_features, num_hidden_units)
        self.final_layer = nn.Linear(num_hidden_units, num_output_features)

    def forward(self, images, edge_index, edge_weight):
        x = self.resnet_model(images)
        x = F.relu(x)
        if(self.num_gcn_layers>0):
            x = self.gconv1(x, edge_index=edge_index, edge_weight=edge_weight)
            x = x[0][None,:]
        x = F.relu(x)
        x = self.hidden_layer(x)
        x = F.relu(x)
        x = self.final_layer(x)
        return x


lr = 1e-5
epochs = 250
batch_size = 1
criterion = nn.MSELoss()
save_model_freq = 5
model_name="model.pt"
output_dir = "./models/cellular_composition_gcn"
mode = "test"
num_gcn_layers = 1

genomics_data = GenomicsDataset(images_dir=r"F:\Datasets\RA\dawood\data\split\train",
                                  counts_path=r"F:\Datasets\RA\dawood\data\location_cellularcounts.csv",
                                  columns_to_count="B-cells,CAFs,Endothelial,Epithelial,Myeloid,Plasma Cells,PVL,T-cells",
                                  radius_to_fetch_neighbors=10,
                                  num_gcn_layers=num_gcn_layers)

num_output_features = len(genomics_data.columns_to_count)
print("Number of classes to count ",num_output_features)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Net(num_gcn_layers=num_gcn_layers).to(device)

if(mode == "test"):
    batch_size = 1

if(num_gcn_layers==0):
    dataloader = DataLoader(genomics_data, batch_size=batch_size, shuffle=True)
else:
    dataloader = DataLoader(genomics_data, batch_size=batch_size, shuffle=True, collate_fn=genomics_collate_fn)

model.to(device)
print("Num params: ", sum(p.numel() for p in model.parameters()))
print("Num trainable params: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

model_path = os.path.join(output_dir, model_name)

if(os.path.exists(model_path)):
    model.load_state_dict(torch.load(model_path))

if(mode == "train"):
    mkdir(output_dir)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(0, epochs):
        epoch_loss = 0.0  # for one full epoch
        for (b_idx, batch) in enumerate(dataloader):
            _, images, counts, edge_index, edge_weight = batch
            edge_index = edge_index.long().cuda()
            edge_weight = edge_weight.cuda()
            images = images.cuda()
            counts = counts.cuda()
            optimizer.zero_grad()
            pred_counts = model(images.float(), edge_index, edge_weight)
            loss = criterion(pred_counts, counts.float())
            epoch_loss += loss.item()  # accumulate
            loss.backward()  # compute gradients
            optimizer.step() # update weights

        print("epoch number = %4d  |  loss = %0.4f" % (epoch, epoch_loss/batch_size))
        if(epoch%save_model_freq==0):
            print("Saving the model")
            torch.save(model.state_dict(), model_path)
else:

    if not os.path.exists(model_path):
        print("Please give model path as the following path doesn't exist ",model_path)
        exit(0)

    model.eval()
    real_counts = []
    pred_counts = []
    patient_real_counts_dict = {}
    patient_pred_counts_dict = {}

    for (b_idx, batch) in enumerate(dataloader):
        image_names, images, counts, edge_index, edge_weight = batch
        edge_index = edge_index.long().cuda()
        edge_weight = edge_weight.cuda()
        images = images.cuda()
        counts = counts.cuda()
        pred_counts_batch = model(images.float(), edge_index, edge_weight)

        real_counts_batch_np = counts.cpu().detach().numpy().tolist()
        pred_counts_batch_np = pred_counts_batch.cpu().detach().numpy().tolist()
        print(images)
        print(edge_index)
        print(edge_weight)
        print(pred_counts_batch_np)

        real_counts = real_counts + real_counts_batch_np
        pred_counts = pred_counts + pred_counts_batch_np

        #patient wise predictions
        patient_id = image_names[0].split("_")[0][0]
        if(patient_id in patient_real_counts_dict):
            patient_real_counts_dict[patient_id]+=real_counts_batch_np
            patient_pred_counts_dict[patient_id]+=pred_counts_batch_np
        else:
            patient_real_counts_dict[patient_id] = real_counts_batch_np
            patient_pred_counts_dict[patient_id] = pred_counts_batch_np


    real_counts = np.array(real_counts)
    pred_counts = np.array(pred_counts)

    #generate overall report
    overall_metrics = compute_metrics(real_counts, pred_counts)
    generate_report("overall_report", overall_metrics)

    #patient wise reports
    for patient_id in patient_real_counts_dict:
        patient_metrics = compute_metrics(np.array(patient_real_counts_dict[patient_id]), np.array(patient_pred_counts_dict[patient_id]))
        generate_report(patient_id, patient_metrics)

    print("Reports Generated")