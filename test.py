import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler

from model.DeepFM import DeepFM
from data.dataset import CriteoDataset

import pandas as pd
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = torch.device("cuda")

# 900000 items for training, 10000 items for valid, of all 1000000 items
Num_train = 9000
bs=100
continous_features = 13
test_data = pd.read_csv('data/test.txt',header=None,nrows=bs).values
print(test_data.shape)
# index of continous features are one
Xi_coutinous = np.zeros_like(test_data[:, :continous_features])
Xi_categorial = test_data[:, continous_features:]
# print(np.concatenate((Xi_coutinous, Xi_categorial)).astype(np.int32).shape)
Xi = torch.from_numpy(np.concatenate((Xi_coutinous, Xi_categorial), axis=1).astype(np.int32)).unsqueeze(-1).to(device=device,dtype=torch.long)
print(Xi.shape,Xi.dtype)

# value of categorial features are one (one hot features)
Xv_categorial = np.ones_like(test_data[:, continous_features:])
Xv_coutinous = test_data[:, :continous_features]

Xv = torch.from_numpy(np.concatenate((Xv_coutinous, Xv_categorial), axis=1).astype(np.int32)).to(device=device,dtype=torch.float)
print(Xv.shape,Xv.dtype)
# Num_test=100
# train_data = CriteoDataset('./data', train=False)
# loader_train = DataLoader(train_data, batch_size=100,
#                           sampler=sampler.SubsetRandomSampler(range(Num_test)))


feature_sizes = np.loadtxt('./data/feature_sizes.txt', delimiter=',')
feature_sizes = [int(x) for x in feature_sizes]
print(len(feature_sizes), feature_sizes)

model = DeepFM(feature_sizes, use_cuda=True).eval().to(device)
# set model to evaluation mode
# for xi, xv in loader_train :
#     print(type(xi), type(xv),xi.shape,xv.shape)
#     print(xi.dtype, xv.dtype)
#     xi = xi.to(device=device, dtype=torch.long)  # move to device, e.g. GPU
#     xv = xv.to(device=device, dtype=torch.float)
#     print(type(xi), type(xv),xi.shape,xv.shape)
#     print(xi.dtype, xv.dtype)
#     total = model(xi, xv)
#     preds = (torch.sigmoid(total) > 0.5)
#     print(preds.shape)

total = model(Xi, Xv)
preds = (torch.sigmoid(total) > 0.5)
print(preds.shape)
