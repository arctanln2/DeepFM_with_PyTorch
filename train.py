import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler

import os
from model.DeepFM import DeepFM
from data.dataset import CriteoDataset
from utils.dataPreprocess import *
from time import time
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-bs', type=int, default=128,
                        help='batchsize')
    parser.add_argument('-gpuid', type=str, default="0",
                        help='warm up training phase')
    parser.add_argument('-logpath', type=str, default='logs/test',
                        help="Log the test result.")
    
    args = parser.parse_args()
    dirpath = args.logpath[:args.logpath.rfind('/')]
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuid
    # 900000 items for training, 10000 items for valid, of all 1000000 items
    Num_train = 90000
    Num_test = 10000

    # data preprocess
    # begin=time()
    # preprocess('./data/raw', './data',num_train_sample=Num_train,num_test_sample=Num_test)
    # end=time()
    # print(f"cost {end-begin} s")
    # load data
    train_data = CriteoDataset('./data', train=True)
    loader_train = DataLoader(train_data, batch_size=args.bs,
                            sampler=sampler.SubsetRandomSampler(range(int(Num_train*0.9))))
    val_data = CriteoDataset('./data', train=True)
    loader_val = DataLoader(val_data, batch_size=args.bs,
                            sampler=sampler.SubsetRandomSampler(range(int(Num_train*0.9), Num_train)))

    feature_sizes = np.loadtxt('./data/feature_sizes.txt', delimiter=',')
    feature_sizes = [int(x) for x in feature_sizes]
    print(feature_sizes)


    sys.stdout = open(args.logpath, 'w')
    model = DeepFM(feature_sizes, use_cuda=True)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.0)
    model.fit(loader_train, loader_val, optimizer, epochs=1000, verbose=True, print_every=1)

    checkpoint_path = "checkpoint"
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    weights_path = os.path.join(checkpoint_path, f'deepfm.pth')
    model.eval()
    torch.save(model.state_dict(), weights_path)
