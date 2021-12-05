import random

import numpy as np
import torch
from icecream import ic
from torchsummary import summary
from tqdm import tqdm

from agent.network import Imitator
from config import NETWORK_CONFIG
from train_utils.data_set import DataSet
from utils import plotSparseMatrix


def showFeatures(features):
    for i in range(features.shape[0]):
        plotSparseMatrix(features[i], "none")


# FIX seed
np.random.seed(0)
random.seed(0)
torch.manual_seed(0)

imitator = Imitator()
summary(imitator.net, (NETWORK_CONFIG.in_channels, 10, 20), batch_size=512)

data_set = DataSet()
data_set.load("dataset/data_Top1.pkl")
train_iter = data_set.trainIter()

imitator.predict(data_set.data_buffer[0][0])

with tqdm(total=len(train_iter)) as pbar:
    for data_batch in train_iter:
        loss, acc = imitator.trainStep(data_batch)
        pbar.update()
        ic(loss, acc)
