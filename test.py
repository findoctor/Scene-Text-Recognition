from __future__ import print_function
from __future__ import division

import argparse
import random
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import numpy as np
#from warpctc_pytorch import CTCLoss
import os
#import dataset
from dataset import lmdbDataset, resizeNormalize, randomSequentialSampler, alignCollate
import create_dataset

# Create dataset first
data_path = 'Data\IIIT5K\train'

train_dataset = lmdbDataset(root=data_path)  # root to dataset
assert train_dataset

sampler = randomSequentialSampler(train_dataset, 100)  # batch size = 100

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=opt.batchSize,
    shuffle=True, sampler=sampler,
    num_workers=int(opt.workers),
    collate_fn=dataset.alignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio=opt.keep_ratio))
test_dataset = dataset.lmdbDataset(
    root=opt.valroot, transform=dataset.resizeNormalize((100, 32)))