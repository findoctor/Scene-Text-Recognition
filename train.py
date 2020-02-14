import os.path
import argparse
import random
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import os
from dataset import lmdbDataset, resizeNormalize, randomSequentialSampler, alignCollate
import utils.config as cfg
from utils.convert import LabelConverter
from model.CRNN import CRNN

Resume_flag = False
check_point_path = ''

def resumeModel(crnn, optimizer):
    checkpoint = torch.load(check_point_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    #model.train()

# First Load dataset from lmdb file.
train_dataset = lmdbDataset(root=cfg.lmdb_train_path)
assert train_dataset
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=cfg.batch_size,
    shuffle=True, sampler=None,
    num_workers=4,
    collate_fn=alignCollate(imgH=cfg.imgH, imgW=cfg.imgW))
# Usage: train_iter = iter(train_loader)
#        images, texts = train_iter.next()

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.xavier_uniform(m.weight)
        #m.bias.data.fill_(0.01)
    if classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
# Define CRNN, Converter, Loss, Optimizer
crnn = CRNN(cfg.imgH, cfg.imgW, cfg.n_hidden)
crnn.apply(init_weights)

converter = LabelConverter(cfg.alphabet)
# len(cfg.alphabet)-1
criterion = torch.nn.CTCLoss( blank=len(cfg.alphabet)-1, reduction='mean' )
optimizer = optim.Adam(crnn.parameters(), lr=cfg.lr , betas=(cfg.beta1, 0.999))

# Define images, texts as Variable used in Pytorch
images = torch.FloatTensor(cfg.batch_size, cfg.n_channel, cfg.imgH, cfg.imgW)
# texts = torch.IntTensor(cfg.batch_size * 5)
images = Variable(images)
# texts = Variable(texts)

# Train batch for n_epochs
for epoch in range(cfg.n_epoch):
    # Resume the model
    if Resume_flag:
        resumeModel(crnn, optimizer)
    running_loss = 0.0
    print("lr = "+ str(cfg.lr))
    for i, data in enumerate(train_loader, 0):
        for p in crnn.parameters():
            p.requires_grad = True
        crnn.train()

        inputs, labels = data
        labels = [s[2:-1] for s in labels]

        cfg.load_data(images, inputs)

        targets, target_length = converter.encode(labels)
        n_target = targets.numel()
        texts = torch.IntTensor(n_target)
        texts = Variable(texts)
        cfg.load_data(texts, targets)

        target_length = Variable(target_length)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = crnn(images)

        input_lengths = torch.full((cfg.batch_size,), outputs.shape[0] , dtype=torch.long)
        loss = criterion(outputs, targets, input_lengths, target_length)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        # we have 2000 training data in one batch, 100 per batch
        if i % 5 == 4:    # print every 5 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 5))
            running_loss = 0.0
    # Save the model
    torch.save({'epoch': epoch+1,\
                'model_state_dict': crnn.state_dict(),\
                'optimizer_state_dict': optimizer.state_dict(),\
                'loss': loss,
                }, os.path.join('checkpoints', '_epoch_{}.pth'.format(epoch+1)))
    check_point_path = os.path.join('checkpoints', '_epoch_{}.pth'.format(epoch+1))

print('Finished Training')