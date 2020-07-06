from __future__ import division
from __future__ import print_function

import os
import glob
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from utils import load_data, accuracy

class FGCN(object):
    def __init__(self, adj, in_c, out_c, hid_c=5): # [3,5,2]
        self.W1 = nn.Parameter(torch.randn(in_c, hid_c) ) # [3, 5]
        self.W2 = nn.Parameter(torch.randn(hid_c, out_c) ) # [5, 2]
        self.adj = adj # [4, 4]

    def foward(self, H1):
        A1 = self.adj.mm(H1) # [4,3]
        Z1 = A1.mm(self.W1) # [4,5]
        self.mask = (Z1>0)
        H2 = torch.where(self.mask>0, Z1, torch.zeros_like(Z1) ) # [4,5]
        # H2 = F.relu(A1.mm(self.W1) ) # [4,5]
        A2 = self.adj.mm(H2) # [4,5]
        out = A2.mm(self.W2) # [4,2]
        return out, A1, A2

    def gradient(self, pred, target, A1, A2):
        # pred = F.softmax(out, dim=0) # [4,2]
        dy = (pred - target) / pred.shape[0]
        dW2 = A2.t().mm(dy) # [4,5].T * [4,2] -> [5,2]
        dH2 = adj.mm(dy).mm(self.W2.t()) # [5,2].T [2,4].T [4,4].T
        dW1 = A1.t().mm(torch.where(self.mask, dH2, torch.zeros_like(dH2) ) )
        return dW1, dW2

    def update(self, dW1, dW2, lr):
        self.W1 = self.W1 - lr * dW1
        self.W2 = self.W2 - lr * dW2

adj, features, labels, idx_train, idx_val, idx_test = load_data()
# N, F1, F2 = 10, 10, 4
# H1 = torch.randn(N,F1)
# adj = torch.rand(N,N)
# adj = torch.where(adj>0, adj, torch.zeros_like(adj) )
N, in_c = features.shape
cls_ = int(labels.max()) + 1 
model = FGCN(adj, in_c, cls_)

labels = torch.zeros(N, cls_).scatter_(1, torch.LongTensor(labels).unsqueeze(1), 1)

for epoch in range(1000):
    out, A1, A2 = model.foward(features)
    pred, target = torch.zeros_like(out), torch.zeros_like(labels)
    pred[idx_train] = F.softmax(out[idx_train], dim=1)
    target[idx_train] = labels[idx_train]
    
    dW1, dW2 = model.gradient(pred, target, A1, A2)
    model.update(dW1, dW2, 100)
    if epoch % 100 == 99:
        loss = F.cross_entropy(pred[idx_train], target[idx_train].argmax(dim=1).long() )
        print("Epoch: {:03d}, train_loss: {:.4f}".format(epoch, loss.item() ) )

        out, A1, A2 = model.foward(features)

        pred_val = out[idx_val].argmax(dim=1)
        target_val = labels[idx_val].argmax(dim=1)
        acc_val = pred_val.eq(target_val).sum().item() / pred_val.shape[0]
        print("Val acc:", acc_val)

        pred_test = out[idx_test].argmax(dim=1)
        target_test = labels[idx_test].argmax(dim=1)
        acc_test = pred_test.eq(target_test).sum().item() / pred_test.shape[0]
        print("Test acc:", acc_test)

# out = torch.rand([8,6],requires_grad=True)
# target = torch.zeros_like(out)
# target[:,3] = 1
# print(F.softmax(out,dim=1)-target)
# target = target.argmax(dim=1).long()
# loss = F.cross_entropy(out, target)
# loss.backward()

# print(out.grad*out.shape[0])
    






