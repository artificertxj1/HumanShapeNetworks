# -*- coding: utf-8 -*-
"""
Created on Sat Jul 28 19:25:02 2018

@author: x_j_t
"""

import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from HSN_model import HSnet
from CustomDataLoader import ContourDataset



def main():
    ## device configuration
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    ## hyper parameters
    num_epochs=5
    num_classes=20
    batch_size=2
    learning_rate=0.0001
    ## model loss and optimizer
    model=HSnet(num_classes).to(device)
    L2Loss=nn.MSELoss()
    optimizer=torch.optim.Adam(model.parameters(), lr=learning_rate)
    # Train the model
    frontPath="../../data/3Dmesh/frontview"
    sidePath="../../data/3Dmesh/sideview"
    betaPath="../../data/3Dmesh/betas"
    paths=[frontPath, sidePath, betaPath]
    dataset=ContourDataset(paths)
    train_loader=DataLoader(dataset, batch_size=batch_size)
    total_step=len(train_loader)
    print(total_step)
    
    for epoch in range(num_epochs):
        for i, data in enumerate(train_loader):
            images, labels=data
            images=images.to(device)
            #print(images.size())
            outputs=model(images)
            loss=L2Loss(outputs,labels.float())
            #print("Loss is: {}".format(loss.item()))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i+1) % 100 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
    torch.save(model.state_dict(), 'model.ckpt')

if __name__=="__main__":
    main()
