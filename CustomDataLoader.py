# -*- coding: utf-8 -*-
"""
Created on Sun Jul 29 21:32:34 2018

@author: x_j_t
"""
import sys
#import cv2
import os
import numpy as np
import torch
from skimage import io
from torch.utils.data.dataset import Dataset
from torchvision import transforms
#from PIL import Image
#import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

class ContourDataset(Dataset):
    def __init__(self, paths, transform=None):
        for path in paths:
            if not os.path.exists(path):
                print("input path {} not found, quit on error".format(path))
                sys.exit(1)
        self.transform=transform
        self.default_transform=transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=0, translate=(0.05,0.025)),
            transforms.ToTensor()
        ])
        self.paths=paths
        ##assuming paths=[frontal, side, betas]
        self.frontImgs=[]
        self.sideImgs=[]
        self.betaFiles=[]
        for item in os.listdir(paths[0]):
            if not item.startswith(".") and os.path.isfile(os.path.join(paths[0],item)):
                self.frontImgs.append(item)
        for item in os.listdir(paths[1]):
            if not item.startswith(".") and os.path.isfile(os.path.join(paths[1],item)):
                self.sideImgs.append(item)
        for item in os.listdir(paths[2]):
            if not item.startswith(".") and os.path.isfile(os.path.join(paths[2],item)):
                self.betaFiles.append(item)
        
                       
    
    def __getitem__(self, index):
        frontImgName=self.frontImgs[index]
        nameParts=frontImgName.split('.')
        fileName=nameParts[-2]
        ext='.'+nameParts[-1]
        sideImgName=fileName+ext
        betaFileName=fileName+'.txt'
        print("filenames: {}, {}, {}".format(frontImgName,sideImgName,betaFileName))
        betas=np.loadtxt(os.path.join(self.paths[2],betaFileName),dtype=float,delimiter="\t")
        frontImg=io.imread(os.path.join(self.paths[0],frontImgName))
        sideImg =io.imread(os.path.join(self.paths[1],sideImgName))
        frontImg=frontImg[0:264,0:192,:]
        sideImg=sideImg[0:264,0:192,:]
        if self.transform is not None:
            frontTensor=self.transform(frontImg)
            sideTensor =self.transform(sideImg)
        else:
            frontTensor=self.default_transform(frontImg)
            sideTensor =self.default_transform(sideImg)
        imgTensor = torch.cat((frontTensor, sideTensor), 0)
        betaTensor=torch.from_numpy(betas)
        return imgTensor, betaTensor
    
    def __len__(self):
        return len(self.frontImgs)


#################################
## helper functions

"""
def im_show(name, image, resize=1):
    H,W = image.shape[0:2]
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, image.astype(np.uint8))
    cv2.resizeWindow(name, round(resize*W), round(resize*H))


def tensor_to_img(img, mean=0, std=1):
    #img = np.transpose(img.numpy(), (1, 2, 0))
    img=img.numpy()
    img = (img*std+ mean)*255
    img = img.astype(np.uint8)
    #img = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
    return img
"""
#################################        

if __name__=='__main__':
    frontPath="../../data/3Dmesh/FrontView"
    sidePath="../../data/3Dmesh/SideView"
    betaPath="../../data/3Dmesh/Betas"
    paths=[frontPath, sidePath, betaPath]
    dataset=ContourDataset(paths)
    loader=DataLoader(dataset, batch_size=4)
    for epoch in range(1):
        print("epoch = {}\n".format(epoch))
        for i, data in enumerate(loader,0):
            print("i={}\n".format(i))
            imageTensor, labels=data
            print(imageTensor.size(),labels.size())
            """
            image=imageTensor[0,1,:,:]
            print("shape of image tensor is {}".format(image.size()))
            im_show('image', tensor_to_img(image), resize=2 )
            cv2.waitKey(0)
            """
 