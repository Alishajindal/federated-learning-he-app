import sys
import os
import glob

import numpy as np 

from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.utils.data as data
import torch 

import medmnist
from medmnist import INFO

from utils import blood_noniid, distribute_data

import random 

seed = 105
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)


def distribute_images(dataset_name,train_data, num_clients, test_data, batch_size, num_workers = 8):
    """
    This method splits the dataset among clients.
    train_data: train dataset 
    test_data: test dataset 
    batch_size: batch size

    """

    if dataset_name == 'bloodmnist':
        _, testloader, train_dataset, _ = bloodmnist(batch_size= batch_size)
        _, CLIENTS_DATALOADERS, _ = blood_noniid(num_clients, train_dataset, batch_size =batch_size)
        
    return CLIENTS_DATALOADERS, testloader

def bloodmnist(input_size =224, batch_size = 32, num_workers= 8, download = True):
    """
        Get train/test loaders and sets for bloodmnist from medmnist library. 

        Input: 
            input_size (int): width of the input image which issimilar to height 
            batch_size (int)
            num_workers (int): Num of workeres used for in creating the loaders 
            download (bool): Whether to download the dataset or not
        
        return: 
            train_loader, test_loader, train_dataset, test_dataset
    """

    data_flag = 'bloodmnist'
    info = INFO[data_flag]
    DataClass = getattr(medmnist, info['python_class'])

    data_transform_train = transforms.Compose([
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees= 10, translate=(0.1,0.1)),
        transforms.RandomResizedCrop(input_size, (0.75,1), (0.9,1)), 
        transforms.ToTensor(),
        ]) 
    
    data_transform_teest = transforms.Compose([
        transforms.Resize(224), 
        transforms.ToTensor(),
        ])
    
    train_dataset = DataClass(split='train', transform=data_transform_train, download=download)
    test_dataset = DataClass(split='test', transform=data_transform_teest, download=download)

    train_loader = data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=2*batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, test_loader, train_dataset, test_dataset
