import os
import torch 

import numpy as np
import pandas as pd
from PIL import Image

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


def weight_vec(network):
    A = []
    for w in network.parameters():
        A.append(torch.flatten(w))
    return torch.cat(A)


def weight_dec_global(pyModel, weight_vec): 
    """
    Reshape the weight back to its original shape in pytorch and then 
    plug it to the model
    """
    c = 0
    for w in pyModel.parameters():
        m = w.numel()
        D = weight_vec[c:m+c].reshape(w.data.shape) 
        c+=m
        if w.data is None:
            w.data = D+0
        else:
            with torch.no_grad():
                w.set_( D+0 )
    return pyModel


def distribute_data(numOfClients, train_dataset, batch_size):
    """
    numOfClients: int 
    train_dataset: train_dataset (torchvision.datasets class)
    return distributed dataloaders for each client
    """
    # distribution list to fill the number of samples in each entry for each client
    distribution = []
    # rounding the number to get the number of dataset each client will get
    p = round(1/numOfClients * len(train_dataset))
    
    # the remainder data that won't be able to split if it's not an even number
    remainder_data = len(train_dataset) - numOfClients * p 
    # if the remainder data is 0 ---> all clients will get the same number of dataset
    if remainder_data == 0: 
        distribution = [p for i in range(numOfClients)]
    else:
        distribution = [p for i in range(numOfClients-1)]
        distribution.append(p+remainder_data)

    # splitting the data to different dataloaders
    data_split = torch.utils.data.random_split(train_dataset, distribution)
    # CLIENTS DATALOADERS
    ClIENTS_DATALOADERS = [torch.utils.data.DataLoader(data_split[i], batch_size=batch_size,shuffle=True, num_workers=32) for i in range(numOfClients)]
    
    print(f"Length of the training dataset: {len(train_dataset)} sample")
    return ClIENTS_DATALOADERS

def blood_noniid(numOfAgents, dataset, batch_size):
    """
    Fast non-IID partitioning for BloodMNIST.
    Uses label-based probability assignment but avoids loading images.
    """

    # Pre-allocate index lists for each client
    client_indices = [[] for _ in range(numOfAgents)]
    agents = np.arange(numOfAgents)

    # Extract dataset labels (no image loading)
    labels = dataset.labels.squeeze()  # (N,) array

    for idx, label in enumerate(labels):
        # Base probability distribution
        p = np.ones(numOfAgents)

        # Your label-based non-IID rules
        if label in [0, 1, 7]:
            p[0] = p[1] = p[2] = numOfAgents
        if label == 2:
            p[0] = p[3] = p[5] = numOfAgents
        if label == 3:
            p[0] = p[4] = p[5] = numOfAgents
        if label in [4, 5]:
            p[3] = p[4] = p[5] = numOfAgents
        if label == 6:
            p[4] = p[5] = numOfAgents

        # Normalize to sum to 1
        p = p / np.sum(p)

        # Sample which agent receives this sample
        chosen_agent = np.random.choice(agents, p=p)
        client_indices[chosen_agent].append(idx)

    # Dataset visualization (optional)
    dataset_vis = []
    for i in range(numOfAgents):
        dataset_vis.append(client_indices[i])

    # Create DataLoaders using Subset (lazy transforms, fast)
    dataset_loaders = [
        DataLoader(
            torch.utils.data.Subset(dataset, client_indices[i]),
            batch_size=batch_size,
            shuffle=True,
            num_workers=2  # safe number; increase only if CPU is strong
        )
        for i in range(numOfAgents)
    ]

    return client_indices, dataset_loaders, dataset_vis
