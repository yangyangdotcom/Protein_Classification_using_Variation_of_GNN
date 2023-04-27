import os
import torch
import numpy as np
import math
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader

# covid_data
processed_dir="/Users/benjamin/Documents/Dataset/Graphs/covid_600/data/processed/"
npy_file = "/Users/benjamin/Documents/Dataset/Graphs/covid_600/sample_details.npy"

# 8000 dataset
# processed_dir="/Users/benjamin/Documents/Dataset/Graphs/8000_dataset/data/processed/"
# # npy_file = "/Users/benjamin/Documents/Classification/Graph/Data/sample_details_all_balanced.npy"
# npy_file = "/Users/benjamin/Documents/Dataset/Graphs/8000_dataset/8000_sample_details.npy"

def label_representation(group):
    print(type(group))
    rep = []
    for i in group:
        # if i == "Isomerase":
        #     rep.append([0,0,1,0,0])
        # elif i == "Kinase":
        #     rep.append([0,0,0,1,0])
        # elif i == "Phosphatase":
        #     rep.append([0,0,0,0,1])
        # elif i == "Protease":
        #     rep.append([1,0,0,0,0])
        # elif i == "Receptor":
        #     rep.append([0,1,0,0,0])
        if i == "Spike":
            rep.append(torch.tensor(0))
        elif i == "Heavy":
            rep.append(torch.tensor(1))
        elif i == "Light":
            rep.append(torch.tensor(2))
    return np.array(rep)

def encode_onehot(labels):
    print(labels)
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    print(labels_onehot)
    exit()
    return labels_onehot

class LabelledDataset(Dataset):
    def __init__(self, npy_file, processed_dir):
        self.npy_ar = np.load(npy_file, allow_pickle=True)
        self.processed_dir = processed_dir
        self.protein = self.npy_ar[:,1]
        # self.label = encode_onehot(self.npy_ar[:,2])
        # self.label = label_representation(self.npy_ar[:,2])
        self.label = self.npy_ar[:,3]

        self.n_samples = self.npy_ar.shape[0]

    def __len__(self):
        return(self.n_samples)

    def __getitem__(self, index):
        prot = os.path.join(self.processed_dir, self.protein[index] + ".pt")
        prot = torch.load(prot)
        return prot, torch.tensor(self.label[index])

dataset = LabelledDataset(processed_dir=processed_dir, npy_file=npy_file)
size = dataset.__len__()
print(size)

seed = 42
torch.manual_seed(seed)

trainset,testset = torch.utils.data.random_split(dataset, [math.floor(0.8 * size), size - math.floor(0.8 * size)])

trainloader = DataLoader(dataset = trainset, batch_size = 32, num_workers = 0)
testloader = DataLoader(dataset = testset, batch_size = 32, num_workers = 0)
explainloader = DataLoader(dataset = dataset, batch_size = 1, num_workers = 0)
