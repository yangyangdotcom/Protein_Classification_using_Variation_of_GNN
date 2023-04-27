import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from tqdm import tqdm
import math
from torch.optim.lr_scheduler import MultiStepLR

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

import torch.nn as nn
from data_prepare import dataset, trainloader, testloader
from model import GCNN, AttGNN

def train(model, device, trainloader, optimizer, epoch):
    print(f'Training on {len(trainloader)} samples.....')
    model.train()
    loss_func = nn.CrossEntropyLoss()
    predictions = torch.Tensor()
    labels = torch.Tensor()
    # scheduler = MultiStepLR(optimizer, milestones=[1,5], gamma=0.5)
    correct = 0
    for count, (prot, label) in enumerate(trainloader):
        prot = prot.to(device)
        print(prot)
        print(prot.size())
        # exit()
        output = model(prot)
        predictions = torch.cat((predictions,output.cpu()), 0)
        labels = torch.cat((labels, label.cpu()), 0)
        print("-==-=-=-=output-=-=-===-=")
        print(output)
        print("-==-=-=-=label-=-=-===-=")
        print(label)
        # exit()
        loss = loss_func(output, label.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # scheduler.step()
    labels = labels.detach().numpy()
    # predictions = predictions.detach().numpy()
    predictions = torch.argmax(predictions, dim = 1)
    print("-==-=-=-=predictions-=-=-===-=")
    print(predictions)
    labels = torch.from_numpy(labels)
    print("-==-=-=-=labels-=-=-===-=")
    print(labels)
    # print(type(labels))
    # acc = accuracy(labels, predictions)
    correct += (predictions == labels).sum().item()
    acc = 100 * correct / len(predictions)
    print(f"Epoch: {epoch} Loss: {loss} Accuracy: {acc}")
    return loss, acc

def plot_trend(num_epochs, loss_list, acc_list):
    epochs = range(num_epochs)
    # print(type(loss_list))
    # print(loss_list)
    # loss_list = loss_list.numpy()
    # acc_list = acc_list.numpy()
    plt.figure(figsize=(10, 10))
    plt.subplot(2, 1, 1)
    plt.plot(epochs, acc_list, 'b', label='Training accuracy')
    plt.title('Training accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    # plt.show()


    plt.subplot(2, 1, 2)
    plt.plot(epochs, loss_list, 'g', label='Training loss')
    plt.title('Training loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    # plt.show()
    plt.savefig("loss.png")

model = AttGNN()
model.to(device)
num_epochs = 100
optimizer =  torch.optim.Adam(model.parameters(), lr= 0.01)
loss_list = []
acc_list = []

for epoch in tqdm(range(num_epochs)):
    loss, acc = train(model, device, trainloader, optimizer, epoch)
    torch.save(model.state_dict(), "GCN_tt.pth")
    loss_list.append(loss.item())
    acc_list.append(acc)

plot_trend(num_epochs, loss_list, acc_list)