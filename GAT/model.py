import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool as gep, dense
import numpy as np

class GCNN(nn.Module):
    def __init__(self, output = 5, num_features = 20, output_dim = 128, dropout = 0.3):
        super(GCNN, self).__init__()

        print("GCNN Loaded")
        self.batch = None
        self.output = output
        self.final_conv_acts_prot = None
        self.final_conv_grads_prot = None

        self.conv1 = GCNConv(num_features, num_features)
        self.conv2 = GCNConv(num_features, num_features)
        self.conv3 = GCNConv(num_features, num_features)
        self.pro1_fc1 = dense.Linear(num_features, output)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax()

        self.fc1 = nn.Linear(output_dim, 64)
        self.fc2 = nn.Linear(256, 64)
        self.out = nn.Linear(20, self.output)

    def activations_hook_prot(self, grad1):
        self.final_conv_grads_prot1 = grad1

    def forward(self, prot):
        prot_x, prot_edge_index, prot_dist, prot_batch = prot.x, prot.edge_index, prot.edge_attr, prot.batch

        self.batch = prot_batch

        prot_x.requires_grad = True
        with torch.enable_grad():
            self.final_conv_acts_prot1 = self.conv1(prot_x, prot_edge_index)
        self.final_conv_acts_prot1.register_hook(self.activations_hook_prot)

        # print(self.batch)
        # print(self.batch.size())
        # exit()
        # x = self.conv1(prot_x, prot_edge_index)
        # x = x.to(torch.float32)

        # x = self.conv2(x, prot_edge_index, prot_dist)
        # x = x.to(torch.float32)

        # x = self.conv3(x, prot_edge_index, prot_dist)
        # x = x.to(torch.float32)
        # print(x)
        # print(x.size())
        
        x = F.relu(self.final_conv_acts_prot1)

        x = gep(x, prot_batch)   
        # flatten
        # x = self.relu(self.pro1_fc1(x))
        # x = self.dropout(x)

        print(x)
        print(x.size())
        # exit()

        # x = self.fc1(x)
        # x = self.relu(x)
        # x = self.dropout(x)

        # x = self.fc2(x)
        # x = self.relu(x)
        # x = self.dropout(x)
        # print(x)
        
        out = self.out(x)
        print("-=-=-=-=Before softmax-==-=-=-=-=")
        print(out)
        out = self.softmax(out)
        # print(out)
        
        # max_index = out.argmax(axis=1)
        # out[np.arange(out.shape[0]), max_index] = 1
        # out[out != 1] = 0
        return out

# net = GCNN()
# print(net)

class AttGNN(nn.Module):
    def __init__(self, n_output=5, num_features_pro= 20, output_dim=20, dropout=0.2, heads = 1 ):
        super(AttGNN, self).__init__()

        print('AttGNN Loaded')

        self.hidden = 8
        self.heads = 1
        
        # for protein 1
        self.pro1_conv1 = GATConv(num_features_pro, 20, heads=self.heads, dropout=0.2)
        self.pro1_fc1 = nn.Linear(128, output_dim)

        self.relu = nn.LeakyReLU()
        self.softmax = nn.Softmax()
        self.dropout = nn.Dropout(dropout)
        # combined layers
        self.fc1 = nn.Linear(output_dim, 256)
        self.fc2 = nn.Linear(256, 64)
        self.out = nn.Linear(20, n_output)
        


    def forward(self, prot):

        # get graph input for protein 1 
        prot_x, prot_edge_index, prot_dist, prot_batch = prot.x, prot.edge_index, prot.edge_attr, prot.batch
        
        x = self.pro1_conv1(prot_x, prot_edge_index)
        x = self.relu(x)
        
	# global pooling
        x = gep(x, prot_batch)  
       
        # flatten
        # x = self.relu(self.pro1_fc1(x))
        # x = self.dropout(x)

        # add some dense layers
        # xc = self.fc1(x)
        # xc = self.relu(xc)
        # xc = self.dropout(xc)
        # xc = self.fc2(xc)
        # xc = self.relu(xc)
        # xc = self.dropout(xc)
        out = self.out(x)
        out = self.softmax(out)
        return out

net_GAT = AttGNN()
print(net_GAT)

