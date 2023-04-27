import torch
import torch.nn.functional as F
# from data import load_bbbp
import torch.nn as nn
import numpy as np
from model import GCNN
import matplotlib.pyplot as plt
# import hyperparameters as hp
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
import matplotlib
import matplotlib.cm as cm
from skimage.io import imread
from cairosvg import svg2png, svg2ps
import os
from torch_geometric.data import DataLoader
import pandas as pd
from rdkit import Chem
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import random
from data_prepare import explainloader, testloader
# from test_data_prepare import test_dataset_loader
import csv
from visualize import *
# df = pd.read_csv('bbbp/BBBP.csv')

prot_list = []
predicted_class_list = []
node_num_list = []
edge_num_list = []
top5residue_list = []
top5index_list = []

def int_to_class(ten):
    x = torch.argmax(ten)
    if x == 0:
    #     return "Isomerase"
    # elif x == 1:
        return "Kinase"
    elif x == 1:
    #     return "Phosphatase"
    # elif x == 3:
        return "Protease"
    elif x == 2:
        return "Receptor"
    
# draw an image for the molecule graph 
def img_for_mol(mol, atom_weights=[]):
    # print(atom_weights)
    highlight_kwargs = {}
    if len(atom_weights) > 0:
        norm = matplotlib.colors.Normalize(vmin=-1, vmax=1)
        cmap = cm.get_cmap('bwr')
        plt_colors = cm.ScalarMappable(norm=norm, cmap=cmap)
        atom_colors = {
            i: plt_colors.to_rgba(atom_weights[i]) for i in range(len(atom_weights))
        }
        highlight_kwargs = {
            'highlightAtoms': list(range(len(atom_weights))),
            'highlightBonds': [],
            'highlightAtomColors': atom_colors
        }
        # print(highlight_kwargs)


    rdDepictor.Compute2DCoords(mol)
    drawer = rdMolDraw2D.MolDraw2DSVG(280, 280)
    drawer.SetFontSize(1)

    mol = rdMolDraw2D.PrepareMolForDrawing(mol)
    drawer.DrawMolecule(mol, **highlight_kwargs)
                        # highlightAtoms=list(range(len(atom_weights))),
                        # highlightBonds=[],
                        # highlightAtomColors=atom_colors)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    svg = svg.replace('svg:', '')
    svg2png(bytestring=svg, write_to='tmp.png', dpi=100)
    img = imread('tmp.png')
    os.remove('tmp.png')
    return img

# def highlight_top_5(row):
#     if row["status"] == "Positive":
#         return ['background-color: yellow'] * len(row)
#     else:
#         return ['background-color: white'] * len(row)
def statistic_table(prot_list, predicted_class_list, node_num_list, edge_num_list, top5residue_list, top5index_list):
    df = pd.DataFrame(list(zip(prot_list, predicted_class_list, node_num_list, edge_num_list, top5residue_list, top5index_list)),
               columns =['Proteins', 'Predicted_class','Nodes', 'Edge', 'Top_5_residue', 'Top_5_index'])
    df.to_csv("statistic_all_3.csv")

def PPI_to_csv(prot_1, prot_1_seq, explain_prot_1,out, edge_index):
    adr = '/Users/benjamin/Documents/Classification/Graph-redo (work) copy/GCN'
    # adr = '/opt/data/proteins/mix_explain/'
   
    for count, i in enumerate(prot_1):
        print(i)
        path = str(i)
        print(path[-8:-4])
        filename = adr + int_to_class(out[count]) + '_' + path[-8:-4] + '.csv'
        print(filename)
        # filename = adr + str(prot_1[i])[-10:-4] + '_vs_' + str(prot_2[i])[-10:-4] + '_1.csv'

        # print(filename)
        try:
            f = open(filename , 'w')
            writer = csv.writer(f)
            writer.writerows(prot_1_seq[count])
            f.close()
        except Exception as e:
            print(e)
            exit()

        PPI = pd.read_csv(filename,names=['residue_prot1'], header=None)
        seq_size_prot1 = len(prot_1_seq[count])
        PPI['Probability_1'] = explain_prot_1
        print(explain_prot_1[:seq_size_prot1])
        # explain_prot_1 = explain_prot_1[seq_size_prot1:]

        tmp = PPI.nlargest(10,'Probability_1')
        # tmp = PPI[PPI["Probability_1"] == 0]
        print(tmp)
        top5index = tmp[tmp.columns[0]].index.to_list()
        top5residue = tmp[tmp.columns[0]].to_list()
        print(top5residue)
        print(top5index)
        
        PPI.style.highlight_max(color='blue', axis = 1)
        PPI.to_csv(filename)
        # exit()
        #visualization
        visualize_prot = visualize("/Users/benjamin/Documents/Classification/Graph/Data/raw_remove_inseertion_high_weight_3_class/"+ path[-8:], prot_1_seq[count], top5index, filename, explain_prot_1)
        node_num, edge_num = visualize_prot._show_graph_with_labels()
        
        prot_list.append(path[-8:-4])
        predicted_class_list.append(int_to_class(out[count]))
        node_num_list.append(node_num)
        edge_num_list.append(edge_num)
        top5residue_list.append(top5residue)
        top5index_list.append(top5index)

    return

def plot_explanations(model,out):
    print("--seq1-----")
    print(prot_1.seq)
    print(prot_1.name)
    print(prot_1.edge_index)
    # exit()
    final_conv_acts1_prot1 = model.final_conv_acts_prot1
    final_conv_grads1_prot1 = model.final_conv_grads_prot1
    print("---------final_conv_acts1_prot1------------")
    print(final_conv_acts1_prot1)
    print(len(final_conv_acts1_prot1))
    
    grad_cam_weights_prot1 = grad_cam(final_conv_acts1_prot1, final_conv_grads1_prot1)
    scaled_grad_cam_weights_prot1 = MinMaxScaler(feature_range=(0,1)).fit_transform(np.array(grad_cam_weights_prot1).reshape(-1, 1)).reshape(-1, )
    print("-------scaled_grad_cam_weights_prot1--------")
    print(grad_cam_weights_prot1)
    np.savetxt("foo_prot1.csv", grad_cam_weights_prot1, delimiter=" ")
    

    print("----------prot_1.edge_index.cpu().numpy()--------------")
    print(prot_1.edge_index.cpu().numpy())

    PPI_to_csv(prot_1.name, prot_1.seq, scaled_grad_cam_weights_prot1,out,prot_1.edge_index)

def saliency_map(input_grads):
    # print('saliency_map')
    node_saliency_map = []
    for n in range(input_grads.shape[0]): # nth node
        node_grads = input_grads[n,:]
        node_saliency = torch.norm(F.relu(node_grads)).item()
        node_saliency_map.append(node_saliency)
    return node_saliency_map

def grad_cam(final_conv_acts, final_conv_grads):
    # print('grad_cam')
    node_heat_map = []
    # this calculate the ((alpha)^c)_k
    alphas = torch.mean(final_conv_grads, axis=0) # mean gradient for each feature (512x1)
    print(final_conv_acts)
    print(final_conv_acts.shape[0])
    for n in range(final_conv_acts.shape[0]): # nth node
        # @ here is matrix multiplication
        # this calculate the 
        node_heat = F.relu(alphas @ final_conv_acts[n]).item()
        node_heat_map.append(node_heat)
        # print("-----final_conv_acts.shape[0]----")
        # print(final_conv_acts.shape[0])
        # break
    print(node_heat_map)
    print(len(node_heat_map))
    # exit()
    return node_heat_map

def ugrad_cam(mol, final_conv_acts, final_conv_grads):
    # print('new_grad_cam')
    node_heat_map = []
    alphas = torch.mean(final_conv_grads, axis=0) # mean gradient for each feature (512x1)
    for n in range(final_conv_acts.shape[0]): # nth node
        node_heat = (alphas @ final_conv_acts[n]).item()
        node_heat_map.append(node_heat)

    node_heat_map = np.array(node_heat_map[:mol.GetNumAtoms()]).reshape(-1, 1)
    pos_node_heat_map = MinMaxScaler(feature_range=(0,1)).fit_transform(node_heat_map*(node_heat_map >= 0)).reshape(-1,)
    neg_node_heat_map = MinMaxScaler(feature_range=(-1,0)).fit_transform(node_heat_map*(node_heat_map < 0)).reshape(-1,)
    return pos_node_heat_map + neg_node_heat_map

# dataset = load_bbbp(hp.N)
# random.Random(hp.shuffle_seed).shuffle(dataset)
# split_idx = int(np.floor(len(dataset)*hp.train_frac))
# test_dataset = dataset[split_idx:]
# loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
# loader = testloader
# np.random.seed(0)
# torch.manual_seed(0)
loader = explainloader
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCNN().to(device)
# model.load_state_dict(torch.load('gcn_state_dict.pt'))
model.load_state_dict(torch.load('/Users/benjamin/Documents/Graph-explain/GCN-93.pth', map_location=torch.device('cpu')))
# model.load_state_dict(torch.load('/opt/data/proteins/GCN.pth'))


model.eval()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print(model)
model.train()
total_loss = 0
counter = 0
# for prot_1, prot_2, label in testloader:
predictions = torch.Tensor()
labels = torch.Tensor()

for prot_1, label in explainloader:
    counter = counter+1
    prot_1 = prot_1.to(device)

    optimizer.zero_grad()
    out = model(prot_1)
    predictions = torch.cat((predictions,out.cpu()), 0)
    labels = torch.cat((labels, label.cpu()), 0)

    loss_func = nn.CrossEntropyLoss()
    loss = loss_func(out, label.to(device))    
    loss.backward()

    plot_explanations(model,out)

labels = labels.detach().numpy()
# predictions = predictions.detach().numpy()
predictions = torch.argmax(predictions, dim = 1)
print("====================")
print(labels)
print(predictions)
print(confusion_matrix(labels, predictions))

# statistic - it uses data from visualization
statistic_table(prot_list, predicted_class_list, node_num_list, edge_num_list, top5residue_list, top5index_list)
# print(counter)