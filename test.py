import torch 
# from metrics import *
from data_prepare import testloader
import torch.nn as nn
from model import GCNN, AttGNN
from sklearn import metrics
import pandas as pd
from sklearn.metrics import confusion_matrix

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

model = GCNN()
model.load_state_dict(torch.load("GCN_tt.pth"))
model.to(device)
model.eval()
predictions = torch.Tensor()
labels = torch.Tensor()
loss_func = nn.CrossEntropyLoss()
correct = 0
with torch.no_grad():
    for prot, label in testloader:
        prot_1 = prot.to(device)
        output = model(prot)
        print("-==-=-=-=label-=-=-===-=")
        print(label)
        print("-==-=-=-=output-=-=-===-=")
        print(output)
        loss = loss_func(output, label.to(device))

        predictions = torch.cat((predictions, output.cpu()), 0)
        labels = torch.cat((labels, label.cpu()), 0)

# labels = torch.tensor(labels.numpy())
# labels = labels.detach().numpy()
# predictions = torch.tensor(predictions.numpy(),dtype=torch.float32)
predictions = torch.argmax(predictions, dim = 1)


correct += (predictions == labels).sum().item()
acc = 100 * correct / len(predictions)



# acc = accuracy(labels, predictions)
# prec = precision(labels,predictions)
# rec = recall(labels,predictions)
# f1 = f1(labels,predictions)

print(f'loss : {loss}')
print(f'Accuracy : {acc}')
# print(f'f-score : {f1}')
# print(f'precision: {prec}')
# print(f'recall: {rec}')

print(labels)
t_np = labels.numpy() #convert to Numpy array
df = pd.DataFrame(t_np) #convert to a dataframe
df.to_csv("labels.csv",index=False) #save to file
print(predictions)
t_np = predictions.numpy() #convert to Numpy array
df = pd.DataFrame(t_np) #convert to a dataframe
df.to_csv("predictions.csv",index=False) #save to file
print(metrics.classification_report(labels, predictions, digits=3))

print(confusion_matrix(labels, predictions))