import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union
from torch.utils.tensorboard import SummaryWriter

class NeuralNetworkLayerCup(nn.Module):
    def __init__(self):
        super(NeuralNetworkLayerCup, self).__init__()

        #Layer 1 Input: 10 Output: 15
        self.layer1 = nn.Linear(10, 20)
        #self.dropout1 = nn.Dropout(0.5)
        self.bn1 = nn.BatchNorm1d(20)
        #Layer 2 Input: 15  Output: 10
        self.layer2 = nn.Linear(20, 10)
        #self.dropout2 = nn.Dropout(0.5)
        self.bn2 = nn.BatchNorm1d(10)
        #Layer 3 Input: 10 Output: 3
        self.layer3 = nn.Linear(10, 3)

    def forward(self, x):
        x = self.layer1(x)
        #x = self.dropout1(x)
        x = self.bn1(x)
        #x = F.relu(x)

        x = self.layer2(x)
        #x = self.dropout2(x)
        x = self.bn2(x)
        #x = F.relu(x)

        x = self.layer3(x)
        return x


class NeuralNetworkLayerMonk(nn.Module):
    def __init__(self):
        super(NeuralNetworkLayerMonk, self).__init__()

        #Layer 1 Input: 6 Output: 8
        self.layer1 = nn.Linear(6, 3)
        self.dropout1 = nn.Dropout(0.5)
        self.bn1 = nn.BatchNorm1d(3)
        #Layer 2 Input: 64  Output: 32
        self.layer2 = nn.Linear(3, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.dropout1(x)
        x = self.bn1(x)
        x = self.layer2(x)
        #x = F.sigmoid(x)
        return x
    

def moveNetToDevice(net:Union[NeuralNetworkLayerCup, NeuralNetworkLayerMonk], 
                    device:str) -> Union[NeuralNetworkLayerCup, NeuralNetworkLayerMonk]:
    if device is not None:
        return net.to(device)
    return net

def printGraph(net:Union[NeuralNetworkLayerCup, NeuralNetworkLayerMonk], 
               dataset_input:torch.Tensor) -> SummaryWriter:
    writer = SummaryWriter('runs/my_model')
    writer.add_graph(net, dataset_input)
    writer.close()
    return writer