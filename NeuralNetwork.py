import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

class NeuralNetworkLayer(nn.Module):
    def __init__(self):
        super(NeuralNetworkLayer, self).__init__()

        #Layer 1 Input: 10 Output: 64
        self.layer1 = nn.Linear(10, 12)
        #self.dropout1 = nn.Dropout(0.5)
        self.bn1 = nn.BatchNorm1d(12)
        #Layer 2 Input: 64  Output: 32
        self.layer2 = nn.Linear(12, 8)
        #self.dropout2 = nn.Dropout(0.5)
        self.bn2 = nn.BatchNorm1d(8)
        #Layer 3 Input: 32 Output: 3
        self.layer3 = nn.Linear(8, 3)

    def forward(self, x):
        x = self.layer1(x)
        #x = self.dropout1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.layer2(x)
        #x = self.dropout2(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.layer3(x)
        return x

def moveNetToDevice(net:NeuralNetworkLayer, device:str) -> NeuralNetworkLayer:
    if device is not None:
        return net.to(device)
    return net

def printGraph(net:NeuralNetworkLayer, dataset_input:torch.Tensor) -> SummaryWriter:
    writer = SummaryWriter('runs/my_model')
    writer.add_graph(net, dataset_input)
    writer.close()
    return writer