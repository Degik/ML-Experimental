import numpy
import torch
import torch.nn as nn

class NeuralNetworkLayer(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetworkLayer, self).__init__()

        #Definisco i layer
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()

        self.layer2 = nn.Linear(input_size, hidden_size)
        self.relu2 = nn.ReLU()

        self.layer3 = nn.Linear(input_size, hidden_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu1(x)

        x = self.layer2(x)
        x = self.relu2(x)

        x = self.layer3(x)
        x = self.sigmoid(x)

        return x

def createNet(input_size:int, hidden_size:int, output_size:int) -> NeuralNetworkLayer:
    return NeuralNetworkLayer(input_size, hidden_size, output_size)