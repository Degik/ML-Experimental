import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralNetworkLayer(nn.Module):
    def __init__(self):
        super(NeuralNetworkLayer, self).__init__()

        #Layer 1 Input: 10 Output: 50
        self.layer1 = nn.Linear(10, 50)
        #Layer 2 Input: 50 Output: 100
        self.layer2 = nn.Linear(50, 100)
        #Layer 3 Input: 100 Output: 100
        self.layer3 = nn.Linear(100, 100)
        #Layer 4 Input: 100 Output: 50
        self.layer4 = nn.Linear(100, 50)
        #Layer 5 Input: 50 Output: 3
        self.layer5 = nn.Linear(50, 3)

    def forward(self, x):
        x = self.layer1(x)
        x = F.relu(x)

        x = self.layer2(x)
        x = F.relu(x)

        x = self.layer3(x)
        x = F.relu(x)

        x = self.layer4(x)
        x = F.relu(x)

        x = self.layer5(x)
        return x

def moveNetToDevice(net:NeuralNetworkLayer, device:str) -> NeuralNetworkLayer:
    if device is not None:
        return net.to(device)
    return net