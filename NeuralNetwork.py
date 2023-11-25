import torch.nn as nn

class NeuralNetworkLayer(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetworkLayer, self).__init__()

        #Definisco i layer
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.tanh1 = nn.Tanh()
        #self.relu1 = nn.ReLU()

        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.tanh2 = nn.Tanh()
        #self.relu2 = nn.ReLU()

        self.layer3 = nn.Linear(input_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.layer1(x)
        x = self.tanh1(x)

        x = self.layer2(x)
        x = self.tanh2(x)

        x = self.layer3(x)
        x = self.sigmoid(x)

        return x

def createNet(input_size:int, hidden_size:int, output_size:int) -> NeuralNetworkLayer:
    return NeuralNetworkLayer(input_size, hidden_size, output_size)