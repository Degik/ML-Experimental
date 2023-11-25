import torch
import NeuralNetwork
import torch.nn as nn
import torch.optim as optim

class Trainer:
    def __init__(self, epochs:int, batch:int, patience:int, device:str, net:NeuralNetwork.NeuralNetworkLayer) -> None:
        #Train settings
        self.epochs = epochs
        self.batch = batch
        self.patience = patience
        self.device = device

        #Save net inside trainer
        self.net = net
        #Train Criterion
        self.criterion = nn.CrossEntropyLoss()

        #Optimizer
        learning_rate = 0.1 #To see
        self.optimizer = optim.SGD(net.parameters(), lr=learning_rate)


def createTrainer(args:dict[str, any], net) -> Trainer:
    #GPU enable
    #https://pytorch.org/get-started/locally/
    if args['device'] == 'gpu':
        #GPU device check
        if not torch.cuda.is_available():
            print("GPU CUDA is not available \n Try to check Nvidia drivers")
            exit(1)

    # Construct trainer obj with train settings
    trainer = Trainer(args['epochs'], args['batch'], args['patience'], args['device'], net)

    if trainer is not None:
        return trainer
    else:
        print("Failed to create trainer obj")
        exit(1)
