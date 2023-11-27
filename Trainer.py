import torch
import NeuralNetwork
import torch.nn as nn
import torch.optim as optim

class Trainer:
    def __init__(self, epochs:int, batch:int, patience:int, device:str, workers:int, tensorB:bool, net:NeuralNetwork.NeuralNetworkLayer) -> None:
        #Train settings
        self.epochs = epochs
        self.batch = batch
        self.patience = patience
        #
        if device == "cuda":
            self.device = torch.device("cuda:0")
        else:
            self.device = None
        self.workers = workers
        self.tensorB = tensorB
        #Save net inside trainer
        self.net = net
        #Train Criterion
        self.criterion = nn.MSELoss()
        #self.criterion = nn.L1Loss()
        #self.criterion = nn.HuberLoss()

        #Optimizer
        learning_rate = 0.001 #To see
        self.optimizer = optim.SGD(net.parameters(), lr=learning_rate)


def createTrainer(args:dict[str, any], net) -> Trainer:
    #GPU enable
    #https://pytorch.org/get-started/locally/
    if args['device'] == 'cuda':
        #GPU device check
        if not torch.cuda.is_available():
            print("GPU CUDA is not available \n Try to check Nvidia drivers")
            exit(1)

    # Construct trainer obj with train settings
    trainer = Trainer(args['epochs'], args['batch'], args['patience'], args['device'], args['workers'], args['tensorB'], net)

    if trainer is not None:
        return trainer
    else:
        print("Failed to create trainer obj")
        exit(1)
