import torch
import Trainer as t
import NeuralNetwork

def validateNet(tensor_val:torch.Tensor, tensor_target:torch.Tensor, net:NeuralNetwork.NeuralNetworkLayer, trainer:t.Trainer):
    with torch.no_grad():
        val_outputs = net(tensor_val)
        val_loss = trainer.criterion(val_outputs, tensor_target)
        print(f"Final Loss: {val_loss}")