import torch
import pandas as pd

def createTensor(dataset:pd.DataFrame) -> torch.Tensor:
    try:
        tensor = torch.tensor(dataset.values, dtype=torch.float64)
    except Exception as e:
        print("Error | Creating tensor!")
        print(e)
    return tensor

def createListOfTensors(*args) -> list:
    tensorsList = []
    for tensor in args:
        tensorsList.append(tensor)
    return tensorsList

def moveTensorToDevice(tensor:torch.Tensor, device:str) -> torch.Tensor:
    if device is not None:
        return tensor.to(device)
    return tensor