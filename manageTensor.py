import torch
import pandas as pd

def createTensor(dataset:pd.DataFrame) -> torch.Tensor:
    try:
        return torch.tensor(dataset.values, dtype=torch.float64)
    except Exception as e:
        print("Error | Creating tensor!")
        print(e)
        exit(1)

def moveTensorToDevice(tensor:torch.Tensor, device:str) -> torch.Tensor:
    if device is not None:
        return tensor.to(device)
    return tensor