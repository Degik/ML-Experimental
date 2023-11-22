import torch
import pandas as pd

def createTensor(dataset:pd.DataFrame) -> torch.Tensor:
    try:
        tensor = torch.tensor(dataset.values, dtype=torch.float64)
    except Exception as e:
        print("Error | Creating tensor!")
        print(e)
    return tensor