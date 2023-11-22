import numpy
import torch
import torch.nn as nn
import pandas as pd

import NeuralNetwork as nt
from parserData import importDataSetCUP
from manageTensor import createTensor

# Net settings
input_size=10
hidden_size=20
output_size=20
file_path = "ML-CUP23-TR.csv"

#take dataset from csv file
dataset = importDataSetCUP(file_path, True)
#Create tenser from dataset
tensor = createTensor(dataset)

#In progress