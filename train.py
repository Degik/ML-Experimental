import Trainer as tr
import NeuralNetwork as nt
from parserData import importDataSetCUP
from manageTensor import createTensor
from argumentParser import argumentParser

#Net settings
input_size=10
hidden_size=20
output_size=20
file_path = "dataset/ML-CUP23-TR.csv"
#Create net obj
net = nt.createNet(input_size,hidden_size,output_size)
#Parsing arguments
args = argumentParser()
#Create trainer obj
trainer = tr.createTrainer(args, net)
#take dataset from csv file
dataset = importDataSetCUP(file_path, True)
#Create tensor from dataset
tensor = createTensor(dataset)
#Training loop
for epoch in range(trainer.epochs):
    #Forward pass
    outputs = net(tensor)
    #Training loss
    loss = trainer.criterion(outputs)

    trainer.optimizer.zero_grad()
    
