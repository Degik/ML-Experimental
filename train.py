import Trainer as tr
import parserData as prd
import NeuralNetwork as nt
from manageTensor import createTensor
from argumentParser import argumentParser

#Net settings
input_size=10
hidden_size=10
output_size=3
file_path = "dataset/ML-CUP23-TR.csv"
#Create net obj
net = nt.createNet(input_size,hidden_size,output_size)
#Parsing arguments
args = argumentParser()
#Create trainer obj
trainer = tr.createTrainer(args, net)
#Take dataset from csv file
dataset = prd.importDataSetCUP(file_path, trainer.blind)
#Take dataset input information
dataset_input = prd.takeInputDataset(dataset, trainer.blind)
#Take dataset output information
dataset_output = prd.takeOutputDataset(dataset, trainer.blind)
#Create tensor from dataset
tensor_input = createTensor(dataset_input)
tensor_output = createTensor(dataset_output)
# Setting all data in double
net = net.double()
tensor_input = tensor_input.double()
tensor_output = tensor_output.double()
#Training loop
for epoch in range(trainer.epochs):
    #Forward pass
    outputs = net(tensor_input)
    #Training loss
    loss = trainer.criterion(outputs, tensor_output)
    #Backward and optimization
    trainer.optimizer.zero_grad()
    loss.backward()
    trainer.optimizer.step()
    
