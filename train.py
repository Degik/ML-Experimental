import torch
import Trainer as tr
import parserData as prd
import NeuralNetwork as nt
from validation import validateNet
from argumentParser import argumentParser
from torch.utils.data import TensorDataset, DataLoader
from manageTensor import createTensor, moveTensorToDevice

#Dataset settings
file_path_training_set = "dataset/ML-CUP23-TR.csv"
file_path_test_set_input = "dataset/ML-CUP23-TS.csv"
file_path_test_set_target = "dataset/ML-CUP23-TARGET.csv"
#Create net obj
net = nt.NeuralNetworkLayer()
#Parsing arguments
args = argumentParser()
#Create trainer obj
trainer = tr.createTrainer(args, net)
#Take dataset for training from csv file
dataset = prd.importDataSetCUP(file_path_training_set, blind=False)
print(dataset)
#Take dataset for testing from csv file
dataset_test_input = prd.importDataSetCUP(file_path_test_set_input, blind=False)
dataset_test_target = prd.importDataSetCUPValidationTarget(file_path_test_set_target)
#print(dataset_test_target)
#Take dataset input information
dataset_input = prd.takeInputDataset(dataset, blind=False)
#Take dataset output information
dataset_output = prd.takeOutputDataset(dataset, blind=False)
#Create tensor from dataset
tensor_input = createTensor(dataset_input)
tensor_output = createTensor(dataset_output)
tensor_test = createTensor(dataset_test_input)
tensor_target = createTensor(dataset_test_target)
#Move tensor to selected device
net = nt.moveNetToDevice(net, trainer.device)
# Setting all data in double
net = net.double()
tensor_input = tensor_input.double()
tensor_output = tensor_output.double()
#Create tensorDataset for use TenserDataset
tensor_dataset = TensorDataset(tensor_input, tensor_output)
# Create data loader, is important for use batch computing inside of traning loop
data_loader = DataLoader(tensor_dataset, batch_size=trainer.batch, shuffle=True)
#Training loop
for epoch in range(trainer.epochs):
    for batch_input, batch_output in data_loader:
        batch_input = moveTensorToDevice(batch_input, trainer.device)
        batch_output = moveTensorToDevice(batch_output, trainer.device)
        #Forward pass
        outputs = net(batch_input)
        #Training loss
        loss = trainer.criterion(outputs, batch_output)
        #Backward and optimization
        trainer.optimizer.zero_grad()
        loss.backward()
        trainer.optimizer.step()
    print(f'Epoch [{epoch+1}/{trainer.epochs}], Loss: {loss.item():.4f}')

#validateNet(tensor_test, tensor_target, net, trainer)