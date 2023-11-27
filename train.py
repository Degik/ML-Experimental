import torch
import Trainer as tr
import parserData as prd
import NeuralNetwork as nt
from clearml import Task
from argumentParser import argumentParser
from torch.utils.data import TensorDataset, DataLoader
from manageTensor import createTensor, moveTensorToDevice


#Dataset settings
file_path_training_set = "dataset/ML-CUP23-TR.csv"
file_path_test_set_input = "dataset/ML-CUP23-TS.csv"
file_path_test_set_target = "dataset/ML-CUP23-TARGET.csv"
# Inizialize clearml
task = Task.init(project_name='ML-Project', task_name='MyTask')
logger = task.get_logger()
#Create net obj
net = nt.NeuralNetworkLayer()
# Draw net structure
with open('model_structure.txt', 'w') as f:
    print(net, file=f)
# Upload net structure on clearml
task.upload_artifact('model_structure', artifact_object='model_structure.txt')
#Parsing arguments
args = argumentParser()
#Create trainer obj
trainer = tr.createTrainer(args, net)
#Take dataset for training from csv file
dataset = prd.importDataSetCUP(file_path_training_set, blind=False)
print(dataset)
#Take dataset for testing from csv file
dataset_test_input = prd.importDataSetCUP(file_path_test_set_input, blind=True)
dataset_test_target = prd.importDataSetCUPValidationTarget(file_path_test_set_target)
#Take dataset input information
dataset_input = prd.takeInputDataset(dataset, blind=False)
#Take dataset output information
dataset_output = prd.takeOutputDataset(dataset, blind=False)
#Create tensor from dataset
tensor_input = createTensor(dataset_input)
tensor_output = createTensor(dataset_output)
tensor_test = createTensor(dataset_test_input)
tensor_target = createTensor(dataset_test_target)
#Upload tensor to clearml
task.upload_artifact(name='tensor_input', artifact_object=tensor_input)
#Move tensor to selected device
net = nt.moveNetToDevice(net, trainer.device)
# Setting all data in double
net = net.double()
tensor_input = tensor_input.double()
tensor_output = tensor_output.double()
#Create tensorDataset for use TenserDataset
tensor_dataset = TensorDataset(tensor_input, tensor_output)
tensor_dataset_test = TensorDataset(tensor_test, tensor_target)
# Create data loader, is important for use batch computing inside of traning loop
data_loader = DataLoader(tensor_dataset, batch_size=trainer.batch, shuffle=True)
# DataLoader for validation phase
data_loader_test = DataLoader(tensor_dataset_test, batch_size=trainer.batch, shuffle=False)
#Draw Graph on TensorBoard
if trainer.tensorB:
    writer = nt.printGraph(net, tensor_input.to(trainer.device))
#Training loop
for epoch in range(trainer.epochs):
    total_loss = 0
    for batch_input, batch_output in data_loader:
        batch_input = moveTensorToDevice(batch_input, trainer.device)
        batch_output = moveTensorToDevice(batch_output, trainer.device)
        #Forward pass
        outputs = net(batch_input)
        #Training loss
        loss = trainer.criterion(outputs, batch_output)
        total_loss+=loss.item()
        #Backward and optimization
        trainer.optimizer.zero_grad()
        loss.backward()
        trainer.optimizer.step()
    
    train_loss = total_loss / len(data_loader)
    #plots on clearml
    logger.report_scalar(title='Loss', series='loss', value=train_loss, iteration=epoch)
    #Print all data on tensorboard
    if trainer.tensorB:
        writer.add_scalar('Loss/train', loss.item(), epoch)
        for name, param in net.named_parameters():
            writer.add_histogram(f'{name}/weights', param.data.cpu().numpy(), epoch)
            writer.add_histogram(f'{name}/grads', param.grad.data.cpu().numpy(), epoch)


    # Validation phase
    net.eval()
    with torch.no_grad():
        total_loss = 0
        for batch_input, batch_target in data_loader_test:
            batch_input = moveTensorToDevice(batch_input, trainer.device)
            batch_target = moveTensorToDevice(batch_target, trainer.device)

            # Forward pass
            outputs = net(batch_input)

            # Calculate loss
            lossVal = trainer.criterion(outputs, batch_target)
            total_loss += lossVal.item()

        # Calculate medium loss
        val_loss = total_loss / len(data_loader_test)
        
        #
        logger.report_scalar(title='Validation Loss', series='loss', value=val_loss, iteration=epoch)
        if trainer.tensorB:
            writer.add_scalar('Loss/validation', val_loss, epoch)
            
    # Return to train mode
    net.train()
    
    print(f'Epoch [{epoch+1}/{trainer.epochs}], Train-Loss: {train_loss:.4f}, Validate-Loss: {val_loss:.4f}')

