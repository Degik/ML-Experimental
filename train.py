import torch
import Trainer as tr
import parserData as prd
import NeuralNetwork as nt
from plotFn import plotModel
from clearml import Task
from argumentParser import argumentParser
from torch.utils.data import TensorDataset, DataLoader
from manageTensor import createTensor, moveTensorToDevice

#Dataset settings
#Cup dataset
file_path_cup_tr = "dataset/CUP/ML-CUP23-TR.csv"
file_path_cup_ts = "dataset/CUP/ML-CUP23-TS.csv"
file_path_cup_target = "dataset/CUP/ML-CUP23-TARGET.csv"
#Monk dataset
file_path_monk_tr = "dataset/MONK/monks-1.train"
file_path_monk_ts = "dataset/MONK/monks-1.test"

#Parsing arguments
args = argumentParser()

if args['clearml']:
    # Inizialize clearml
    task = Task.init(project_name='ML-Project', task_name='MyTask')
    logger = task.get_logger()

if args['cup']:
    #Create net obj
    net = nt.NeuralNetworkLayerCup()
else:
    net = nt.NeuralNetworkLayerMonk()

#Create trainer obj
trainer = tr.createTrainer(args, net)

if trainer.cup:
    #If cup dataset
    #Take dataset for training and validation from csv file
    dataset_tr = prd.importDatasetCUP(file_path_cup_tr, blind=False)
    dataset_ts_input = prd.importDatasetCUP(file_path_cup_ts, blind=False)
    dataset_ts_output = prd.importDataSetCUPValidationTarget(file_path_cup_target)
    #Create tensor_tr from input and output
    tensor_tr_input = createTensor(prd.takeCupInputDataset(dataset_tr, blind=False))
    tensor_tr_output = createTensor(prd.takeCupOutputDataset(dataset_tr, blind=False))
    #Create tensor_ts from input and output
    tensor_ts_input = createTensor(prd.takeCupInputDataset(dataset_ts_input, blind=False))
    tensor_ts_output = createTensor(prd.takeCupOutputDataset(dataset_ts_output, blind=False))
else:
    #If monk dataset
    dataset_tr = prd.importMonkDataset(file_path_monk_tr)
    dataset_ts = prd.importMonkDataset(file_path_monk_ts)
    #Take input and output information while converting to tensor
    tensor_tr_input = createTensor(prd.takeMonkInputDataset(dataset_tr))
    tensor_tr_output = createTensor(prd.takeMonkOutputDataset(dataset_tr))
    tensor_ts_input = createTensor(prd.takeMonkInputDataset(dataset_ts))
    tensor_ts_output = createTensor(prd.takeMonkOutputDataset(dataset_ts))



#
#Move tensor to selected device
net = nt.moveNetToDevice(net, trainer.device)

# Setting all data in double
net = net.double()
tensor_tr_input = tensor_tr_input.double()
tensor_tr_output = tensor_tr_output.double()
tensor_ts_input = tensor_ts_input.double()
tensor_ts_output = tensor_ts_output.double()

#Upload the struct of the net on clearml
if trainer.clearml:
    #Create file to upload
    with open('model_structure.txt', 'w') as f:
        print(net, file=f)
    #Upload the file on clearml
    task.upload_artifact('model_structure', artifact_object='model_structure.txt')
    #Upload tensor_tr_input on clearml
    task.upload_artifact(name='tensor_tr_input', artifact_object=tensor_tr_input)

#Create tensorDataset for use TenserDataset
tensor_dataset_tr = TensorDataset(tensor_tr_input, tensor_tr_output)
tensor_dataset_ts = TensorDataset(tensor_ts_input, tensor_ts_output)
# Create data loader, is important for use batch computing inside of traning loop
data_loader_tr = DataLoader(tensor_dataset_tr, batch_size=trainer.batch, shuffle=True)
# DataLoader for validation phase
data_loader_ts = DataLoader(tensor_dataset_ts, batch_size=trainer.batch, shuffle=False)
#Draw Graph on TensorBoard
if trainer.tensorB:
    writer = nt.printGraph(net, tensor_tr_input.to(trainer.device))
#Training loop
for epoch in range(trainer.epochs):
    total_loss = 0
    for batch_input, batch_output in data_loader_tr:
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
    
    train_loss = total_loss / len(data_loader_tr)
    #Plots on clearml
    if trainer.clearml:
        logger.report_scalar(title='Training Metrics', series='Loss', value=loss.item(), iteration=epoch)
        logger.report_scalar(title='Training Metrics', series='Avg-Loss', value=train_loss, iteration=epoch)
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
        for batch_input, batch_target in data_loader_ts:
            batch_input = moveTensorToDevice(batch_input, trainer.device)
            batch_target = moveTensorToDevice(batch_target, trainer.device)

            # Forward pass
            outputs = net(batch_input)

            # Calculate loss
            lossVal = trainer.criterion(outputs, batch_target)
            total_loss += lossVal.item()

        # Calculate medium loss
        val_loss = total_loss / len(data_loader_ts)
        
        #Plots on clearml
        if trainer.clearml:
            #logger.report_scalar(title='Training Metrics', series='Val-AvgLoss', value=lossVal.item(), iteration=epoch)
            logger.report_scalar(title='Training Metrics', series='Val-Loss', value=val_loss, iteration=epoch)
        if trainer.tensorB:
            writer.add_scalar('Loss/validation', val_loss, epoch)
            
    # Return to train mode
    net.train()
    
    print(f'Epoch [{epoch+1}/{trainer.epochs}], Train-Loss: {train_loss:.4f}, Validate-Loss: {val_loss:.4f}')

#plotModel(tensor_input,tensor_output, net, trainer) # Dind't work