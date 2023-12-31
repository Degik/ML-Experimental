import pandas as pd

def importDatasetCUP(file_name:str, blind:bool=False) -> pd.DataFrame:
    dataset = None
    columns_name = ["ID"] + [f"V{i}" for i in range(1, 11)] + ["X", "Y", "Z"]
    try:
        dataset = pd.read_csv(file_name, dtype=float, names=columns_name)
    except Exception as e:
        print("Error | Parsing dataset!")
        print(e)
    if blind:
        dataset = dataset.iloc[:, :-3] #Remove the last 3 columns
    dataset.set_index('ID', inplace=True)
    return dataset

def importDataSetCUPValidationTarget(file_name:str) -> pd.DataFrame:
    dataset = None
    columns_name = ["ID"] + ["X", "Y", "Z"]
    try:
        dataset = pd.read_csv(file_name, dtype=float, names=columns_name)
    except Exception as e:
        print("Error | Parsing target dataset for validation!")
        print(e)
    dataset.set_index('ID', inplace=True)
    return dataset

def importMonkDataset(file_name:str) -> pd.DataFrame:
    dataset = None
    columns_name = ["Y"] + [f"X{i}" for i in range(1,7)] + ["ID"]
    try:
        dataset = pd.read_csv(file_name, sep=" ", names=columns_name)
    except Exception as e:
        print("Error | Parsing target dataset for validation!")
        print(e)
    dataset.set_index('ID', inplace=True)
    return dataset

def takeMonkInputDataset(dataset:pd.DataFrame) -> pd.DataFrame:
    return dataset.iloc[:, 1:] #Return dataset without first and last column

def takeMonkOutputDataset(dataset:pd.DataFrame) -> pd.DataFrame:
    return dataset.iloc[:,[0]] #Return dataset with only first column

def takeCupInputDataset(dataset:pd.DataFrame, blind:bool=False) -> pd.DataFrame:
    #Create dataset_input and dataset output
    if not blind:
        dataset_input = dataset.iloc[:, :-3] #Take all columns without the last 3 columns (output dataset)
        return dataset_input
    else:
        return dataset #Dataset is already an input dataset
    
def takeCupOutputDataset(dataset:pd.DataFrame, blind:bool=False) -> pd.DataFrame:
    #Create dataset_output and dataset output
    if not blind:
        dataset_output = dataset.iloc[:, -3:] #Take the last 3 columns
        return dataset_output
    else:
        print("Info | Dataset doesn't have output information")
        return None