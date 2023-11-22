import pandas as pd

def importDataSetCUP(file_name:str, Blind:bool=False) -> pd.DataFrame:
    dataset = None
    columns_name = ["ID"] + [f"V{i}" for i in range(1, 11)] + ["X", "Y", "Z"]
    try:
        dataset = pd.read_csv(file_name, dtype=float, names=columns_name)
    except Exception as e:
        print("Error | Parsing dataset!")
        print(e)
    if Blind:
        dataset = dataset.iloc[:, :-3] #Remove the last 3 columns
    dataset.set_index('ID', inplace=True)
    return dataset