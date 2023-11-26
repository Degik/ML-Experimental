import argparse

def argumentParser() -> dict[str, any]:
    parser = argparse.ArgumentParser()
    parser.add_argument('-e','--epochs', type=int, default=100, help='Total training epochs', required=False)
    parser.add_argument('-b','--batch', type=int, default=10, help='Total batch size', required=False)
    parser.add_argument('-p','--patience', type=int, default=100,help='Patiance for earlystopping', required=False)
    parser.add_argument('-d','--device', type=str, default='cuda',help='Select device cuda (gpu) or cpu', required=False)
    parser.add_argument('-w','--workers', type=int, default=4,help='Number of wokers for DataLoader', required=False)
    parser.add_argument('-tb','--tensorB', type=bool, default=False,help='Activate tensorBoard for drawing the graph of Neural Network', required=False)
    return vars(parser.parse_args())