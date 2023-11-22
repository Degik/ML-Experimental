import argparse


parser = argparse.ArgumentParser()
parser.add_argument('-e','--epochs', type=int, default=100, help='Total training epochs', required=False)
parser.add_argument('-b','--batch', type=int, default=1, help='Total batch size', required=False)
parser.add_argument('-p','--patience', type=int, default=100,help='Patiance for earlystopping', required=False)
parser.add_argument('-d','--device', type=str, default='gpu',help='Select device gpu or cpu', required=False)
