
import argparse
import os
import yaml
import sys

__author__ = 'jhennies'


# Makes sure the specified file is both a valid file and readable by the current user
def readablefile(prospective_file):
    if not os.path.isfile(prospective_file):
        raise argparse.ArgumentTypeError("readablefile:{0} is not a valid file".format(prospective_file))
    if os.access(prospective_file, os.R_OK):
        return prospective_file
    else:
        raise argparse.ArgumentTypeError("readablefile:{0} is not a readable file".format(prospective_file))

parser = argparse.ArgumentParser(description='Train neuronal network on large dataset.')
parser.add_argument('ConfigFile', type=readablefile, help='Specify YAML configuration file.')
args = parser.parse_args()

configfile = args.ConfigFile
print 'Selected configuration file:'
print configfile

# Yaml to dict reader
def yaml2dict(path):
     with open(path, 'r') as f: readict = yaml.load(f)
     return readict

# TODO: replace this by YAML file input
dataconfig = {
    'datapath': '/media/julian/Daten/mobi/h1.hci/data/fib25/raw_fib25.h5',
    'dataname': 'data',
    'roispath': '/media/julian/Daten/mobi/h1.hci/data/fib25/roissl512.pkl',
    'popcubes': 10
}

sys.path.append('/media/julian/Daten/src/hci/nasim/')
import nn_upscale
nnupsc = nn_upscale.nn_upscale(path=dataconfig['datapath'], datapath=dataconfig['dataname'], resultfile=None)
nnupsc.train_nn(roispath=dataconfig['roispath'], popcubes=dataconfig['popcubes'])