import torch

PARAMETERS_NAME = ['--num_iter', '--norm_mode', '--split_mode', '--num_neighbors', '--num_epochs', '--num_shots', '--margin', '--weights', '--k_groups', '--is_improved']
DEFAULT_VALUES = [1, 'global max-min norm', 'Hold-out', 3, 100, 100, 0.01, [0.7, 0.1], 8, False]
TYPES = ['int']+['string']*2+['int']*3+['float', 'list', 'int', 'bool']
FILES_NAME = [['TS1', 'TS2', 'TS3', 'TS4', 'VS1', 'CE','CP','SE'], 'profile']
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def printHelp():
    print('here')

def printError(error):
    if error == True:
        printHelp()
        exit(1)

def setParameter(args, param_name, default_value, type):
    param = None
    for arg in args:
        if arg.startswith(param_name):
            if type == 'int':
                param = int(arg.split('=')[1])
            elif type == 'float':
                param = float(arg.split('=')[1])
            elif type == 'bool':
                param = bool(arg.split('=')[1])
            elif type == 'string':
                param = str(arg.split('=')[1])
            else:
                param = list(map(float, arg.split('=')[1].split(',')))
    return param if param is not None else default_value

def getConfigs(args):
    return [setParameter(args, parameter_name, default_value, s_type) for parameter_name, default_value, s_type in zip(PARAMETERS_NAME, DEFAULT_VALUES, TYPES)]