import sys
from config.config import *
from taskManager import *
from data.statisticalTools import *

DTW_PARAMS_LEN = 8 #main.py DTW --num_iter=<num_iterations> --norm_mode=<norm_mode> --split_mode=<split_mode> --num_neighbors=<num_neighbors> --weights='<weights>' --k_groups=<k_groups>
AE_PARAMS_LEN = 9 #main.py AE --num_iter=<num_iteractions> --norm_mode=<norm_mode> --split_mode=<split_mode> --num_neighbors=<num_neighbors> --num_epochs=<num_epochs> --weights=<weights> --k_groups=<k_groups>
SIAMESE_PARAMS_LEN = 11 #main.py SIAMESE --num_iter=<num_iteraction> --norm_mode=<norm_mode> --split_mode=<split_mode> --num_neighbors=<num_neighbors> --num_epochs=<num_epochs> --num_shots=<num_shoots> --margin=<margin> --weights=<weights> --k_groups=<k_groups>
TRIPLET_PARAMS_LEN = 12 #main.py TRIPLET --num_iter=<num_iteraction> --norm_mode=<norm_mode> --split_mode=<split_mode> --num_neighbors=<num_neighbors> --num_epochs=<num_epochs> --num_shots=<num_shoots> --margin=<margin> --weights=<weights> --k_groups=<k_groups> --is_improved=<is_improved>

if __name__ == '__main__':
    params = sys.argv
    
    if '-h' in params or '--help' in params:
        printHelp()
        exit(0)

    model = params[1].upper()
    config = getConfigs(params[2:])
    num_iter = config[0]
    config = config[1:]
    plot = PlotManager()
    global_evaluation_metrics_by_run = []
    
    for iter in range(num_iter):
        print('\nIteration {}\n' .format(iter+1))
        manager = taskManager(model, config, plot)
        manager.runClassifierBasedOnModel()
        global_evaluation_metrics_by_run += manager.local_evaluation_metrics_by_run
        plot = manager.plot
        printError(manager.error)
         
    manager.plot.saveAllImages()
    print(global_evaluation_metrics_by_run)
    analyser = StatisticalAnalyzer(manager.classifier.categories_name, manager.classifier.metric_names, global_evaluation_metrics_by_run)
    analyser.printAnalysis()
    print('Done!')
    exit(0)