import torch
import numpy as np
import random
from torch.utils.data import Dataset
from .dataManager import *

class SiamesePairDataset(Dataset, DataManager):
    def __init__(self, instance_file_names, target_file_name, file_type = '.txt', folder_path='../data/raw/', selected_features=[0, 4], normalization_mode='global max-min norm', split_mode = 'Hold-out', shots_number=200, weights=[0.7, 0.1], k_groups=8, support_set_flag=True, is_train_ds=True):
        DataManager.__init__(self, instance_file_names, target_file_name, file_type, folder_path, selected_features, normalization_mode, split_mode, shots_number, weights, k_groups, support_set_flag)
        self.setRefDatasetBasedOnFlagMode(is_train_ds)
        
    def __getitem__(self, index):
        data0 = random.choice(self.ref_ds)
        should_get_same_class = random.randint(0,1) 
        if should_get_same_class:
            while True:
                data1 = random.choice(self.ref_ds) 
                if data0[-1].tolist() == data1[-1].tolist():
                    break
        else:
            while True:
                data1 = random.choice(self.ref_ds) 
                if data0[-1].tolist() != data1[-1].tolist():
                    break
        return data0[0], data1[0], torch.from_numpy(np.array([int(data1[-1].tolist() != data0[-1].tolist())], dtype=np.float32))
    
    def __len__(self):
        return len(self.ref_ds)  