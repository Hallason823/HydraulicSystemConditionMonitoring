import random
from torch.utils.data import Dataset
from .dataManager import *

class TripletPairDataset(Dataset, DataManager):
    def __init__(self, instance_file_names, target_file_name, file_type = '.txt', folder_path='../data/raw/', selected_features=[0, 4], normalization_mode='global max-min norm', split_mode = 'Hold-out', shots_number=200, weights=[0.7, 0.1], k_groups=8, support_set_flag=True, is_train_ds=True):
        DataManager.__init__(self, instance_file_names, target_file_name, file_type, folder_path, selected_features, normalization_mode, split_mode, shots_number, weights, k_groups, support_set_flag)
        self.setRefDatasetBasedOnFlagMode(is_train_ds)
        
    def __getitem__(self, index):
        anchor = random.choice(self.ref_ds)
        sample = random.choice(self.ref_ds)
        if anchor[-1].tolist() == sample[-1].tolist():
            positive = sample
            while True:
                sample = random.choice(self.ref_ds)
                if anchor[-1].tolist() != sample[-1].tolist():
                    negative = sample
                    break
        else:
            negative = sample
            while True:
                sample = random.choice(self.ref_ds)
                if anchor[-1].tolist() == sample[-1].tolist():
                    positive = sample
                    break
        return anchor[0], positive[0], negative[0]
    
    def __len__(self):
        return len(self.ref_ds)