import torch
from torch.utils.data import TensorDataset
import numpy as np

class DataManager:
    def __init__(self, instance_file_names, target_file_name, file_type = '.txt', folder_path='../data/raw/', selected_features=[0, 4], normalization_mode='global max-min norm', split_mode = 'Hold-out', shots_number=200, weights=[0.7, 0.1], k_groups=8, support_set_flag=False):
        self.instance_file_names = instance_file_names
        self.target_file_name = target_file_name
        self.file_type = file_type
        self.folder_path = folder_path
        self.selected_features = selected_features
        self.normalization_mode = normalization_mode
        self.split_mode = split_mode
        self.shots_number = shots_number
        self.weights = weights
        self.k_groups = k_groups
        self.support_set_flag = support_set_flag
        self.k_val = None
        self.loadDatasfromFiles()
        self.findAllCategories()
        self.processAllDatas()
        
    def loadDatasfromSingleFile(self, file_name):
        with open(self.folder_path+file_name+self.file_type, 'r') as file:
            rows = file.readlines()
            datas = [list(map(float, row.strip().split())) for row in rows]
        return datas
    
    def loadAllInstancesfromFiles(self):
        self.all_instances = []
        for instance_file_name in self.instance_file_names:
            self.all_instances.append(self.loadDatasfromSingleFile(instance_file_name)) 
            
    def convertAllInstancesToInstancesbySample(self):
        self.loadAllInstancesfromFiles()
        self.samples_number = len(self.all_instances[0])
        self.instances_by_sample = [torch.tensor([instance[sample] for instance in self.all_instances]) for sample in range(self.samples_number)]
    
    def selectFeaturesTargets(self, targets):
        return [[target[idx] for idx in self.selected_features] for target in targets]
    
    def loadDatasfromFiles(self):
        self.convertAllInstancesToInstancesbySample()
        self.target_by_sample = self.selectFeaturesTargets(self.loadDatasfromSingleFile(self.target_file_name))
    
    def findAllCategories(self):
        single_categories = list(set(tuple(target) for target in self.target_by_sample))
        self.categories = [list(category) for category in single_categories]

    def createSupportSet(self):
        shuffle_idx = list(range(self.samples_number))
        np.random.shuffle(shuffle_idx)
        shuffle_targets = [self.target_by_sample[idx] for idx in shuffle_idx]
        selected_idx = []
        for category in self.categories:
            specific_category_idx = []
            for idx, target in enumerate(shuffle_targets):
                if target == category:
                    specific_category_idx.append(idx)
                    if len(specific_category_idx) == self.shots_number:
                        break
            selected_idx += specific_category_idx
            non_selected_idx = list(set(shuffle_idx)-set(selected_idx))
        shuffle_instances_by_sample = [self.instances_by_sample[idx] for idx in shuffle_idx]
        shuffle_targets_by_sample = [self.target_by_sample[idx] for idx in shuffle_idx]
        self.support_set_instances = [shuffle_instances_by_sample[idx] for idx in selected_idx]
        self.support_set_targets = [shuffle_targets_by_sample[idx] for idx in selected_idx]
        self.non_support_instances = [shuffle_instances_by_sample[idx] for idx in non_selected_idx]
        self.non_support_targets = [shuffle_targets_by_sample[idx] for idx in non_selected_idx]
    
    def updateTheDistributionFolds(self):
        self.val_idx = self.folds_idx[self.k_val]
        self.train_idx = torch.cat([self.folds_idx[i] for i in range(self.k_groups) if i != self.k_val], dim=0)
        self.k_val += 1
    
    def getSamplesbyIdx(self):
        set_splits = {"train": self.train_idx,"val": self.val_idx,"test": self.test_idx}
        for split in set_splits:
            setattr(self, f"{split}_instances", [self.non_support_instances[idx] for idx in set_splits[split]])
            setattr(self, f"{split}_targets", [self.non_support_targets[idx] for idx in set_splits[split]])
    
    def splitSamples(self):
        length = len(self.non_support_instances)
        idx = list(range(length))
        np.random.shuffle(idx)
        if self.split_mode == 'Hold-out':
            samples_lengths = [int(np.floor(sum(self.weights[:i+1])*length)) for i in range(len(self.weights))]
            self.train_idx, self.val_idx, self.test_idx = torch.tensor(idx[:samples_lengths[0]]), torch.tensor(idx[samples_lengths[0]:samples_lengths[1]]), torch.tensor(idx[samples_lengths[1]:]) 
        elif self.split_mode == 'K-fold cross-validation':
            self.k_val = 0
            test_set_length = int((1-sum(self.weights))*length)
            val_train_set_length = length-test_set_length
            samples_lengths = [test_set_length+int(np.floor(i/self.k_groups*val_train_set_length)) for i in range(self.k_groups+1)]
            self.test_idx = torch.tensor(idx[:test_set_length])
            self.folds_idx = [torch.tensor(idx[samples_lengths[i]:samples_lengths[i+1]]) for i in range(self.k_groups)]
            self.updateTheDistributionFolds()
        else:
            print('\nInvalidy split mode!\n')
            
    def normalizeDatasSeparatedbySample(self, datas):
        if self.normalization_mode == 'global max-min norm' or self.normalization_mode == 'global [-1, 1] norm':
            min_values_by_unit = [torch.min(data, dim=1).values for data in datas]
            max_values_by_unit = [torch.max(data, dim=1).values for data in datas]
            min_values = torch.min(torch.stack(min_values_by_unit), dim=0).values.unsqueeze(0).transpose(0, 1).repeat(1, len(datas[0][0])) 
            max_values = torch.max(torch.stack(max_values_by_unit), dim=0).values.unsqueeze(0).transpose(0, 1).repeat(1, len(datas[0][0]))
            normalized_datas = [(data - min_values) / (max_values - min_values) for data in datas] if self.normalization_mode == 'global max-min norm' else [2*(data - min_values) / (max_values - min_values)-1 for data in datas]
        elif self.normalization_mode == 'local decimal scaling norm':
            normalized_datas = [data/torch.pow(10, torch.max(torch.ceil(torch.log10(torch.abs(data))), dim=1, keepdim=True)[0]) for data in datas]
        elif self.normalization_mode == 'local z-score norm':
            normalized_datas = [(data - torch.mean(data)) / torch.std(data) for data in datas]
        else:
            print("\nInvalidy normalization mode!\n")
        return normalized_datas
    
    def normalizeInstancesofSupportSet(self):
        self.normalized_support_set_instances = self.normalizeDatasSeparatedbySample(self.support_set_instances)
        
    def normalizeAllInstancesSeparatedbySample(self):
        self.normalized_train_instances = self.normalizeDatasSeparatedbySample(self.train_instances)
        self.normalized_val_instances = self.normalizeDatasSeparatedbySample(self.val_instances)
        self.normalized_test_instances = self.normalizeDatasSeparatedbySample(self.test_instances)
    
    def buildSingleTensorDataset(self, instances, targets):
        return TensorDataset(torch.stack([sample.unsqueeze(0) for sample in instances], dim=0), torch.tensor(targets))
    
    def buildAllTensorDataset(self):
        self.single_sample_train_ds = self.buildSingleTensorDataset(self.normalized_train_instances, self.train_targets)
        self.single_sample_val_ds = self.buildSingleTensorDataset(self.normalized_val_instances, self.val_targets)
        self.single_sample_test_ds = self.buildSingleTensorDataset(self.normalized_test_instances, self.test_targets)
    
    def buildDatasPartition(self):
        self.getSamplesbyIdx()
        self.normalizeAllInstancesSeparatedbySample()
        self.buildAllTensorDataset()
        
    def processAllDatas(self):
        if self.support_set_flag:
            self.createSupportSet()
            self.normalizeInstancesofSupportSet()
        else:
            self.non_support_instances = self.instances_by_sample
            self.non_support_targets = self.target_by_sample
        self.splitSamples()
        self.buildDatasPartition()

    def setRefDatasetBasedOnFlagMode(self, is_train_ds):
        self.is_train_ds = is_train_ds
        self.ref_ds = self.single_sample_train_ds if self.is_train_ds else self.single_sample_val_ds