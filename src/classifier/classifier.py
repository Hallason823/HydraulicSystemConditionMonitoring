import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from .similarity.similarityMetrics import *

class Classifier:
    def __init__(self, mode_idx, evaluated_samples, evaluated_targets, ref_samples, ref_targets, categories=None, categories_name=['Cooler Condition','Stable Flag'], shots_number=200, T_neighbors=3):
        self.mode_idx = mode_idx
        self.evaluated_samples = evaluated_samples
        self.evaluated_targets = evaluated_targets
        self.ref_samples = ref_samples
        self.ref_targets = ref_targets
        self.categories = categories
        self.categories_name = categories_name
        self.shots_number = shots_number
        self.T_neighbors = T_neighbors
        self.estimateConditionsforAnyMode()
        self.computeEvaluationMetricsbyCategory()
    
    def estimateConditionsbyWeightedCombination(self):
        for similarities_by_sample in self.similarities_by_samples:
            T_closest_neighbors = np.argpartition(similarities_by_sample, -self.T_neighbors)[-self.T_neighbors:] if self.is_reversed else np.argpartition(similarities_by_sample, self.T_neighbors)[:self.T_neighbors] 
            sum_T_closest_neighbors = sum([similarities_by_sample[idx] if self.is_reversed else 1/similarities_by_sample[idx] for idx in T_closest_neighbors]).item()
            estimated_conditions_by_closest = [((similarities_by_sample[idx].item() if self.is_reversed else 1/similarities_by_sample[idx].item())/(sum_T_closest_neighbors))*np.array(self.ref_targets[idx]) for idx in T_closest_neighbors]
            estimated_condition = [sum(estimated_condition_by_closest) for estimated_condition_by_closest in zip(*estimated_conditions_by_closest)]
            self.estimated_conditions.append(estimated_condition)
         
    def estimateConditionsbyAdaptedKNN(self):
        self.partitions = [i*self.shots_number for i in range(len(self.categories)+1)]
        for similarities_by_sample in self.similarities_by_samples:
            partition_of_similarities_by_sample = [similarities_by_sample[self.partitions[partition]:self.partitions[partition+1]] for partition in range(len(self.partitions)-1)]
            sorted_partition_of_similarities_by_samples = [sorted(partition_similarities, reverse=True) for partition_similarities in partition_of_similarities_by_sample]
            shoot_weights = [0]*len(self.categories)
            estimated_flag = False
            while estimated_flag == False:
                lower_similarities_by_partition = [partition_of_similarities[-1] for partition_of_similarities in sorted_partition_of_similarities_by_samples]
                smaller_similarity = min(lower_similarities_by_partition)
                partition_of_smaller_similarity = lower_similarities_by_partition.index(smaller_similarity)
                shoot_weights[partition_of_smaller_similarity] += 1
                changed_partition = sorted_partition_of_similarities_by_samples[partition_of_smaller_similarity]
                changed_partition.pop()
                sorted_partition_of_similarities_by_samples[partition_of_smaller_similarity] = changed_partition
                for idx, shoot_weight in enumerate(shoot_weights):
                    if shoot_weight >= self.T_neighbors:
                        self.estimated_conditions.append(self.categories[idx])
                        estimated_flag = True
        
    def obtainValuesbyCategory(self):
        all_values_by_categories = list(zip(*(self.ref_targets)))
        self.values_by_category = [list(set(all_values_by_category)) for all_values_by_category in all_values_by_categories]
        
    def normalizeSingleEstimation(self, param, possible_param_values):
        min_distance = float('inf')
        normalized_param_condition = None
        for value in possible_param_values:
            distance = abs(param-value)
            if distance < min_distance:
                min_distance = distance
                normalized_param_condition = value
        return normalized_param_condition
    
    def normalizeAllEstimations(self):
        self.obtainValuesbyCategory()
        normalized_estimated_conditions = []
        for estimated_condition in self.estimated_conditions:
            single_normalized_estimated_condition = []
            for estimated_param, possible_param_values in zip(estimated_condition, self.values_by_category):
                single_normalized_estimated_condition.append(self.normalizeSingleEstimation(estimated_param, possible_param_values))
            normalized_estimated_conditions.append(single_normalized_estimated_condition)
        self.estimated_conditions = normalized_estimated_conditions
    
    def estimateConditionsforAnyMode(self):
        self.categories_number = len(self.categories_name)
        self.estimated_conditions = []
        aux = SimilarityMetrics(self.evaluated_samples, self.ref_samples, self.mode_idx)
        self.similarities_by_samples = aux.all_similarities
        if self.mode_idx == 0 or self.mode_idx == 1:
            self.is_reversed = False if self.mode_idx == 0 else True
            self.estimateConditionsbyWeightedCombination()
            self.normalizeAllEstimations()
        elif self.mode_idx == 2:
            self.estimateConditionsbyAdaptedKNN()
        else:
            print("\nInvalidy mode index!\n")

    def computeEvaluationMetricsbyCategory(self):
        real_conditions_by_category = [[real_condition[idx] for real_condition in self.evaluated_targets] for idx in range(len(self.categories_name))]
        estimated_conditions_by_category = [[estimated_condition[idx] for estimated_condition in self.estimated_conditions] for idx in range(len(self.categories_name))]
        self.metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-score']
        self.metric_funcs = [accuracy_score, precision_score, recall_score, f1_score]
        self.evaluation_metrics = {}
        for metric_name, metric_func in zip(self.metric_names, self.metric_funcs):
            self.evaluation_metrics[metric_name] = {}
            for idx, category_name in enumerate(self.categories_name):
                if metric_name == 'Accuracy':
                    self.evaluation_metrics[metric_name][category_name] = 100 * metric_func(real_conditions_by_category[idx], estimated_conditions_by_category[idx])
                else:
                    self.evaluation_metrics[metric_name][category_name] = 100 * metric_func(real_conditions_by_category[idx], estimated_conditions_by_category[idx], average='weighted')