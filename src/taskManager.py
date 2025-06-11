from config.config import *
from data.dataManager import *
from data.siamesePairDataset import *
from data.tripletPairDataset import *
from models.evaluators import *
from classifier.classifier import *
from plot.plot import *

class taskManager:
    def __init__(self, model, config, plot):
        self.model = model
        self.config = config
        self.plot = plot
        self.iter_limit = 1 if self.config[1] == 'Hold-out' else self.config[7]
        self.local_evaluation_metrics_by_run = []
        self.error = False
        self.initializeDatasetBasedOnModel()

    def initializeDatasetBasedOnModel(self):
        if self.error == False:
            if self.model == 'DTW':
                self.ds = DataManager(*FILES_NAME, normalization_mode=self.config[0], split_mode=self.config[1], weights=self.config[6], k_groups=self.config[7])
            elif self.model == 'AE':
                self.ds = DataManager(*FILES_NAME, normalization_mode=self.config[0], split_mode=self.config[1], weights=self.config[6], k_groups=self.config[7])
            elif self.model == 'SIAMESE':
                self.ds = SiamesePairDataset(*FILES_NAME, normalization_mode=self.config[0], split_mode=self.config[1], shots_number=self.config[4], weights=self.config[6], k_groups=self.config[7])
            elif self.model == 'TRIPLET':
                self.ds = TripletPairDataset(*FILES_NAME, normalization_mode=self.config[0], split_mode=self.config[1], shots_number=self.config[4], weights=self.config[6], k_groups=self.config[7])
            else:
                self.error = True
        
    def initializeClassifierBasedOnModel(self):
        if self.model in {'DTW', 'AE', 'SIAMESE', 'TRIPLET'} and self.error == False:
            if self.model == 'DTW':
                self.classifier = Classifier(0, self.ds.normalized_test_instances, self.ds.test_targets, self.ds.normalized_train_instances,
                                            self.ds.train_targets, T_neighbors=self.config[2]
                                            )
            elif self.model == 'AE':
                self.evaluator = EvaluatorNetworks(0, DEVICE, self.ds.normalized_train_instances, self.ds.normalized_val_instances, n_epochs=self.config[3])
                self.plot.plotLosses(self.evaluator.history, 'MSE Loss')
                self.classifier = Classifier(1, self.evaluator.evaluateSamplesSet(self.ds.normalized_test_instances), self.ds.test_targets,
                                            self.evaluator.evaluateSamplesSet(self.ds.normalized_train_instances+self.ds.normalized_val_instances),
                                            self.ds.train_targets+self.ds.val_targets, T_neighbors=self.config[2]
                                            )
            elif self.model == 'SIAMESE':
                self.evaluator = EvaluatorNetworks(1, DEVICE, self.ds, self.ds, n_epochs=self.config[3], margin=self.config[5])
                self.plot.plotLosses(self.evaluator.history, 'Contrastive Loss')
                self.classifier = Classifier(2, self.evaluator.evaluateSamplesSet(self.ds.normalized_test_instances), self.ds.test_targets,
                                            self.evaluator.evaluateSamplesSet(self.ds.normalized_support_set_instances), self.ds.support_set_targets,
                                            self.ds.categories, shots_number=self.config[4], T_neighbors=self.config[2]
                                            )
            else:
                self.evaluator = EvaluatorNetworks(2, DEVICE, self.ds, self.ds, n_epochs=self.config[3], margin=self.config[5], is_improved=self.config[8])
                self.plot.plotLosses(self.evaluator.history, 'Triplet Loss')
                self.classifier = Classifier(2, self.evaluator.evaluateSamplesSet(self.ds.normalized_test_instances), self.ds.test_targets,
                                            self.evaluator.evaluateSamplesSet(self.ds.normalized_support_set_instances), self.ds.support_set_targets,
                                            self.ds.categories, shots_number=self.config[4], T_neighbors=self.config[2]
                                            )
            self.plot.plotResults(self.classifier.categories_name, self.classifier.estimated_conditions, self.classifier.evaluated_targets)
            #self.plot.plotConfusionMatrix(self.classifier.categories_name, self.classifier.values_by_category, self.classifier.estimated_conditions, self.classifier.evaluated_targets)
            self.local_evaluation_metrics_by_run += [self.classifier.evaluation_metrics]
        else:
            self.error = True
            
    def runClassifierBasedOnModel(self):
        if self.error == False:
            if (self.config[1] == 'Hold-out') or (self.config[1] == 'K-fold cross-validation' and self.ds.k_val == self.iter_limit):
                self.initializeClassifierBasedOnModel()
            elif self.config[1] == 'K-fold cross-validation' and self.ds.k_val < self.iter_limit:
                self.initializeClassifierBasedOnModel()
                self.ds.updateTheDistributionFolds()
                self.ds.buildDatasPartition()
                self.runClassifierBasedOnModel()
            else:
                self.error = True