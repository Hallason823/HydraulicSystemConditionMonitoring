import numpy as np
from scipy.stats import norm

class StatisticalAnalyzer:
    def __init__(self, categories_name, metric_names, evaluation_metrics):
        self.categories_name = categories_name
        self.metric_names = metric_names
        self.evaluation_metrics = evaluation_metrics
        self.calculaterStatisticalParam()
        
    def separateAccuracybyCategory(self):
        self.evaluation_metrics_by_category = {}
        for metric_name in self.metric_names:
            single_metric = [metrics_by_run[metric_name] for metrics_by_run in self.evaluation_metrics]
            single_metric_by_category = {}
            for category_name in self.categories_name:
                single_metric_by_category[category_name] = np.array([metric_by_run[category_name] for metric_by_run in single_metric])
            self.evaluation_metrics_by_category[metric_name] = single_metric_by_category
                
    def calculaterStatisticalParam(self):
        self.separateAccuracybyCategory()
        self.statistical_param_names = ['mean', 'std', 'max']
        self.statistical_param_funcs = [np.mean, np.std, np.max]
        self.statistical_params = {}
        for metric_name in self.metric_names:
            statistical_params_by_metric = {}
            for param_name, param_func in zip(self.statistical_param_names, self.statistical_param_funcs):
                statistical_params_by_metric[param_name] = [param_func(self.evaluation_metrics_by_category[metric_name][category_name]) for category_name in self.categories_name]
            self.statistical_params[metric_name] = statistical_params_by_metric

    def printAnalysis(self):
        for metric_name in self.metric_names:
            for idx, category_name in enumerate(self.categories_name):
                print("\nThe {} {}:" .format(category_name, metric_name))
                for param_name in self.statistical_param_names:
                    print("The {} is {:.2f}%." .format(param_name, self.statistical_params[metric_name][param_name][idx]))