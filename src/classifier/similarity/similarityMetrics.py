import torch
import math

class SimilarityMetrics:
    def __init__(self, evaluated_samples, ref_samples, similarity_mode):
        self.evaluated_samples = evaluated_samples
        self.ref_samples = ref_samples
        self.similarity_mode = similarity_mode
        self.computeSimilarities()
    
    def computeEuclideanDistance(self, evaluated_sample, ref_sample):
        return torch.dist(evaluated_sample, ref_sample)
    
    def getSimilarityBasedOnEuclideanDistance(self, evaluated_sample, ref_sample):
        return 1/(1+self.computeEuclideanDistance(evaluated_sample, ref_sample))
    
    def initializeDistanceMatrixAndTaketheParamNumber(self):
        self.param_number = len(self.evaluated_samples[0])
        self.N = self.evaluated_samples[0][0].shape[0]
        self.M = self.ref_samples[0][0].shape[0]
        self.distance_matrix = torch.zeros((self.N,self. M))
    
    def initializeCostMatrix(self):
        self.cost_matrix = torch.zeros((self.N+1, self.M+1))
        self.cost_matrix[1:, 0] = float('inf')
        self.cost_matrix[0, 1:] = float('inf')
        
    def computeDistanceMatrixBetweenSamples(self, evaluated_sample, ref_sample):
        for i in range(self.N):
            for j in range(self.M):
                self.distance_matrix[i][j] = math.sqrt(sum([(evaluated_sample[idx][i]-ref_sample[idx][j])**2 for idx in range(self.param_number)]))
                
    def getSimlaritybyDTWBetweenSamples(self, evaluated_sample, ref_sample):
        self.computeDistanceMatrixBetweenSamples(evaluated_sample, ref_sample)
        self.initializeCostMatrix()
        for i in range(self.N):
            for j in range(self.M):
                penalty = [self.cost_matrix[i, j], self.cost_matrix[i, j+1], self.cost_matrix[i+1, j]]
                penalty_min_idx = torch.argmin(torch.tensor(penalty))
                self.cost_matrix[i+1, j+1] = self.distance_matrix[i, j] + penalty[penalty_min_idx]
        print(self.cost_matrix[self.N][self.M])
        return self.cost_matrix[self.N][self.M]+0.0001
    
    def computeSimilarities(self):
        self.all_similarities = []
        if self.similarity_mode == 0:
            self.initializeDistanceMatrixAndTaketheParamNumber()
        for evaluated_sample in self.evaluated_samples:
            similarities_by_evaluated_sample = []
            for ref_sample in self.ref_samples:
                if self.similarity_mode == 0:
                    similarities_by_evaluated_sample.append(self.getSimlaritybyDTWBetweenSamples(evaluated_sample, ref_sample))
                elif self.similarity_mode == 1:
                    similarities_by_evaluated_sample.append(self.getSimilarityBasedOnEuclideanDistance(evaluated_sample, ref_sample))
                elif self.similarity_mode == 2:
                    similarities_by_evaluated_sample.append(self.computeEuclideanDistance(evaluated_sample, ref_sample))
                else:
                    print("\nInvalidy similaritty mode!\n")
            self.all_similarities.append(similarities_by_evaluated_sample)