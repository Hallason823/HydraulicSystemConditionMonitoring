# Results History - Hydraulic System Condition Monitoring

## Experimental Setup

### Data Split

| Model   | Support Set | Test Set | Train + Validation | Validation |
|---------|-------------|----------|-------------------|------------|
| DTW     | 10.00%      | 20.00%   | -                 | Hold-out   |
| AE      | -           | 20.00%   | 80.00%            | 8-fold     |
| SIAMESE | 13.60%      | 20.00%   | 66.40%            | 8-fold     |
| TRIPLET | 13.60%      | 20.00%   | 66.40%            | 8-fold     |

### Model Configuration

**DTW:**
- Number of neighbors: 3 (fixed)
- Classifier: Weighted combination
- Validation: Hold-out

**Autoencoder (AE):**
- Epochs: 50
- Activation function: Leaky ReLU
- Optimizer: Adam
- Loss: Mean Square Error
- Classifier: Weighted combination
- Validation: 8-fold cross-validation

**Siamese Network:**
- Epochs: 100
- Margin (α): 0.0075
- Distance: Euclidean
- Classifier: Adapted KNN
- Validation: 8-fold cross-validation

**Triplet Network:**
- Epochs: 100
- Margin (α): 0.0075
- Distance: Euclidean
- Classifier: Adapted KNN
- Validation: 8-fold cross-validation

## Detailed Results

### Cooling Condition

Results in format: mean | max

| Model      | Normalization | N. Neighbors | Accuracy (%) | F1-score (%) |
|------------|---------------|--------------|--------------|--------------|
| **DTW**    | max-min       | 3            | 100.00 \| 100.00 | 100.00 \| 100.00 |
|            | [-1, 1]       | 3            | 97.78 \| 97.78   | 97.78 \| 97.78   |
|            | dec. scaling  | 3            | 100.00 \| 100.00 | 100.00 \| 100.00 |
|            | z-score       | 3            | 100.00 \| 100.00 | 100.00 \| 100.00 |
| **AE**     | max-min       | 5            | 99.35 \| 100.00  | 99.35 \| 100.00  |
|            | [-1, 1]       | 3            | 98.50 \| 99.55   | 98.50 \| 99.55   |
|            | dec. scaling  | 4            | 99.06 \| 100.00  | 99.06 \| 100.00  |
|            | z-score       | 3            | 99.69 \| 99.77   | 99.69 \| 99.77   |
| **SIAMESE**| max-min       | 3            | 99.46 \| 100.00  | 99.46 \| 100.00  |
|            | [-1, 1]       | 2            | 98.98 \| 100.00  | 98.98 \| 100.00  |
|            | dec. scaling  | 4            | 96.58 \| 97.72   | 96.56 \| 97.71   |
|            | z-score       | 1            | 99.84 \| 100.00  | 99.84 \| 100.00  |
| **TRIPLET**| max-min       | 2            | 99.10 \| 100.00  | 99.10 \| 100.00  |
|            | [-1, 1]       | 3            | 98.90 \| 100.00  | 98.90 \| 100.00  |
|            | dec. scaling  | 3            | 99.02 \| 99.67   | 99.02 \| 99.67   |
|            | z-score       | 2            | 99.47 \| 100.00  | 99.47 \| 100.00  |

### Stable Flag

Results in format: mean | max

| Model      | Normalization | N. Neighbors | Accuracy (%) | F1-score (%) |
|------------|---------------|--------------|--------------|--------------|
| **DTW**    | max-min       | 3            | 82.22 \| 82.22 | 82.39 \| 82.39 |
|            | [-1, 1]       | 3            | 73.33 \| 73.33 | 73.33 \| 73.33 |
|            | dec. scaling  | 3            | 93.33 \| 93.33 | 93.27 \| 93.27 |
|            | z-score       | 3            | 95.56 \| 95.56 | 95.56 \| 95.56 |
| **AE**     | max-min       | 5            | 86.28 \| 90.48 | 86.34 \| 90.51 |
|            | [-1, 1]       | 3            | 89.40 \| 95.01 | 89.37 \| 94.98 |
|            | dec. scaling  | 4            | 95.27 \| 97.51 | 95.24 \| 97.49 |
|            | z-score       | 3            | 96.29 \| 97.73 | 96.26 \| 97.72 |
| **SIAMESE**| max-min       | 3            | 85.20 \| 88.66 | 85.45 \| 88.76 |
|            | [-1, 1]       | 2            | 80.09 \| 84.04 | 80.35 \| 84.13 |
|            | dec. scaling  | 4            | 84.36 \| 89.58 | 84.79 \| 89.73 |
|            | z-score       | 1            | 91.40 \| 92.86 | 90.97 \| 92.60 |
| **TRIPLET**| max-min       | 2            | 88.31 \| 90.88 | 88.52 \| 91.03 |
|            | [-1, 1]       | 3            | 78.99 \| 86.64 | 79.29 \| 86.76 |
|            | dec. scaling  | 3            | 88.60 \| 90.88 | 88.69 \| 90.85 |
|            | z-score       | 2            | 90.11 \| 92.83 | 90.29 \| 92.93 |

## Comparative Analysis

### Cooling Condition

**Best overall performance:**
- **SIAMESE + Z-score**: 99.84% mean accuracy (1 neighbor)
- **AE + Z-score**: 99.69% mean accuracy (3 neighbors)
- **DTW**: 100% accuracy in 3 out of 4 normalizations (but O(n²) complexity)

**Observations:**
- All ML models achieved >96% accuracy
- Z-score normalization was the best across all ML models
- DTW showed perfect results but is computationally expensive

### Stable Flag

**Best overall performance:**
- **AE + Z-score**: 96.29% mean accuracy (3 neighbors)
- **DTW + Z-score**: 95.56% accuracy
- **SIAMESE + Z-score**: 91.40% mean accuracy (1 neighbor)

**Observations:**
- All ML models achieved >78% accuracy
- AE outperformed DTW in mean performance
- Decimal scaling also showed good results with AE

## Discussion

### Parameter Influence

1. **Number of neighbors**: The optimal number varies by model and normalization
2. **Normalization**: Z-score consistently yielded the best results
3. **Model**: AE demonstrated better generalization across 3 out of 4 normalizations

### Complexity Comparison

| Model   | Complexity | Execution Time | Suitable for Real-time |
|---------|------------|----------------|------------------------|
| DTW     | O(n²)      | High           | No                     |
| AE      | O(n)       | Low            | Yes                    |
| SIAMESE | O(n)       | Low            | Yes                    |
| TRIPLET | O(n)       | Low            | Yes                    |

### Performance vs Complexity

**DTW:**
- High accuracy (100% on cooling condition)
- O(n²) complexity infeasible for real-time industrial applications
- Best with max-min, decimal scaling, and z-score normalizations

**Autoencoder (AE):**
- Excellent accuracy/efficiency balance
- Better generalization across multiple normalizations
- O(n) complexity after training
- **Recommended for industrial applications**

**Siamese Network:**
- Best performance on cooling condition (99.84%)
- Good performance with limited datasets
- O(n) complexity after training

**Triplet Network:**
- Performance comparable to Siamese
- Addresses boundary sample problems
- O(n) complexity after training

## Conclusions

1. **Best overall model**: AE with z-score normalization
   - Superior generalization capability
   - Low computational complexity
   - Suitable for real-time industrial applications

2. **Best normalization**: Z-score
   - Best results across all ML models
   - Consistency across different metrics

3. **Trade-off**: DTW offers high accuracy but is impractical for fast decisions

4. **Practical application**: AE is recommended for industrial monitoring considering:
   - >99% accuracy on cooling condition
   - >96% accuracy on stable flag
   - O(n) complexity enabling real-time processing

## Metrics Used

- **Accuracy**: Overall correct prediction rate
- **Precision**: Accuracy of positive predictions
- **Recall**: True positive detection rate
- **F1-score**: Harmonic mean of precision and recall
