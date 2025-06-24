# implementing_KNN
This project implements the K-nearest neighbours algorithm from scratch to help classify the different wines into their respective 3 classes. 

# Dataset

The wine dataset was retrieved from the UCI Machine Learning Repository. It comprises 178 instances divided into 3 classes, with 59, 71, and 48 instances, respectively. Each instance consists of 13 attributes: Alcohol, Malic acid, Ash, Alcalinity of ash, Magnesium, Phenols, Flavanoids, Nonflavanoid phenols, Proanthocyanins, Color intensity, Hue, OD (OD280/OD315 of diluted wines), and Proline.

# Distance metric
The Euclidean distance is used to measure the distance between points.

# Feature scaling
Min-max normalisation is applied to each feature in the training set. This ensures consistency as it prevents features with large scale from dominating, and doing it only on the training set prevents data leakage.

# Input
The program takes four arguments as command line arguments: 
1. file name for training data
2. file name for testing data
3. file name for the output file
4. k (the number of neighbours)

# Output
The program outputs the test accuracy and creates a file containing the following information for each instance in the test file:
1. The original class label from the test data
2. The predicted class label according to the kNN implementation
3. The distance between the test instance and each of the neighbours 

# Analysis & Discussion
The classification accuracy on the training set and test set using k=1 and k=3 were analysed.

Classification accuracy on train set using k=1: 100%
Classification accuracy on test set using k=1: 94.444%

Classification accuracy on train set using k=3: 97.887%
Classification accuracy on test set using k=3: 94.444%

## K-value of 1
a k-value of 1 generally maximises the accuracy in the training set as the closest point to any training point is itself. However, this optimal k-value does not improve accuracy in the test set due to the lack of generalisation. 

If the decision boundary between classes is not clearly defined or if the training data has outliers, k=1 might lead to an incorrect classification, reducing the test set accuracy.

## K-value of 3
As k increases, the algorithm considers more neighbours, which leads to less overfitting and thus improving test accuracy as the algorithm becomes more robust to outliers and noise.

## Bias-variance trade-off

Small values of k leads to a model that follows the training data closely, resulting in low bias. However, the algorithm becomes more sensitive to noise and biases in the training data, resulting in high variance. Thus, we typically see high accuracy on the training data but lower accuracy on the test data as the algorithm fails to generalise to unseen data.

As k increases, the algorithm considers more neighbours, which leads to less overfitting and thus improving test accuracy as the algorithm is more robust to outliers and noise. However, large value of k will result in a model that is too general as it fails to capture relevant patterns. Furthermore, the training accuracy might decrease because the model is on longer fitting the training data as closely as when k is small.

# Future work

## Different distance metrics
Future work could look into different distance metric such as Manhattan Distance, Minkowski distance, and Hamming distance, and in which scenario does the different distance metric works best. 

## Cross-validation
Future work could implement cross-validation as a technique for choosing k, as well as choosing odd k's to prevent ties.
