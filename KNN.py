import numpy as np
import pandas as pd
 
def euclidean_dist(x1, x2):
    dist = np.sqrt(np.sum((x1-x2)**2))
    return dist

class KNN:
    def __init__(self, k=3):
        self.k = k
 
    def fit(self, X, y):
        #train model on training data
        self.X_train = X
        self.y_train = y
 
    def predict(self, X):
        X = np.array(X) if isinstance(X, pd.DataFrame) else X #convert to numpy array if it's not one
        predictions = [self._predict(x) for x in X] #apply _predict to each datapoint in X_train
        return predictions
    
    #helper method 
    def _predict(self, x):
        # compute euclidean distance for each data point in the test data against each data point in the training data
        distances = [euclidean_dist(x, x_train) for x_train in self.X_train]
    
        # get ascending sorted indices of smallest distances, selecting the indices of the first k's.
        nearest_neighbours_indices = np.argsort(distances)[:self.k]
        #obtain class labels 
        nearest_neighbours_labels = [self.y_train[i] for i in nearest_neighbours_indices]
        
        #find the most common class label
        label_counts = {}
        for label in nearest_neighbours_labels:
            if label in label_counts:
                label_counts[label] += 1
            else:
                label_counts[label] = 1
        
        #find the most count
        most_common_label = max(label_counts, key = label_counts.get)

        #distances for nearest neighbours
        k_nearest_distances = [distances[i] for i in nearest_neighbours_indices]
        
        return most_common_label, k_nearest_distances