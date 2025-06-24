import numpy as np
import pandas as pd
import sys
from KNN import KNN

def read_data_and_split(file):
    """
    Read a csv dataset, separate the features and class label

    Parameter:
    - file_name (str)

    Returns:
    - X: dataframe, features
    - y: series, class label
    """
    if not file.lower().endswith('.csv'):
        file += '.csv'

    df = pd.read_csv(f"data/{file}")
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    return X, y

train_file = input("Please enter the file name for training data: ")
test_file = input("Please enter the file name for testing data: ")
output_file = input("Please enter the file name for the output file: ")
k = int(input("Please enter the number of neighbours: "))

#construct output file names
output_train_file = f"{output_file}{k}_train.csv"
output_test_file = f"{output_file}{k}_test.csv"

#Read and split datasets
X_train, y_train = read_data_and_split(train_file)
X_test, y_test = read_data_and_split(test_file)
  
#Define class for min-max normalisation
#this will fit the scaler based on the training data only
#and use it to transform both training and testing datasets
class MinMaxNormalisation:
    def __init__(self):
        self.min = None
        self.max = None
        
    def fit(self, X):
        #compute min and max value for each feature in the training set
        self.min_ = np.min(X, axis=0) #column-wise operation
        self.max_ = np.max(X, axis=0)
    
    def normalise(self, X):
        #apply min-max normalisation
        return (X - self.min_) / (self.max_ - self.min_)
 
    def fit_and_normalise(self, X):
        self.fit(X)
        return self.normalise(X)
 
scaler = MinMaxNormalisation()
 
#Normalise training data and then transform the test
#data with the same parameters
X_train_scaled = scaler.fit_and_normalise(X_train)
X_test_scaled = scaler.normalise(X_test)
 
#apply KNN using X_train and y_train
clf = KNN(k)
 
#convert pandas dataframe or series into numpy array for computing euclidean distance 
def convert_to_array(data):
    if isinstance(data, (pd.DataFrame, pd.Series)):
        return data.to_numpy()
    else:
        raise TypeError("Input data must be a pandas Dataframe or Series")

X_train_scaled_np = convert_to_array(X_train_scaled)
y_train_np = convert_to_array(y_train)
X_test_scaled_np = convert_to_array(X_test_scaled)
y_test_np = convert_to_array(y_test)
 
clf.fit(X_train_scaled_np, y_train_np)
 
# store testing data predictions
test_output = []
for x, y_true in zip(X_test_scaled_np, y_test_np):
    predicted_label, distances = clf._predict(x)
    test_output.append([y_true, predicted_label] + distances)
 
#Convert to dataframe for CSV writing
test_output_df = pd.DataFrame(test_output, columns=['y', 'predicted_y'] + [f'distance{i+1}' for i in range(k)])

if not output_test_file.lower().endswith('.csv'):
    output_test_file += '.csv'
 
test_output_df.to_csv(output_test_file, index=False)
 
test_accuracy = np.sum(test_output_df['y'] == test_output_df['predicted_y']) / len(y_test_np)
print("Test Accuracy:", test_accuracy)

#store training data predictions
train_output = []
for x, y_true in zip(X_train_scaled_np, y_train_np):
    predicted_label, distances = clf._predict(x)
    train_output.append([y_true, predicted_label] + distances) 

train_output_df = pd.DataFrame(train_output, columns=['y', 'predicted_y'] + [f'distance{i+1}' for i in range(k)])

if not output_train_file.lower().endswith('.csv'):
    output_train_file += '.csv'
 
train_output_df.to_csv(output_train_file, index=False)

# Calculate training accuracy
train_accuracy = np.sum(train_output_df['y'] == train_output_df['predicted_y']) / len(y_train_np)
print("Training Accuracy:", train_accuracy)

