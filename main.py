from os import listdir
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# hyperparameters
nearest_N = 15      # for knn function
n_folds = 50        # for k-fold cross validation

files = ['test.csv', 'train.csv']



def KNN_Euclid(X_test, X_train, y_train, k=1):
    """outputs a class membership, determined by a plurality vote of its k nearest neighbors determined using the Euclidean norm"""
    
    k_nearest = [] # x_ij denotes the euclidean distance of the i'th test to the j'th training example
    for test in X_test:
        distances = []
        for train in X_train:
            distances.append(np.linalg.norm(test-train))
        
        nearest_indices = np.argsort(distances)[:k]
        labels = [y_train[i][1] for i in nearest_indices]
        k_nearest.append(max(set(labels), key = labels.count))

    return k_nearest

# 1. Preprocessing

# import some data
movement_data = pd.concat([pd.read_csv(f) for f in files])
# partition the row indices
index_partitions = np.array_split(np.random.permutation(movement_data.index), n_folds)
folds = [movement_data.iloc[partition,:] for partition in index_partitions]

# determine output file

# 2. K-fold cross valid
for k in range(1):
    # split data for k-fold cross-validation
    data = folds.copy()
    test_set = data.pop(k).values
    train_set = pd.concat([fold for fold in data]).values

    X_test = test_set[:,:-2]
    X_train = train_set[:,:-2]
    y_train = train_set[:,-2:]

    y_true = test_set[:,-1]
    y_pred = KNN_Euclid(X_test, X_train, y_train,k=nearest_N)
    np.save('data', [y_true, y_pred])
