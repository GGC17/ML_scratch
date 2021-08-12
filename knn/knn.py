import numpy as np
import pandas as pd
from collections import Counter
from sklearn.datasets import load_iris

def euclidean_distance(x1, x2):
    
    return np.sqrt(np.sum(x1-x2)**2)

def accuracy(y_true, y_pred):
    
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    
    return accuracy

class KNN:

    def __init__(self, k=5):

        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        
        y_pred = [self.compute_distances(x1) for x1 in X]
    
        return np.array(y_pred)

    def compute_distances(self, x1):
        
        distances = [euclidean_distance(x1, x2) for x2 in self.X_train]

        ki = np.argsort(distances)[:self.k]
    
        k_labels = [self.y_train[i] for i in ki]
        
        most_common = Counter(k_labels).most_common(1)
        
        return most_common[0][0]


iris = load_iris()
iris_df = pd.DataFrame(iris['data'], columns=iris['feature_names'])

iris_df['label'] = iris['target']

# Shuffle Data
df = iris_df.sample(frac=1)

# Slice
df_train = df.iloc[0:120, :]
df_test = df.iloc[120:, :]

# Split
X_train = np.array(df_train.iloc[:, :-1])
X_test = np.array(df_test.iloc[:, :-1])
y_train = np.array(df_train.iloc[:, -1])
y_test = np.array(df_test.iloc[:, -1])

k = 3
knn = KNN(k=k)
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)
print("KNN classification accuracy", accuracy(y_test, predictions))