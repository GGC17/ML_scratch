import numpy as np
import pandas as pd



class LogisticRegression:
    def __init__(self, alpha, epochs):
        self.alpha = alpha
        self.epochs = epochs
        self.w = None
        self.b = None

    def fit(self, X, y):

        m = len(y)

        self.w = np.zeros(X_train.shape[1])
        self.b = 0

        for epoch in range(self.epochs):

            z = X.dot(self.w) + self.b
            loss = self.sigmoid(z) - y

            dw = X.T.dot(loss) / m
            db = np.sum(loss) / m

            self.w -= self.alpha * dw
            self.b -= self.alpha * db

    def predict(self, X, threshold=0.5):

        predictions = X.dot(self.w) + self.b
    
        y_pred = self.sigmoid(predictions)
        
        y_pred_class = [1 if i > threshold else 0 for i in y_pred]
        
        return np.array(y_pred_class)

    def sigmoid(self, z):
    
        sigmoid = 1 / (1 + np.exp(-z))
        
        return sigmoid


df = pd.read_csv('university_admissions.csv')
df = df.sample(frac=1) # Shuffle Data

# Slice
df_train = df.iloc[0:1232, :]
df_test = df.iloc[1232:, :]

# Split
X_train = df_train.iloc[:, :-1]
X_test = df_test.iloc[:, :-1]
y_train = df_train.iloc[:, -1]
y_test = df_test.iloc[:, -1]

def accuracy(y_test, y_pred_class):
    
    accuracy = np.sum(y_test == y_pred_class) / len(y_test)
    
    return accuracy 


cls = LogisticRegression(alpha=0.01, epochs=10000)
cls.fit(X_train, y_train)
predictions = cls.predict(X_test)

accuracy = accuracy(y_test, predictions)
print("Accuracy: ", accuracy)
