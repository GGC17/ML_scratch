import numpy as np 
import pandas as pd 

def feat_scaling(X):
    
    mean = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    
    X_scale = (X-mean) / sigma
    
    return X_scale

def r2(predictions, y):
    
    ssr = np.sum((predictions - y)**2)
    sst = np.sum((y-y.mean())**2)
    
    r2 = 1 - (ssr/sst)
    
    return r2


class LinearRegression:
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

            h = X.dot(self.w) + self.b
            loss = h - y

            dw = X.T.dot(loss) / m
            db = np.sum(loss) / m

            self.w -= self.alpha * dw
            self.b -= self.alpha * db

    def predict(self, X):

        predictions = X.dot(self.w) + self.b

        return predictions


data = pd.read_excel('energy.xlsx')
data = data.sample(frac=1) # Shuffle Data
data = feat_scaling(data)

# Slice
data_train = data.iloc[0:7654, :]
data_test = data.iloc[7654:, :]

# Split
X_train = data_train.iloc[:, :-1]
X_test = data_test.iloc[:, :-1]
y_train = data_train.iloc[:, -1]
y_test = data_test.iloc[:, -1]

lr = LinearRegression(alpha=0.01, epochs=2000)
lr.fit(X_train, y_train)
predictions = lr.predict(X_test)

r2 = r2(predictions, y_test)
print("R2: ", r2)