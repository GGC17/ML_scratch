import numpy as np
from tensorflow.keras import datasets

def load_process_data():

    (X_train, y_train), (X_test, y_test) = datasets.mnist.load_data()

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1]*X_train.shape[2])
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1]*X_test.shape[2])

    X_train = X_train.T
    X_test = X_test.T

    X_train = (X_train/255.0).astype('float32')
    X_test = (X_test/255.0).astype('float32')

    return X_train, X_test, y_train, y_test


 # Auxiliary Functions   
def ReLU(Z): # ReLU
    return np.maximum(0, Z)


def softmax(Z): # Softmax
    return np.exp(Z) / sum(np.exp(Z))


def one_hot(Y): # One Hot Encoding
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def deriv_ReLU(Z): #Derivative of ReLU for backprop
    return Z > 0


def get_accuracy(predictions, Y):

    accuracy = np.sum(predictions == Y) / Y.size
    
    return accuracy



class NeuralNetwork():

    def __init__(self, n_in, n_hidden_1, n_hidden_2, n_hidden_3, n_out):

        self.n_in = n_in
        self.n_hidden_1 = n_hidden_1
        self.n_hidden_2 = n_hidden_2
        self.n_hidden_3 = n_hidden_3
        self.n_out = n_out


        self.W1 = np.random.rand(self.n_hidden_1, self.n_in) - 0.5
        self.b1 = np.random.rand(self.n_hidden_1, 1) - 0.5
        
        self.W2 = np.random.rand(n_hidden_2, self.n_hidden_1) - 0.5
        self.b2 = np.random.rand(n_hidden_2, 1) - 0.5
        
        self.W3 = np.random.rand(self.n_hidden_3, n_hidden_2) - 0.5
        self.b3 = np.random.rand(self.n_hidden_3, 1) - 0.5
        
        self.W4 = np.random.rand(self.n_out, self.n_hidden_3) - 0.5
        self.b4 = np.random.rand(self.n_out, 1)  - 0.5

    def forward_prop(self, X):

        self.Z1 = self.W1.dot(X) + self.b1
        self.A1 = ReLU(self.Z1)
        
        self.Z2 = self.W2.dot(self.A1) + self.b2
        self.A2 = ReLU(self.Z2)
        
        self.Z3 = self.W3.dot(self.A2) + self.b3
        self.A3 = ReLU(self.Z3)
        
        self.Z4 = self.W4.dot(self.A3) + self.b4
        self.A4 = softmax(self.Z4)

    def back_prop(self, X, Y):
        
        m = Y.shape[0]
    
        one_hot_Y = one_hot(Y)
        
        self.dZ4 = self.A4 - one_hot_Y
        self.dW4 = 1/m * self.dZ4.dot(self.A3.T)
        self.db4 = 1/m * np.sum(self.dZ4)
        
        self.dZ3 = self.W4.T.dot(self.dZ4) * deriv_ReLU(self.Z3)
        self.dW3 = 1/m * self.dZ3.dot(self.A2.T)
        self.db3= 1/m * np.sum(self.dZ3)

        
        self.dZ2 = self.W3.T.dot(self.dZ3) * deriv_ReLU(self.Z2)
        self.dW2 = 1/m * self.dZ2.dot(self.A1.T)
        self.db2 = 1/m * np.sum(self.dZ2)
        
        
        self.dZ1 = self.W2.T.dot(self.dZ2) * deriv_ReLU(self.Z1)
        self.dW1 = 1/m * self.dZ1.dot(X.T)
        self.db1 = 1/m * np.sum(self.dZ1)

    def update_params(self, alpha):
        
        self.W1 = self.W1 - alpha * self.dW1
        self.b1 = self.b1 - alpha * self.db1
        
        self.W2 = self.W2 - alpha * self.dW2
        self.b2 = self.b2 - alpha * self.db2
        
        self.W3 = self.W3 - alpha * self.dW3
        self.b3 = self.b3 - alpha * self.db3
        
        self.W4 = self.W4 - alpha * self.dW4
        self.b4 = self.b4 - alpha * self.db4

    def get_predictions(self):

        predictions = np.argmax(self.A4, 0)

        return predictions


    def gradient_descent(self, X, Y, epochs, alpha):
              
        for epoch in range(epochs):
            self.forward_prop(X)
            self.back_prop(X, Y)
            self.update_params(alpha)

            predictions = self.get_predictions()
            accuracy = get_accuracy(predictions, Y)
            
            if (epoch % 10 == 0):
                print("Epoch: ", epoch)
                print("Accuracy: ", accuracy)

    def make_predictions(self, X):
    
        self.forward_prop(X)
        predictions = self.get_predictions()
    
        return predictions

    def test_predictions(self, index):
        current_image = X_test[:, index, None]
        prediction = self.make_predictions(X_test[:, index, None])
        label = y_test[index]
        print(f'Index: {index}', "Prediction: ", prediction, "Label: ", label)


X_train, X_test, y_train, y_test = load_process_data()
nn = NeuralNetwork(784, 128, 64, 32, 10)
nn.gradient_descent(X_train, y_train, 500, 0.1)
nn.test_predictions(31)

