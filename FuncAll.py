import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

def compute_cost(X, y, theta):
    m = len(y)
    predictions = X.dot(theta)
    cost = (1/(2*m)) * np.sum(np.square(predictions - y))
    return cost


def gradient_descent(X, y, theta, accurate=0.0001, learning_rate=0.5, iterations=1000):
    m = len(y) 
    n = X.shape[1] 

    for _ in range(iterations):
        predictions = np.dot(X, theta)
        error = predictions - y
        gradients = (1 / m) * np.dot(X.T, error)
        theta -= learning_rate * gradients

        if np.linalg.norm(gradients) < accurate:
            break

    final_cost = compute_cost(X, y, theta)  # Assuming you have a compute_cost function
    print('GD iterations: ', _+1)
    print('GD data calculated per iteration: ', m)
    print('GD cost: ', final_cost)
    return theta

def stochastic_gradient_descent(X, y,theta,accurate = 0.0001, learning_rate=0.5, iterations=1000):
    m = len(y)  
    n = X.shape[1]
    for _ in range(iterations):
        for i in range(m):
            rand_ind = np.random.randint(0, m)  
            X_i = X[rand_ind, :].reshape(1, n)
            y_i = y[rand_ind].reshape(1,)

            prediction = np.dot(X_i, theta)
            error = prediction - y_i

            gradients = X_i.T.dot(error)
            
            theta -= learning_rate * gradients
            
        if(abs(sum(gradients))/m < accurate):
                break
        
    final_cost = compute_cost(X, y, theta)
    print('SGD iterations: ',_+1)
    print('SGD data calculated per iteration: ', m)
    print('SGD cost: ', final_cost)
    return theta


def mini_batch_gradient_descent(X, y,theta,accurate = 0.0001, learning_rate=0.5, iterations=1000, batch_size=10):
    m = len(y)  
    n = X.shape[1] 
    for _ in range(iterations):
        indices = np.random.permutation(m)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        gradient_magnitude = np.linalg.norm(X_shuffled.T.dot(X_shuffled.dot(theta) - y_shuffled)) / m
        if gradient_magnitude < accurate:
            break

        for i in range(0, m, batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]

            y_prediected = X_batch.dot(theta)
            error = y_prediected - y_batch

            gradient = (1/batch_size) * X_batch.T.dot(error)
            theta -= learning_rate * gradient
            

    itera = int(m/batch_size)
    if(_ !=0):
        itera *= (_+1)

    final_cost = compute_cost(X, y, theta)
    print('MBGD iterations: ',itera)
    print('MBGD data calculated per iteration: ', batch_size)
    print('MBGD cost: ', final_cost)
    return theta



# np.random.seed(42)
# X = np.random.rand(100, 2)
# y = 1+ 5 * X[:, 0] + 2 * X[:, 1] 

# X_b = np.c_[np.ones((100, 1)), X]

# theta = mini_batch_gradient_descent(X_b, y)
# print()
# theta_ = gradient_descent(X_b, y)
# print()
# theta__ = stochastic_gradient_descent(X_b, y)


# test_data_X = np.random.rand(100, 2)
# predicted_data_y = theta[0]+ theta[1]*test_data_X[:, 0] + theta[2]*test_data_X[:, 1]

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d') #set up 3d plot

# ax.scatter(X[:, 0], X[:, 1], y, c='red', label='random data')
# ax.scatter(test_data_X[:,0], test_data_X[:,1], predicted_data_y, c = 'blue', label='predicted data')

# ax.legend() 
# plt.show()

def create_randomint_array(n,min,max, randomseed = 1):
    np.random.seed(randomseed)
    return np.array([[np.random.randint(min, max)] for _ in range(n)])

def transpose_merge_one(X):
    return np.c_[np.ones((X.shape[0], 1)), X]

class LinearRegression:
    def __init__(self,X,y):
        self.theta = np.zeros(X.shape[1]) 
        self.X = np.array(X)
        self.y = np.array(y)
    def GD_fit(self,accurate = 0.0001,learning_rate=0.5, iterations=1000):
        self.theta = gradient_descent(self.X, self.y,self.theta , accurate, learning_rate, iterations)
    def SGD_fit(self,accurate = 0.0001,learning_rate=0.5, iterations=1000):
        self.theta = stochastic_gradient_descent(self.X, self.y,self.theta , accurate, learning_rate, iterations)
    def MBGD_fit(self,accurate = 0.0001,learning_rate=0.5, iterations=1000, batch_size=10):
        self.theta = mini_batch_gradient_descent(self.X, self.y,self.theta , accurate, learning_rate, iterations, batch_size)
    def Predict(self, X):
        z = np.dot(X, self.theta)
        return z
    def Save_Theta(self,directory_path, filename):
        os.makedirs(directory_path, exist_ok=True)
        file_path = os.path.join(directory_path, filename)
        np.save(file_path, self.theta)
    def Load_Theta(self, directory_path,filename):
        os.makedirs(directory_path, exist_ok=True)
        file_path = os.path.join(directory_path, filename)
        self.theta = np.load(file_path)

def accuracy_score(y_true, y_pred):
    correct_predictions = np.sum(y_true == y_pred)
    total_predictions = len(y_true)
    
    accuracy = correct_predictions / total_predictions
    return accuracy

class LogisticRegresstion():
    def __init__(self,X,y):
        self.theta = np.zeros(X.shape[1]) 
        self.X = np.array(X)
        self.y = np.array(y)
    def Sigmoid(self,z):
        s = 1 / (1 + np.exp(-z))
        return s
    def compute_log_loss(self,X, y, theta):
        m = len(self.y)
        z = np.dot(self.X, theta)
        h = self.Sigmoid(z)
        loss = -np.mean(self.y * np.log(h) + (1 - self.y) * np.log(1 - h))
        return loss
    def Sigmoid_GD_fit(self, accurate = 0.0001,learning_rate=0.5, iterations=1000):
        m = len(self.y) 
        n = self.X.shape[1] 
        self.theta = np.zeros(n) 
        for _ in range(iterations):
            predictions = np.dot(self.X, self.theta)

            error = self.Sigmoid(predictions) - self.y
            gradients = (1 / m) * np.dot(self.X.T, error)

            self.theta -= learning_rate * gradients

            if(abs(sum(gradients))/m < accurate):
                break

        final_cost = self.compute_log_loss(self.X, self.y, self.theta)
        print('Sigmoid_GD iterations: ', _+1)
        print('Sigmoid_GD data calculated per iteration: ', m)
        print('Sigmoid_GD cost: ', final_cost)
    def Predict_Probability(self, X):
        z = np.dot(X, self.theta)
        h = self.Sigmoid(z)
        return h
    def Predict(self, X):
        probabilities = self.Sigmoid(np.dot(X, self.theta))
        y_pred = (probabilities >= 0.5).astype(int)
        return y_pred
    def Save_Theta(self,directory_path, filename):
        os.makedirs(directory_path, exist_ok=True)
        file_path = os.path.join(directory_path, filename)
        np.save(file_path, self.theta)
    def Load_Theta(self, directory_path,filename):
        os.makedirs(directory_path, exist_ok=True)
        file_path = os.path.join(directory_path, filename)
        self.theta = np.load(file_path)


def load_ubyte(filename):
    with open(filename, 'rb') as f:
        # Read magic number
        magic_number = int.from_bytes(f.read(4), 'big')

        if magic_number == 2051:  # idx3 (image data) format
            # Read number of items, rows, and columns
            num_items = int.from_bytes(f.read(4), 'big')
            num_rows = int.from_bytes(f.read(4), 'big')
            num_cols = int.from_bytes(f.read(4), 'big')

            # Read the rest of the file as unsigned bytes
            data = np.fromfile(f, dtype=np.uint8)
            data = data.reshape(num_items, num_rows, num_cols)

        elif magic_number == 2049:  # idx1 (label data) format
            # Read number of items
            num_items = int.from_bytes(f.read(4), 'big')

            # Read the rest of the file as unsigned bytes
            data = np.fromfile(f, dtype=np.uint8)

        else:
            raise ValueError("Unsupported magic number. The file is not in idx1 or idx3 format.")

    return data
