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

    final_cost = compute_cost(X, y, theta) 
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
    print(f"Accuracy: {accuracy * 100:.2f}%")

class LogisticRegresstion():
    def __init__(self, X=None, y=None):
        if X is not None and y is not None:
            self.theta = np.zeros(X.shape[1]) 
            self.X = np.array(X)
            self.y = np.array(y)
        else:
            self.theta = None
            self.X = None
            self.y = None
    def Sigmoid(self,z):
        s = 1 / (1 + np.exp(-z))
        return s
    def compute_log_loss(self):
        m = len(self.y)
        z = np.dot(self.X, self.theta)
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

        final_cost = self.compute_log_loss()
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


class OneVsRestLogisticRegression():
    def __init__(self, classes):
        self.classes = classes
        self.classifiers = {}
    
    def GD_fit(self, X, y, accurate = 0.0001,learning_rate=0.5, iterations=1000):
        for cls in self.classes:
            y_binary = (y == cls).astype(int)
            classifier = LogisticRegresstion(X, y_binary)
            print(f"{cls}.")
            classifier.Sigmoid_GD_fit(accurate,learning_rate,iterations)
            self.classifiers[cls] = classifier
    
    def predict_proba(self, X):
        probabilities = {}
        for cls, classifier in self.classifiers.items():
            probabilities[cls] = classifier.Predict_Probability(X)
        return probabilities
    
    def predict(self, X):
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
            
        probabilities = self.predict_proba(X)
        predictions = np.argmax(np.array(list(probabilities.values())), axis=0)
        return [list(self.classes)[p] for p in predictions]
    
    def save_thetas(self, directory_path):
        for cls, classifier in self.classifiers.items():
            classifier.Save_Theta(directory_path, f"{cls}_theta.npy")
    
    def load_thetas(self, directory_path):
        try:
            for cls in self.classes:
                classifier = LogisticRegresstion() 
                classifier.Load_Theta(directory_path, f"{cls}_theta.npy")
                self.classifiers[cls] = classifier
            return True
        except Exception as e:
            print(f"Load model failed: {e}")
            return False

class SoftmaxRegression:
    def __init__(self, num_classes) :
        self.theta = None
        self.num_classes = num_classes
    def softmax(self,X):
        if X.ndim == 1:
            X = X.reshape(1, -1) 
        exp_scores = np.exp(X - np.max(X, axis=1, keepdims=True))
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    def one_hot_encode(self,y):
        m = len(y)
        one_hot = np.zeros((m, self.num_classes))
        one_hot[np.arange(m), y] = 1
        return one_hot
    def compute_cross_entropy_loss(self, y_true, y_pred):
        m = len(y_true)
        return -np.sum(y_true * np.log(y_pred)) / m
    def MBGD_fit(self, X, y, accurate=0.0001, num_iterations=1000, learning_rate=0.5, batch_size=32):
        m, n = X.shape
        W = np.zeros((n, self.num_classes))
        b = np.zeros(self.num_classes)
        
        y_one_hot = self.one_hot_encode(y)
        prev_loss = float('inf')
        
        for i in range(num_iterations):
            shuffled_indices = np.random.permutation(m)
            X_shuffled = X[shuffled_indices]
            y_shuffled = y[shuffled_indices]
            
            for batch_start in range(0, m, batch_size):
                X_batch = X_shuffled[batch_start:batch_start+batch_size]
                y_batch = y_shuffled[batch_start:batch_start+batch_size]
                
                scores = np.dot(X_batch, W) + b
                probs = self.softmax(scores)
                
                gradient_W = np.dot(X_batch.T, probs - self.one_hot_encode(y_batch)) / len(X_batch)
                gradient_b = np.mean(probs - self.one_hot_encode(y_batch), axis=0)
                
                W -= learning_rate * gradient_W
                b -= learning_rate * gradient_b
            
            current_probs = self.softmax(np.dot(X, W) + b)
            loss = self.compute_cross_entropy_loss(y_one_hot, current_probs)
            
            if abs(prev_loss - loss) < accurate:
                break
            
            prev_loss = loss
            
        self.theta = np.concatenate((W.reshape(-1), b))
        final_loss = self.compute_cross_entropy_loss(y_one_hot, self.predict_proba(X))
        print('Softmax_MBGD iterations:', i + 1)
        print('Softmax_MBGD data calculated per iteration:', batch_size)
        print('Softmax_MBGD cost:', final_loss)
    def GD_fit(self,X, y, accurate=0.0001,num_iterations=1000, learning_rate=0.5):
        m, n = X.shape
        W = np.zeros((n, self.num_classes))
        b = np.zeros(self.num_classes)
        
        y_one_hot = self.one_hot_encode(y)
        prev_loss = float('inf') 
        for i in range(num_iterations):
            scores = np.dot(X, W) + b
            probs = self.softmax(scores)

            gradient_W = np.dot(X.T, probs - y_one_hot) / m
            gradient_b = np.mean(probs - y_one_hot, axis=0)

            W -= learning_rate * gradient_W
            b -= learning_rate * gradient_b

            loss = -np.sum(y_one_hot * np.log(probs)) / m

            if abs(prev_loss - loss) < accurate:
                break
            
            prev_loss = loss
            
        self.theta = np.concatenate((W.reshape(-1), b))
        final_loss = self.compute_cross_entropy_loss(y_one_hot, self.predict_proba(X))
        print('Softmax_GD iterations:', i + 1)
        print('Softmax_GD data calculated per iteration:', m)
        print('Softmax_GD cost:', final_loss)
    def predict_proba(self, X):
        if self.theta is None:
            raise ValueError("Model not trained. Please fit the model first.")
        
        w = self.theta[:-self.num_classes].reshape(-1, self.num_classes)
        b= self.theta[-self.num_classes:]
        scores = np.dot(X, w) +  b
        probs = self.softmax(scores)
        return probs
    def predict_classes(self,X):
        probs = self.predict_proba(X)
        classes = np.argmax(probs, axis=1)
        return classes
    def save_theta(self, directory, filename):
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        filepath = os.path.join(directory, filename)
        np.save(filepath, self.theta)
    
    def load_theta(self, directory, filename):
        filepath = os.path.join(directory, filename)
        if os.path.exists(filepath):
            self.theta = np.load(filepath)
            return True
        else:
            print("File not found")
            return False

    


def load_ubyte(filename):
    with open(filename, 'rb') as f:
        magic_number = int.from_bytes(f.read(4), 'big')

        if magic_number == 2051: 
            num_items = int.from_bytes(f.read(4), 'big')
            num_rows = int.from_bytes(f.read(4), 'big')
            num_cols = int.from_bytes(f.read(4), 'big')
            data = np.fromfile(f, dtype=np.uint8)
            data = data.reshape(num_items, num_rows, num_cols)

        elif magic_number == 2049: 
            num_items = int.from_bytes(f.read(4), 'big')
            data = np.fromfile(f, dtype=np.uint8)

        else:
            raise ValueError("Unsupported magic number. The file is not in idx1 or idx3 format.")

    return data
