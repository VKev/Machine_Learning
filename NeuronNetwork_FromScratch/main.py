import numpy as np
import pandas as pd
import os

# Load and prepare the data
data = pd.read_csv("NeuronNetwork_FromScratch/Dataset/train.csv")
data = np.array(data)
np.random.shuffle(data)

data_test = data[0:1000].T
y_test = data_test[0]
x_test = data_test[1:]

data_train = data[1000:].T
y_train = data_train[0]
x_train = data_train[1:]

n = x_train.shape[0]  # Number of features
m = x_train.shape[1]  # Number of training examples

# Normalize data
x_train = x_train / 255.0
x_test = x_test / 255.0

# Dense layer with ReLU activation
class dense_ReLU:
    def __init__(self, neuron, params):
        self.w = np.random.randn(neuron, params) * np.sqrt(2. / params)  # He initialization
        self.b = np.zeros((neuron, 1))
        self.z = None
        self.a = None
        self.dz = None
        self.dw = None
        self.db = None

    def ReLU(self, z):
        return np.maximum(0, z)

    def derivative_ReLU(self, z):
        return z > 0

    def forward_prop(self, input_params):
        self.z = np.dot(self.w, input_params) + self.b
        self.a = self.ReLU(self.z)

    def backward_prop(self, dz, input_params):
        self.dz = dz * self.derivative_ReLU(self.z)
        self.dw = (1 / m) * np.dot(self.dz, input_params.T)
        self.db = (1 / m) * np.sum(self.dz, axis=1, keepdims=True)
    
    def save_weights(self, folder_path):
        os.makedirs(folder_path, exist_ok=True)  # Create directory if not exists
        np.save(os.path.join(folder_path, "weights.npy"), self.w)
        np.save(os.path.join(folder_path, "biases.npy"), self.b)

    def load_weights(self, folder_path):
        if os.path.exists(folder_path):
            self.w = np.load(os.path.join(folder_path, "weights.npy"))
            self.b = np.load(os.path.join(folder_path, "biases.npy"))
            return True
        else :
            return False
# Output layer with Softmax activation
class output_Softmax:
    def __init__(self, neuron, params):
        self.classes = neuron
        self.w = np.random.randn(neuron, params) * np.sqrt(2. / params)  # He initialization
        self.b = np.zeros((neuron, 1))
        self.z = None
        self.a = None
        self.dz = None
        self.dw = None
        self.db = None

    def Softmax(self, z):
        z -= np.max(z, axis=0, keepdims=True)
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z, axis=0, keepdims=True)

    def one_hot_encode(self, labels):
        labels = labels.astype(int)
        num_samples = labels.shape[0]
        one_hot_labels = np.zeros((self.classes, num_samples))
        one_hot_labels[labels, np.arange(num_samples)] = 1
        return one_hot_labels

    def forward_prop(self, input_params):
        self.z = np.dot(self.w, input_params) + self.b
        self.a = self.Softmax(self.z)

    def backward_prop(self, input_labels, previousLayer):
        one_hot_y = self.one_hot_encode(input_labels)
        self.dz = self.a - one_hot_y
        self.dw = (1 / m) * np.dot(self.dz, previousLayer.a.T)
        self.db = (1 / m) * np.sum(self.dz, axis=1, keepdims=True)

    def predict(self, input_params):
        self.forward_prop(input_params)
        predicted_labels = np.argmax(self.a, axis=0)
        return predicted_labels
    
    def save_weights(self, folder_path):
        os.makedirs(folder_path, exist_ok=True)  # Create directory if not exists
        np.save(os.path.join(folder_path, "weights.npy"), self.w)
        np.save(os.path.join(folder_path, "biases.npy"), self.b)

    def load_weights(self, folder_path):
        if os.path.exists(folder_path):
            self.w = np.load(os.path.join(folder_path, "weights.npy"))
            self.b = np.load(os.path.join(folder_path, "biases.npy"))
            return True
        else :
            return False

# Gradient descent function
def gradient_descent(layerlist, x_train, y_train, learning_rate, num_epochs):
    for epoch in range(num_epochs):
        # Forward propagation
        input_params = x_train
        for layer in layerlist:
            layer.forward_prop(input_params)
            input_params = layer.a
        
        # Backward propagation
        dz = layerlist[-1].a - layerlist[-1].one_hot_encode(y_train)
        for i in reversed(range(len(layerlist))):
            layer = layerlist[i]
            input_params = x_train if i == 0 else layerlist[i - 1].a
            if i == len(layerlist) - 1:
                layer.backward_prop(y_train, layerlist[i - 1])
            else:
                dz = np.dot(layerlist[i + 1].w.T, layerlist[i + 1].dz)
                layer.backward_prop(dz, input_params)

        # Update weights and biases
        for layer in layerlist:
            layer.w -= learning_rate * layer.dw
            layer.b -= learning_rate * layer.db

        if epoch % 25 == 0:
            predictions = layerlist[-1].predict(layerlist[-2].a)
            accuracy = np.mean(predictions == y_train)
            print(f'Epoch {epoch}: Training accuracy: {accuracy * 100:.2f}%')

    for layer_index, layer in enumerate(layerlist):
        layer.save_weights(f"NeuronNetwork_FromScratch/SaveThetas/layer{layer_index}")        

def load_weights(layerlist, folder_path):
    try:
        for layer_index, layer in enumerate(layerlist):
            loadsuccess = layer.load_weights(os.path.join(folder_path, f"layer{layer_index}"))
            if loadsuccess == False:
                return False
        return True
    except:
        return False


hiddenLayer1 = dense_ReLU(neuron=128, params=n)
hiddenLayer2 = dense_ReLU(neuron=64, params=128)
outputLayer = output_Softmax(neuron=10, params=64)  # Update the number of neurons here

layerlist = [hiddenLayer1, hiddenLayer2, outputLayer]

loaded_layerlist = [dense_ReLU(neuron=128, params=n), dense_ReLU(neuron=64, params=128), output_Softmax(neuron=10, params=64)]
loadsuccess = load_weights(loaded_layerlist, "NeuronNetwork_FromScratch/SaveThetas")

# If weights are successfully loaded, use the loaded weights, else train the network
if loadsuccess:
    layerlist = loaded_layerlist
else:
    # Train the network
    learning_rate = 0.5
    num_epochs = 100
    gradient_descent(layerlist, x_train, y_train, learning_rate, num_epochs)
# Evaluate on the test set
input_params = x_test
for layer in layerlist:
    layer.forward_prop(input_params)
    input_params = layer.a
test_predictions = layerlist[-1].predict(layerlist[-2].a)
test_accuracy = np.mean(test_predictions == y_test)
print(f'Test accuracy: {test_accuracy * 100:.2f}%')


# # Initialize layers
# hiddenLayer = dense_ReLU(neuron=128, params=n)
# outputLayer = output_Softmax(neuron=10, params=128)

# # Training parameters
# learning_rate = 0.5
# num_epochs = 100
# for epoch in range(num_epochs):
#     hiddenLayer.forward_prop(x_train)
#     outputLayer.forward_prop(hiddenLayer.a)

#     outputLayer.backward_prop(y_train, hiddenLayer)
#     dz = np.dot(outputLayer.w.T, outputLayer.dz)  # Compute dz for hidden layer
#     hiddenLayer.backward_prop(dz, x_train)

#     outputLayer.w -= learning_rate * outputLayer.dw
#     outputLayer.b -= learning_rate * outputLayer.db

#     hiddenLayer.w -= learning_rate * hiddenLayer.dw
#     hiddenLayer.b -= learning_rate * hiddenLayer.db

#     if epoch % 25 == 0:
#         predictions = outputLayer.predict(hiddenLayer.a)
#         accuracy = np.mean(predictions == y_train)
#         print(f'Epoch {epoch}: Training accuracy: {accuracy * 100:.2f}%')
