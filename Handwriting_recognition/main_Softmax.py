import numpy as np
import sys
sys.path.append('../MachineLearning')
import FuncAll as fa
import os
from PIL import Image

Xtrain_all = fa.load_ubyte('Handwriting_recognition/Dataset/train-images.idx3-ubyte')
ytrain_all = fa.load_ubyte('Handwriting_recognition/Dataset/train-labels.idx1-ubyte')

Xtest_all = fa.load_ubyte('Handwriting_recognition/Dataset/t10k-images.idx3-ubyte')
ytest_all = fa.load_ubyte('Handwriting_recognition/Dataset/t10k-labels.idx1-ubyte')

cls = [[0], [1]]
def extract_data(X, y, num_classes=10):
    X_res = []
    y_res = []
    for cls in range(num_classes):
        indices = np.where(y == cls)[0]
        X_cls = X[indices, :] / 255.0
        y_cls = np.asarray([cls] * len(indices))
        X_res.append(X_cls)
        y_res.append(y_cls)
    X_res = np.vstack(X_res)
    y_res = np.hstack(y_res)
    return X_res, y_res



# extract data for training , evaluation image of 1 and 0 handwriting
(X_train, y_train) = extract_data(Xtrain_all, ytrain_all)
# extract data for test 
(X_test, y_test) = extract_data(Xtest_all, ytest_all)




X_flat = X_train.reshape(X_train.shape[0], -1)
X_b = fa.transpose_merge_one(X_flat)

X_flat_test = X_test.reshape(X_test.shape[0], -1)
X_b_test = np.c_[np.ones((len(X_flat_test), 1)), X_flat_test]
 
model = fa.SoftmaxRegression(num_classes= 10)
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "SaveThetas_Softmax")
if model.load_theta(save_dir, 'theta.npy') == False:
    model.GD_fit(X_b, y_train, accurate=0.000001,num_iterations=1000, learning_rate=0.69)
    model.save_theta(save_dir, 'theta.npy')


y_predicted= model.predict_classes(X_b_test)

print("MNIST test data: ")
fa.accuracy_score(y_test, y_predicted)



script_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(script_dir, "test_image_4.png")
image = Image.open(image_path)
image = image.convert("L")
image = image.resize((28, 28))
image_array = np.array(image)
image_array = np.expand_dims(image_array, axis=0)
image_array_flat = image_array.reshape(image_array.shape[0], -1)
image_array_b = fa.transpose_merge_one(image_array_flat)
image_array_b[0] = image_array_b[0]/255
image_array_b[0][0] = 1

print("\nPredict test image 4: ")
print(f"Probabilities:")
probs = model.predict_proba(image_array_b[0])
for i in range(len(probs[0])):
    print(f"  Prob {i}: {(probs[0][i]*100):.2f}%")
print(f"Prediction: {model.predict_classes(image_array_b[0])[0]}")





