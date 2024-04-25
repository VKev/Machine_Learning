import numpy as np
import sys
sys.path.append('../MachineLearning')
import FuncAll as fa
import os


Xtrain_all = fa.load_ubyte('Handwriting_recognition/Dataset/train-images.idx3-ubyte')
ytrain_all = fa.load_ubyte('Handwriting_recognition/Dataset/train-labels.idx1-ubyte')

Xtest_all = fa.load_ubyte('Handwriting_recognition/Dataset/t10k-images.idx3-ubyte')
ytest_all = fa.load_ubyte('Handwriting_recognition/Dataset/t10k-labels.idx1-ubyte')

cls = [[0], [1]]
def extract_data(X, y, classes):
    y_res_id = np.array([])
    for i in cls[0]:
        y_res_id = np.hstack((y_res_id, np.where(y == i)[0]))
    n0 = len(y_res_id)

    for i in cls[1]:
        y_res_id = np.hstack((y_res_id, np.where(y == i)[0]))
    n1 = len(y_res_id) - n0 

    y_res_id = y_res_id.astype(int)
    X_res = X[y_res_id, :]/255.0
    y_res = np.asarray([0]*n0 + [1]*n1)
    return (X_res, y_res)


# extract data for training , evaluation image of 1 and 0 handwriting
(X_train, y_train) = extract_data(Xtrain_all, ytrain_all, cls)

# extract data for test 
(X_test, y_test) = extract_data(Xtest_all, ytest_all, cls)



X_flat = X_train.reshape(X_train.shape[0], -1)
X_b = np.c_[np.ones((len(X_flat), 1)), X_flat]
theta = np.zeros(X_b.shape[1])


# train
current_dir = os.path.dirname(os.path.abspath(__file__))
theta_file_exists = os.path.exists(os.path.join(current_dir, 'theta.npy'))
model = fa.LogisticRegresstion(X_b, y_train)
if theta_file_exists:
    model.Load_Theta(current_dir,'theta.npy')
else:
    model.Sigmoid_GD_fit(learning_rate= 0.05,iterations= 1000)
    model.Save_Theta(current_dir,'theta.npy')


X_flat_test = X_test.reshape(X_test.shape[0], -1)
X_b_test = np.c_[np.ones((len(X_flat_test), 1)), X_flat_test]

y_pred = model.Predict(X_b_test)

accuracy = fa.accuracy_score(y_test,y_pred);
print(f"Accuracy: {accuracy * 100:.2f}%")



