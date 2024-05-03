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
import numpy as np

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


 
classes = [0,1,2,3,4,5,6,7,8,9]
model = fa.OneVsRestLogisticRegression(classes=classes)

save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "SaveThetas")
if model.load_thetas(save_dir) == False:
    model.fit(X_b, y_train, learning_rate=0.69)
    model.save_thetas(save_dir)



y_predicted = model.predict(X_b_test)
fa.accuracy_score(y_test,y_predicted)




