import FuncAll as fa
import numpy as np 
import matplotlib.pyplot as plt


X = fa.create_randomint_array(n=100, min=0, max=100, randomseed=5)
X_one = fa.transpose_merge_one(X)

y = np.array([0 if x < 50 else 1 for x in X])



lr = fa.LogisticRegresstion(X_one,y)
lr.Sigmoid_GD_fit(learning_rate=0.05,iterations=4000)


sigmoidValues = lr.Predict_Probability(X_one)

plt.scatter(X.T,y)
plt.scatter(X, sigmoidValues, label='Sigmoid Function', color='red')
plt.show()






