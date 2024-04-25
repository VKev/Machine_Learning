from __future__ import division, print_function, unicode_literals
import numpy as np 
import matplotlib.pyplot as plt

# height (cm)
X = np.array([[147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180, 183]]).T
# weight (kg)
y = np.array([[ 49, 50, 51,  54, 58, 59, 60, 62, 63, 64, 66, 67, 68]]).T

# diem toi uu cua Linear regression co dang:
# A dagger co nghia la nghich dao cua A
# ¯X la chuyen vi cua X
# w = ((¯X^T)*¯X) dagger * (¯X^T)*y
# hoac w = A dagger * b


one = np.ones((X.shape[0], 1))
Xbar = np.concatenate((one, X), axis = 1)

A = np.dot(Xbar.T, Xbar) # A = (¯X^T) * ¯X
b = np.dot(Xbar.T, y) # b = (¯X^T) * y
w = np.dot(np.linalg.pinv(A), b) # A dagger * b = ((¯X^T)*¯X) dagger * (¯X^T)*y
print('w = ', w)

#sau khi co w thi cho vao ham linear
w_0 = w[0][0]
w_1 = w[1][0]
x0 = np.linspace(145, 200, 2)
y0 = w_0 + w_1*x0


print ('x = ' +str(x0) +'\ny = '+ str(y0))



# Drawing the fitting line 
plt.plot(X.T, y.T, 'ro')     # data 
plt.plot(x0, y0, 'ro',color = 'blue')               # the fitting line
plt.axis([140, 300, 45, 300])
plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')
plt.show()
 