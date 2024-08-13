import matplotlib.pyplot as plt
import numpy as np

X = [12.3, 14.3, 14.5, 14.8, 16.1, 16.8, 16.5, 15.3, 17.0, 17.8, 18.7, 20.2, 22.3, 19.3, 15.5, 16.7, 17.2, 18.3, 19.2,
     17.3, 19.5, 19.7, 21.2, 23.04, 23.8, 24.6, 25.2, 25.7, 25.9, 26.3]
y = [11.8, 12.7, 13.0, 11.8, 14.3, 15.3, 13.5, 13.8, 14.0, 14.9, 15.7, 18.8, 20.1, 15.0, 14.5, 14.9, 14.8, 16.4, 17.0,
     14.8, 15.6, 16.4, 19.0, 19.8, 20.0, 20.3, 21.9, 22.1, 22.4, 22.6]

print(len(X))

X_train = X[0:20]
y_train = y[0:20]
n_train = len(X_train)

X_test = X[20:]
y_test = y[20:]
n_test = len(X_test)

w = -0.1
b = 3
lr = 0.00001



epoches = 100 #循环一百次
for i in range(epoches):
    sum_w = 0.0
    sum_b = 0.0
    for i in range(n_train):
        y_hat = w * X_train[i] + b
        sum_w += (y_train[i] - y_hat) * (-X_train[i])
        sum_b += (y_train[i] - y_hat)* (-1)
    det_w = 2 * sum_w
    det_b = 2 * sum_b
    w = w - lr * det_w
    b = b - lr * det_b

fig,ax = plt.subplots()
ax.scatter(X_train,y_train)
ax.plot([i for i in range(10,27)],[w*i+b for i in range(10,27)])
plt.title('y=w*x+b')
plt.legend(('Model','Data Points'),loc = 'upper left')
plt.show()