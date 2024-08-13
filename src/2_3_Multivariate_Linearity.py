import numpy as np
import matplotlib.pyplot as plt

# 生成数据
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)


plt.scatter(X, y)
plt.xlabel('X')
plt.ylabel('y')
plt.title('Scatter plot of X and y')
plt.show()

X_b = np.c_[np.ones((100,1)),X]
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

x_new = np.linspace(X.min(), X.max(), 100)

# 使用最优参数 theta_best 计算预测值
y_predict = X * theta_best[1] + theta_best[0]

# 绘制散点图和拟合线
plt.scatter(X, y, label='Data points')
plt.plot(X, y_predict, color='red', label='Fitted line')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()