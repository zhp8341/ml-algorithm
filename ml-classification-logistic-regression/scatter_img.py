from matplotlib import pyplot as plt
import numpy as np


def show_img(X, y, model):
    # 绘制决策边界
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    x1_min, x1_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    x2_min, x2_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max, 100),
                           np.linspace(x2_min, x2_max, 100))
    X_new = np.column_stack((xx1.ravel(), xx2.ravel()))
    Z = model.predict(X_new)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, cmap=plt.cm.Spectral, alpha=0.3)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()
