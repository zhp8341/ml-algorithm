from sklearn.neighbors import KNeighborsClassifier

from load_data import load_data
from sklearn.model_selection import train_test_split
import numpy as np
from collections import Counter

from sklearn import datasets
from sklearn.model_selection import train_test_split


class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        distances = [self._euclidean_distance(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

    def _euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))


if __name__ == '__main__':
    X, y = load_data()

    # 将数据集分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("总条数:", len(X))
    print("训练数据条数", len(X_train))
    print("测试条数:", len(X_test))

    knn = KNN(k=1)

    # 训练模型
    knn.fit(X_train, y_train)

    # 预测测试数据的分类
    y_pred = knn.predict(X_test)

    # 输出预测结果
    print("预测的结果值:", y_pred)
    print("实际的结果值:", y_test)
