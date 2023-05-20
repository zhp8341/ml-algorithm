from sklearn.neighbors import KNeighborsClassifier

from load_data import load_data
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    X, y = load_data()

    # 将数据集分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(len(X))
    print(len(X_train))
    print(len(X_test))

    knn = KNeighborsClassifier(n_neighbors=3)

    # 训练模型
    knn.fit(X_train, y_train)

    # 预测测试数据的分类
    y_pred = knn.predict(X_test)

    # 输出预测结果
    print("预测的结果值:", y_pred)
    print("实际的结果值:", y_test)
