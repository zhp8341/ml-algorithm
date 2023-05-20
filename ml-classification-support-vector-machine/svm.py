import pandas as pd
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

if __name__ == '__main__':
    # 加载数据集
    data = pd.read_csv('../data/iris.csv')

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(data.iloc[:, :4], data.iloc[:, 4], test_size=0.3,
                                                        random_state=42)

    print("总条数:", len(data))
    print("训练数据条数", len(X_train))
    print("测试条数:", len(X_test))

    # 初始化 SVM 模型
    svm = SVC(kernel='linear', C=1)

    # 在训练集上训练模型
    svm.fit(X_train, y_train)

    # 在测试集上进行预测
    y_pred = svm.predict(X_test)

    # 计算准确率
    accuracy = (y_pred == y_test).mean()
    print("Accuracy:", accuracy)