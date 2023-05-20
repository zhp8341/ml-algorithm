import pandas as pd
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
# 贝叶斯算法（Bayesian algorithm）是根据已有数据计算出每个类别的概率，并基于此进行分类
# 贝叶斯算法可以分为朴素贝叶斯、高斯朴素贝叶斯、贝叶斯网络等多种形式。
#与其他机器学习算法相比，贝叶斯算法具有以下优点：
#可以有效利用先验知识，提高模型的准确性和稳定性；
#对噪声和缺失数据的容忍度较高；
#适用于处理多分类问题。


from sklearn.naive_bayes import GaussianNB
import numpy as np

if __name__ == '__main__':
    # 加载数据集
    data = pd.read_csv('../data/iris.csv')

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(data.iloc[:, :4], data.iloc[:, 4], test_size=0.3,
                                                        random_state=42)

    print("总条数:", len(data))
    print("训练数据条数", len(X_train))
    print("测试条数:", len(X_test))
    gnb = GaussianNB()

    # 训练模型
    gnb.fit(X_train, y_train)

    # 预测测试数据的分类
    y_pred = gnb.predict(X_test)

    # 输出预测结果
    print("预测的结果值:", y_pred)
    print("实际的结果值:", y_test.values)

    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    print('准确率Accuracy:', accuracy)