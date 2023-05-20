import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

if __name__ == '__main__':
    # 读取数据集
    data = pd.read_csv('../data/iris.csv')

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(data.iloc[:, :4], data.iloc[:, 4], test_size=0.3,
                                                        random_state=42)

    print("总条数:", len(data))
    print("训练数据条数", len(X_train))
    print("测试条数:", len(X_test))

    dtc = DecisionTreeClassifier(max_depth=10)

    # 训练模型
    dtc.fit(X_train, y_train)

    # 预测测试数据的分类
    y_pred = dtc.predict(X_test)

    # 输出预测结果
    print("预测的结果值:", y_pred)
    print("实际的结果值:", y_test.values)

    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    print('准确率Accuracy:', accuracy)

    # 可视化决策树

    from sklearn.tree import export_graphviz

    # 需要本机上安装对应的软件才能查看iris_tree_img.dot
    dot_data = export_graphviz(
        dtc,
        out_file="../export_model/iris_tree_img.dot",
        feature_names=data.columns[:4],
        class_names=data['name'].unique(),
        rounded=True,
        filled=True
    )
