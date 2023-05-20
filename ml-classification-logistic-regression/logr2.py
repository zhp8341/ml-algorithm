import numpy as np
from scatter_img import show_img
from  load_data import load_data
from sklearn.linear_model import LogisticRegression

# sklearn逻辑回归模型


if __name__ == '__main__':
    # 准备数据
    X, y = load_data()
    print("开始训练")
    # 创建一个逻辑回归模型
    model = LogisticRegression()
    model.fit(X, y)

    # 获取损失函数的值
    loss = model.score(X,y)
    print("损失函数的值为：", loss)

    # 预测结果
    X_new = np.array([[5.0, 1.7], [1.4, 0.2], [1.3, 0.2]])
    y_pred = model.predict(X_new)
    print("预测结果：", y_pred)

    # 绘制决策边界
    show_img(X,y,model)




