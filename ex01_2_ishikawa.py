import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import pandas as pd


def iris_data():
    iris = load_iris()
    iris_data = pd.DataFrame(iris.data, columns=iris.feature_names)
    iris_target = pd.Series(iris.target)
    print(iris_data)
    print(iris_target)
    train_x, test_x, train_y, test_y = train_test_split(
        iris_data, iris_target, test_size=0.2, shuffle=True)
    return train_x, test_x, train_y, test_y


def train(train_x, test_x, train_y, test_y):
    dtrain = xgb.DMatrix(train_x, label=train_y)
    param = {'max_depth': 2, 'eta': 1,
             'objective': 'multi:softmax', 'num_class': 3}
    num_round = 20
    bst = xgb.train(param, dtrain, num_round)
    return bst


def test(bst, train_x, test_x, train_y, test_y):
    dtest = xgb.DMatrix(test_x)
    pred = bst.predict(dtest)
    score = accuracy_score(test_y, pred)
    print('score:{0:}'.format(score))


def main():
    train_x, test_x, train_y, test_y = iris_data()
    bst = train(train_x, test_x, train_y, test_y)
    test(bst, train_x, test_x, train_y, test_y)


if __name__ == "__main__":
    main()
