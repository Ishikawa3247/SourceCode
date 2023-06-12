import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier


def german_data():
    names = [
        'existingchecking',
        'duration',
        'credithistory',
        'purpose',
        'creditamount',
        'savings',
        'employmentsince',
        'installmentrate',
        'statussex',
        'otherdebtors',
        'residencesince',
        'property',
        'age',
        'otherinstallmentplans',
        'housing',
        'existingcredits',
        'job',
        'peopleliable',
        'telephone',
        'foreignworker',
        'classification'
    ]

    german_dummies = [
        'existingchecking',
        'credithistory',
        'purpose',
        'savings',
        'employmentsince',
        'statussex',
        'otherdebtors',
        'property',
        'otherinstallmentplans',
        'housing',
        'job',
        'telephone',
        'foreignworker',
    ]
    german = pd.read_csv('german.csv', delimiter=' ',
                         names=names)
    german_data = pd.get_dummies(german, columns=german_dummies, dtype=int)
    german_data = german_data.drop('classification', axis=1)
    german_target = pd.Series(german.classification)
    train_x, test_x, train_y, test_y = train_test_split(
        german_data, german_target, test_size=0.3, random_state=0)
    # print(german_data)
    # print(german_target)
    return train_x, test_x, train_y, test_y


def train(train_x, train_y):
    # 学習モデルを作成
    model = RandomForestClassifier()

    # 学習モデルにテストデータを与えて学習させる
    model.fit(train_x, train_y)
    return model


def test(model, test_x, test_y):
    y_pred_prob = model.predict_proba(test_x)[:, 1]
    # AUCの計算
    print("AUC: {}".format(roc_auc_score(test_y, y_pred_prob)))


def main():
    train_x, test_x, train_y, test_y = german_data()
    model = train(train_x, train_y)
    test(model, test_x, test_y)


if __name__ == '__main__':
    main()
