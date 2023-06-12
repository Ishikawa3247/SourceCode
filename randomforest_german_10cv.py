import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold


def data():
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

    return german_data, german_target


def train_test(german_data, german_target):
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    # kf = KFold(n_splits=10, shuffle=True, random_state=42)
    scores = []

    for train_id, test_id in kf.split(german_data, german_target):
        X_train = german_data.iloc[train_id]
        X_test = german_data.iloc[test_id]
        y_train = german_target[train_id]
        y_test = german_target[test_id]

        model = RandomForestRegressor(n_estimators=100, random_state=1)
        # model = RandomForestClassifier()
        model.fit(X_train, y_train)
        pred_y = model.predict(X_test)
        score = roc_auc_score(y_test, pred_y)
        scores.append(score)
        print(score)

    result = np.array(scores)
    print('mean', result.mean())
    print('std', result.std(ddof=1))
    # regressor
    # mean 0.7863333333333333
    # std 0.036204953940673766

    # classifier
    # mean 0.6516666666666667
    # std 0.03162775586081207

    # グリッドサーチ


def main():
    german_data, german_target = data()
    train_test(german_data, german_target)


if __name__ == '__main__':
    main()
