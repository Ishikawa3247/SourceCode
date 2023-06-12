import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
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
    le = preprocessing.LabelEncoder()
    german_target = le.fit_transform(german_target)
    X_train, X_test, y_train, y_test = train_test_split(
        german_data, german_target, test_size=0.2, random_state=42)

    splitdata = {"X_train": X_train, "X_test": X_test,
                 "y_train": y_train, "y_test": y_test}
    return splitdata


def train_test(splitdata):
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    # kf = KFold(n_splits=10, shuffle=True, random_state=42)
    scores = []
    for train_id, test_id in kf.split(splitdata["X_train"], splitdata["y_train"]):

        X_train = splitdata["X_train"].iloc[train_id]
        X_test = splitdata["X_train"].iloc[test_id]
        y_train = splitdata["y_train"][train_id]
        y_test = splitdata["y_train"][test_id]
        eval_set = [(X_test, y_test)]

        xgb_model = XGBClassifier(early_stopping_rounds=20)
        xgb_model.fit(X_train,
                      y_train,
                      eval_set=eval_set,
                      verbose=True)
        y_pred = xgb_model.predict(splitdata["X_test"])
        score = roc_auc_score(splitdata["y_test"], y_pred)
        scores.append(score)
        print(score)

    result = np.array(scores)
    print('mean', result.mean())
    print('std', result.std(ddof=1))


# グリッドサーチ


def main():
    splitdata = data()
    train_test(splitdata)


if __name__ == '__main__':
    main()
