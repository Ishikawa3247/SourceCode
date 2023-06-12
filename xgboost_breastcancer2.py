from typing import Dict

import pandas as pd
import xgboost as xgb
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split


def get_breast_cancer_data(random_state=42):
    data = load_breast_cancer()

    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["class"] = data.target
    train, test = train_test_split(
        df, stratify=df["class"], random_state=random_state)
    train, val = train_test_split(
        train, stratify=train["class"], random_state=random_state)
    df_data = {"train": train, "val": val, "test": test,
               "feature_name": data.feature_names}
    return df_data


def train(data: Dict[str, pd.DataFrame]):
    model = xgb.XGBClassifier(early_stopping_rounds=20, random_state=42)
    feature_names = data["feature_name"]
    model.fit(
        data["train"][feature_names],
        data["train"]["class"],
        eval_set=[(data["val"][feature_names], data["val"]["class"])],
        eval_metric='auc',
    )
    return model


def test(model, data: Dict[str, pd.DataFrame]):
    feature_names = data["feature_name"]
    pred = model.predict(data["test"][feature_names])
    pred_proba = model.predict_proba(data["test"][feature_names])

    acc = accuracy_score(data["test"]["class"], pred) * 100
    auc = roc_auc_score(
        data["test"]["class"].values.tolist(), pred_proba[:, 1]) * 100

    print(f"test/ACC: {acc:.2f} | test/AUC: {auc:.2f}")


def main():
    data = get_breast_cancer_data()
    model = train(data)
    test(model, data)


if __name__ == "__main__":
    main()
