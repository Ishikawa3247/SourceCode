import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
from scipy.stats import mode
import pandas as pd
from src import (
    Preprocess,
    NN,
    XGBModel,
    LGBModel,
    OptimParam,
    KFoldValidation,
    StratifiedKFoldValidation,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="conf", config_name="main")
def main(config: DictConfig):
    df = pd.read_csv("dataset/train.csv")
    test_data = pd.read_csv("dataset/test.csv")
    df = Preprocess().complete_df(df)
    df_test = Preprocess().complete_df(test_data)
    test_data = Preprocess().make_test_data(test_data)

    X = df.drop("genre", axis=1)
    y = df["genre"]

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42
    )

    xgb_best_params = OmegaConf.to_container(config.model.params, resolve=True)
    xgb_model = XGBModel(xgb_best_params)
    xgb_model.train(X_train, y_train, X_valid, y_valid)
    print(X_train.shape)
    # 重要度の低い特徴量の削除
    X_train, X_valid, df_test = Preprocess().xgb_feature_importance(xgb_model, X_train, X_valid, df_test)
    print(X_train.shape)

    """
    # aaa = OptimParam("xgboost", X_train, y_train, X_valid, y_valid)
    aaa = OptimParam("lightgbm", X_train, y_train, X_valid, y_valid)
    # 本番はn_trialsの回数をもうちょっと増やす
    best_params = aaa.get_best_params(n_trials=5, seed=42)

    lgb_best_params = {
        "objective": "multiclass",
        "seed": 42,
        "num_class": 8,
        "eta": best_params["eta"],
        "max_depth": best_params["max_depth"],
        "verbose": -1,
        "class_weight": "balanced",
    }


    lgb_params = {
        "objective": "multiclass",
        "seed": 42,
        "num_class": 8,
        "eta": 0.1,
        "max_depth": 5,
        "verbose": -1,
        "class_weight": "balanced",
    }


    # lgb_model = LGBModel(lgb_params)
    lgb_model = LGBModel(lgb_best_params)
    lgb_model.train(X_train, y_train, X_valid, y_valid)
    accuracy, f1_score, confusion_matrix, classfication_report = lgb_model.evaluate(
        X_valid, y_valid
    )
    kcv = KFoldValidation(LGBModel, lgb_best_params, n_splits=3)
    models1 = kcv.validation(X_train, y_train)
    """
    
    if config.flag_weight:
        print("flag_weight is True")
        best_weights = OmegaConf.to_container(config.weight.params, resolve=True)
    else:
        best_weights = None
    
    print("best_weights: ", best_weights)

    if config.flag:
        print("flag is True")
        aaa = OptimParam(
            config.model.name,
            config.model.params,
            X_train,
            y_train,
            X_valid,
            y_valid,
            # 重みありの場合
            class_weights=best_weights,
            storage="outputs/optuna_storage",
        )
        config.model.params = aaa.get_best_params(n_trials=50, seed=42)
    else:
        pass
    xgb_best_params = OmegaConf.to_container(config.model.params, resolve=True)
    print("xgb_best_params: ", xgb_best_params)
    xgb_model = XGBModel(xgb_best_params)

    if config.flag_weight:
        xgb_model.train(X_train, y_train, X_valid, y_valid, best_weights)
    else:
        xgb_model.train(X_train, y_train, X_valid, y_valid)

    accuracy, f1_score, confusion_matrix, classfication_report = xgb_model.evaluate(
        X_valid, y_valid
    )

    if config.flag_weight:
        kcv = KFoldValidation(XGBModel, xgb_best_params, best_weights, n_splits=5)
        models2 = kcv.validation(X_train, y_train)
    else:
        kcv = KFoldValidation(XGBModel, xgb_best_params, n_splits=config.n_splits)
        models2 = kcv.validation(X_train, y_train)

    """ skcv = StratifiedKFoldValidation(XGBModel, xgb_best_params, n_splits=5)
    models = skcv.validation(X_train, y_train) """

    preds = []

    def make_prediction(models, df_test):
        for model in models:
            pred = model.predict(df_test)
            preds.append(pred)

        return preds

    def make_prediction2(models, df_test):
        for model in models:
            pred = model.get_y_pred_labels(df_test)
            preds.append(pred)

        return preds

    models1 = []
    models = models1 + models2
    """ preds = make_prediction2(models, df_test)


    # 予測のリストを numpy 配列に変換
    preds_array = np.array(preds)

    mode_preds, _ = mode(preds_array, axis=0) """

    preds = make_prediction(models, df_test)
    sum_array = np.sum(preds, axis=0)
    mean_array = sum_array / len(preds)
    mean_array_labels = np.argmax(mean_array, axis=1)

    def generate_submission_csv(
        predictions, test_data, le, output_filename="submission.csv"
    ):
        decoded_genres = le.inverse_transform(predictions)
        decoded_genres_series = pd.Series(decoded_genres.flatten(), name="Genre")
        submission = pd.concat([test_data, decoded_genres_series], axis=1)
        submission.to_csv(output_filename, index=False)

    # 関数を呼び出し、結果をCSVファイルに保存
    # generate_submission_csv(mode_preds, test_data, le, output_filename="submission1.csv")
    generate_submission_csv(
        mean_array_labels, test_data, le, output_filename="submission.csv"
    )
    """ 
    print(accuracy)
    print(f1_score)
    print(confusion_matrix)
    print(classfication_report) """


if __name__ == "__main__":
    main()
