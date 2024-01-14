from . import XGBModel, LGBModel
import optuna
import os


def update_model_cofig(default_config, best_params):
    for param, value in best_params.items():
        default_config[param] = value
    return default_config


class OptimParam:
    def __init__(
        self,
        model_name,
        default_config,
        X_train,
        y_train,
        X_valid,
        y_valid,
        class_weights=None,
        storage=None,
    ):
        self.model_name = model_name
        self.default_config = default_config
        self.X_train = X_train
        self.y_train = y_train
        self.X_valid = X_valid
        self.y_valid = y_valid
        self.class_weights = class_weights
        self.storage = storage

    def get_model(self, model_config, class_weights=None):
        if self.model_name == "xgboost":
            return XGBModel(model_config)
        if self.model_name == "lightgbm":
            return LGBModel(model_config)
        if self.model_name == "weight":
            return XGBModel(model_config)

    def xgboost_config(self, trial: optuna.Trial):
        xgb_params = {
            "objective": "multi:softprob",
            "seed": 42,
            "num_class": 8,
            "eta": trial.suggest_float("eta", 0.05, 0.1, step=0.01),
            "max_depth": trial.suggest_int("max_depth", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 0.95, step=0.05),
            "min_child_weight": trial.suggest_float(
                "min_child_weight", 0.1, 10, log=True
            ),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", 0.6, 0.95, step=0.05
            ),
            "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
            "alpha": 0.1,
            "lambda": 1.0
            # "alpha": trial.suggest_float("alpha", 1e-8, 1.0),
            # "lambda": trial.suggest_float("lambda", 1e-6, 10),
        }
        return xgb_params

    def lightgbm_config(self, trial: optuna.trial):
        lgb_params = {
            "objective": "multiclass",
            "seed": 42,
            "num_class": 8,
            "eta": trial.suggest_float("eta", 0.05, 0.1, step=0.01),
            "max_depth": trial.suggest_int("max_depth", 1, 10),
            "verbose": -1,
            "class_weight": "balanced",
        }
        return lgb_params

    def weight_config(self, trial: optuna.trial):
        class_weights = {
            0: trial.suggest_float("Dark_Trap", 0.9, 1.1, step=0.01),
            1: trial.suggest_float("Emo", 1.0, 2.5, step=0.01),
            2: trial.suggest_float("Hip_Hop", 0.9, 2.0, step=0.01),
            3: trial.suggest_float("Pop", 1.5, 3.0, step=0.01),
            4: trial.suggest_float("Rap", 1.0, 2.5, step=0.01),
            5: trial.suggest_float("Rnb", 1.0, 2.5, step=0.01),
            6: trial.suggest_float("Trap_Metal", 1.0, 2.5, step=0.01),
            7: trial.suggest_float("Underground_Rap", 0.9, 1.1, step=0.01),
        }
        
        return class_weights

    def get_model_config(self, trial: optuna.Trial):
        if self.model_name == "xgboost":
            return self.xgboost_config(trial)
        if self.model_name == "lightgbm":
            return self.lightgbm_config(trial)
        if self.model_name == "weight":
            return self.weight_config(trial)

    def objective(self, trial: optuna.Trial):
        model_config = self.get_model_config(trial)
        if self.model_name == "weight":
            model = self.get_model(self.default_config)
            model.train(self.X_train, self.y_train, self.X_valid, self.y_valid, model_config)
        else:
            model = self.get_model(model_config)
            # 重みあり
            if self.class_weights is not None:
                model.train(self.X_train, self.y_train, self.X_valid, self.y_valid, self.class_weights)
            else:
                model.train(self.X_train, self.y_train, self.X_valid, self.y_valid)
        (
            _,
            score,
            _,
            _,
        ) = model.evaluate(self.X_valid, self.y_valid)
        return score

    def get_best_params(self, n_trials, seed):
        if self.storage is not None:
            os.makedirs(self.storage, exist_ok=True)
            self.storage = optuna.storages.RDBStorage(
                url=f"sqlite:///{self.storage}/optuna.db",
            )
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=seed),
            storage=self.storage,
        )
        study.optimize(self.objective, n_trials=n_trials, show_progress_bar=True)
        best_params = study.best_params
        default_config = update_model_cofig(self.default_config, best_params)
        return default_config


"""         update_model_cofig(self.default_config, best_params)
        return self.default_config """