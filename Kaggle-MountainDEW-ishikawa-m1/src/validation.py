from sklearn.model_selection import train_test_split, KFold, StratifiedKFold


class BaseValidation:
    def __init__(self, model, params, class_weights=None, n_splits=3, shuffle=True, random_state=42):
        self.model = model
        self.params = params
        self.class_weights = class_weights
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.models = []
        self.scores = []

    def validation(self, X, y):
        NotImplementedError()


class KFoldValidation(BaseValidation):
    def validation(self, X, y):
        kf = KFold(n_splits=self.n_splits, shuffle=self.shuffle,
                   random_state=self.random_state)
        for train_index, val_index in kf.split(X):
            X_train = X.iloc[train_index]
            X_valid = X.iloc[val_index]
            y_train = y[train_index]
            y_valid = y[val_index]
    
            model = self.model(self.params)
            if self.class_weights is None:
                model.train(X_train, y_train, X_valid, y_valid)

            # 重みを取り入れた場合
            else:
                print("重みあり")
                class_weights = {label: self.class_weights[label] for label in set(y_train)}
                model.train(X_train, y_train, X_valid, y_valid, class_weights)
            
            _, score, _, _, = model.evaluate(X_valid, y_valid)
            print(f"score: {score}")
            self.scores.append(score)
            self.models.append(model)

        average_score = sum(self.scores) / len(self.scores)
        print(f"average score: {average_score}")
        return self.models


class StratifiedKFoldValidation(BaseValidation):
    def __init__(self, model, params, n_splits=5, shuffle=True, random_state=42):
        super().__init__(model, params, n_splits, shuffle, random_state)

    def validation(self, X, y):
        skf = StratifiedKFold(
            n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state)
        for train_index, val_index in skf.split(X, y):
            X_train = X.iloc[train_index]
            X_valid = X.iloc[val_index]
            y_train = y[train_index]
            y_valid = y[val_index]
            model = self.model(self.params)
            model.train(X_train, y_train, X_valid, y_valid)
            _, score, _, _, = model.evaluate(X_valid, y_valid)
            print(f"score{score}")
            self.scores.append(score)
            self.models.append(model)

        average_score = sum(self.scores) / len(self.scores)
        print(f"average score{average_score}")
        return self.models
