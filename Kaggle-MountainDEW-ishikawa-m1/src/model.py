import xgboost as xgb
import lightgbm as lgb
import numpy as np
import pandas as pd
import tensorflow as tf


from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping


class BaseTreeModel:
    def __init__(self, params, num_round=100, early_stopping_rounds=30):
        self.params = params
        self.num_round = num_round
        self.early_stopping_rounds = early_stopping_rounds
        self.model = None

    def train(self, X_train, y_train, X_valid, y_valid, class_weights=None):
        NotImplementedError()

    def predict(self, X):
        NotImplementedError()

    def evaluate(self, X, y):
        NotImplementedError()


class XGBModel(BaseTreeModel):
    def train(self, X_train, y_train, X_valid, y_valid, class_weights=None):
        if class_weights is not None:
            sample_weights_train = [class_weights[class_label] for class_label in y_train]
            dtrain = xgb.DMatrix(X_train, label=y_train, weight=sample_weights_train)
            dvalid = xgb.DMatrix(X_valid, label=y_valid)
        else:
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dvalid = xgb.DMatrix(X_valid, label=y_valid)
            
        watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
        self.model = xgb.train(
            self.params,
            dtrain,
            evals=watchlist,
            num_boost_round=self.num_round,
            early_stopping_rounds=self.early_stopping_rounds,
        )

    def predict(self, X_valid):
        dvalid = xgb.DMatrix(X_valid)
        y_pred_prob = self.model.predict(dvalid)
        return y_pred_prob

    def get_y_pred_labels(self, X_valid):
        y_pred_prob = self.predict(X_valid)
        y_pred_labels = np.argmax(y_pred_prob, axis=1)
        return y_pred_labels

    def evaluate(self, X_valid, y_valid):
        y_pred_labels = self.get_y_pred_labels(X_valid)
        accuracy = accuracy_score(y_valid, y_pred_labels)
        f1score = f1_score(y_valid, y_pred_labels, average="micro")
        confusion = confusion_matrix(y_valid, y_pred_labels)
        report = classification_report(y_valid, y_pred_labels)
        return accuracy, f1score, confusion, report

    def get_feature_importance(self, X_train):
        feature_names = list(X_train.columns)
        importance_dict = self.model.get_score(importance_type='weight')
        feature_importance = np.zeros(len(feature_names))

        for feature, importance in importance_dict.items():
            # feature が feature_names に存在する場合のみ処理する
            if feature in feature_names:
                index = feature_names.index(feature)
                feature_importance[index] = importance

        return feature_importance

class LGBModel(BaseTreeModel):
    def train(self, X_train, y_train, X_valid, y_valid):
        dtrain = lgb.Dataset(X_train, label=y_train)
        dvalid = lgb.Dataset(X_valid, label=y_valid)
        self.model = lgb.train(
            self.params,
            dtrain,
            valid_sets=[dtrain, dvalid],
            valid_names=["train", "eval"],
            num_boost_round=self.num_round,
            callbacks=[lgb.early_stopping(self.early_stopping_rounds)],
        )

    def predict(self, X_valid):
        y_pred_prob = self.model.predict(X_valid)
        y_pred_labels = np.argmax(y_pred_prob, axis=1)
        return y_pred_labels

    def evaluate(self, X_valid, y_valid):
        y_pred_labels = self.predict(X_valid)
        accuracy = accuracy_score(y_valid, y_pred_labels)
        f1score = f1_score(y_valid, y_pred_labels, average="micro")
        confusion = confusion_matrix(y_valid, y_pred_labels)
        report = classification_report(y_valid, y_pred_labels)
        return accuracy, f1score, confusion, report


class NN:
    def __init__(self, input_dim=13, outputs=8, lr=0.01, epochs=10, batch_size=32):
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        input_layer = Input(shape=(input_dim,))
        hidden_layer = Dense(units=64, activation="relu")(input_layer)
        output_layer = Dense(units=8, activation="softmax")(hidden_layer)
        self.model = Model(inputs=input_layer, outputs=output_layer)
        self.model.compile(
            optimizer=Adam(learning_rate=self.lr),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        self.scaler = StandardScaler()

    def train(
        self, X: pd.DataFrame, y: np.ndarray, X_valid: pd.DataFrame, y_valid: np.ndarray
    ):
        X = self.scaler.fit_transform(X)
        X_valid = self.scaler.transform(X_valid)
        y = to_categorical(y, num_classes=8)
        y_valid = to_categorical(y_valid, num_classes=8)

        callbacks = [
            EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
            # ModelCheckpoint(filepath='best_model.keras',monitor='val_loss', save_best_only=True)
        ]

        self.model.fit(
            X,
            y,
            validation_data=(X_valid, y_valid),
            batch_size=self.batch_size,
            epochs=self.epochs,
            verbose=1,
            shuffle=True,
            callbacks=callbacks,
        )

    def predict(self, X: pd.DataFrame):
        X = self.scaler.fit_transform(X)
        pred = (self.model.predict(X, batch_size=32, verbose=0),)
        return pred

    def evaluate(self, X_valid, y_valid):
        y_pred_labels = self.predict(X_valid)
        accuracy = accuracy_score(y_valid, y_pred_labels)
        f1score = f1_score(y_valid, y_pred_labels, average="micro")
        confusion = confusion_matrix(y_valid, y_pred_labels)
        report = classification_report(y_valid, y_pred_labels)
        return accuracy, f1score, confusion, report