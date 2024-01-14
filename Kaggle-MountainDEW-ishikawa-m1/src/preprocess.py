import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from itertools import combinations
# import string
# import nltk
#一度だけ実行してデータファイル作成
# nltk.download('stopwords')
# nltk.download('punkt')
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# import tensorflow_hub as hub
# import tensorflow as tf
# import texthero as hero

# from sklearn.manifold import TSNE

import matplotlib.pyplot as plt

class Preprocess:
    def __init__(self):
        pass

    def make_test_data(self, test_data):
        test_data = test_data.loc[:, ["ID"]]
        return test_data

    def make_df(self, df):
        df = df.drop(["ID", "song_name", "type"], axis=1)
        return df

    def make_calculate_two_features(self, df):
        feature_columns = df.select_dtypes(
            include=['float64', 'int64']).columns.tolist()
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(df[feature_columns])
        scaled_df = pd.DataFrame(scaled_features, columns=feature_columns)

        new_features = []
        for (feature1, feature2) in combinations(feature_columns, 2):
            new_features.append(scaled_df[feature1] + scaled_df[feature2])
            new_features.append(scaled_df[feature1] - scaled_df[feature2])
            new_features.append(scaled_df[feature1] * scaled_df[feature2])
            new_features.append(
                scaled_df[feature1] / (scaled_df[feature2] + 1e-8))

        new_features_df = pd.concat(new_features, axis=1)

        new_features_df.columns = [f'{feature1}_{operation}_{feature2}'
                                   for (feature1, feature2) in combinations(feature_columns, 2)
                                   for operation in ['plus', 'minus', 'multiplied_by', 'divided_by']]

        result_df = pd.concat([df, new_features_df], axis=1)

        return result_df

    def calculate_std(self, x, y):
        return pd.concat([x, y], axis=1).std(axis=1, ddof=1)

    def calculate_mean(self, x, y):
        return (x + y) / 2

    def calculate_median(self, x, y):
        return pd.concat([x, y], axis=1).median(axis=1)

    def calculate_q75(self, x, y):
        return pd.concat([x, y], axis=1).quantile(0.75, axis=1)

    def calculate_q25(self, x, y):
        return pd.concat([x, y], axis=1).quantile(0.25, axis=1)

    def calculate_zscore(self, x, mean, std):
        return (x - mean) / (std + 1e-3)

    def make_calculate_two_features2(self, df):
        feature_columns = df.select_dtypes(
            include=['float64', 'int64']).columns.tolist()
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(df[feature_columns])
        scaled_df = pd.DataFrame(scaled_features, columns=feature_columns)

        new_features = []
        for (feature1, feature2) in combinations(feature_columns, 2):
            f1, f2 = scaled_df[feature1], scaled_df[feature2]

            # 既存の特徴量操作
            new_features.append(f1 + f2)
            new_features.append(f1 - f2)
            new_features.append(f1 * f2)
            new_features.append(f1 / (f2 + 1e-8))

            # 新しい特徴量操作
            new_features.append(self.calculate_mean(f1, f2))
            new_features.append(self.calculate_median(f1, f2))
            new_features.append(self.calculate_q75(f1, f2))
            new_features.append(self.calculate_q25(f1, f2))
            zscore_f1 = self.calculate_zscore(
                f1, self.calculate_mean(f1, f2), self.calculate_std(f1, f2))
            new_features.append(zscore_f1)

        new_features_df = pd.concat(new_features, axis=1)

        # カラム名の更新
        new_features_df.columns = [f'{feature1}_{operation}_{feature2}'
                                   for (feature1, feature2) in combinations(feature_columns, 2)
                                   for operation in ['plus', 'minus', 'multiplied_by', 'divided_by', 'mean', 'median', 'q75', 'q25', 'zscore_f1']]

        result_df = pd.concat([df, new_features_df], axis=1)

        return result_df

    def add_feature(self, df):
        song_name_counts = df['song_name'].value_counts()
        df['song_name_unique'] = df['song_name'].apply(
            lambda x: 1 if song_name_counts[x] == 1 else 0)
        # 雰囲気は思いけど、live感が強いdarktrap
        df["key_dark"] = (1 / (df["key"] + 1)) + df["liveness"]
        df["key_dark_aco"] = (1 / (df["key"] + 1)) + df["acousticness"]
        df['danceability_tempo_ratio'] = df['danceability'] / \
            (df['tempo'] + 0.001)

        # Bin 'loudness' and 'tempo' and encode the bins
        le = LabelEncoder()
        loudness_bins = pd.qcut(df['loudness'], q=3, labels=[
            'Low', 'Medium', 'High'])
        df['loudness_bin'] = le.fit_transform(loudness_bins)
        tempo_bins = pd.qcut(df['tempo'], q=3, labels=[
                             'Low', 'Medium', 'High'])
        df['tempo_bin'] = le.fit_transform(tempo_bins)
        return df
    

    def xgb_feature_importance(self, xgb_model, X_train, X_valid, df_test, threshold=2):
        feature_importance = xgb_model.get_feature_importance(X_train)
        feature_names = X_train.columns
        # 閾値よりも低い重要度を持つ特徴量を抽出
        low_importance_features = [feature_names[i] for i, importance in enumerate(feature_importance) if i >= 13 and importance < threshold]

        X_train_filtered = X_train.drop(columns=low_importance_features)
        X_valid_filtered = X_valid.drop(columns=low_importance_features) 
        df_test_filtered = df_test.drop(columns=low_importance_features) 

        # 重要度を可視化
        self.make_importance_pdf(xgb_model, X_train_filtered)
    
        return X_train_filtered, X_valid_filtered, df_test_filtered

    def make_importance_pdf(self, xgb_model, X_train):
        feature_importance_train = xgb_model.get_feature_importance(X_train)
        feature_names = X_train.columns

        plt.figure(figsize=(100, 80))
        plt.barh(feature_names, feature_importance_train)
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title('XGBoost Feature Importance')
        plt.savefig('feature_importance_plot.pdf')

    def add_song_vec(self, df):
        if(df.shape[0] == 11011):
            song_embed = np.load("dataset/song_embed_train.npy")
            song_embed = pd.DataFrame(song_embed, columns=[f"song_embed_{i}" for i in range(song_embed.shape[1])])
            df = pd.concat([df, song_embed], axis=1)  
        else:
            song_embed = np.load("dataset/song_embed_test.npy")
            song_embed = pd.DataFrame(song_embed, columns=[f"song_embed_{i}" for i in range(song_embed.shape[1])])
            df = pd.concat([df, song_embed], axis=1)
        return df

    def complete_df(self, df):
        df = self.make_calculate_two_features2(df)
        # df = self.add_feature(df)
        df = self.add_song_vec(df)
        df = self.make_df(df)
        return df
