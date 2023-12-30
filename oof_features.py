# -*- coding: utf-8 -*-
"""
Created on Sat Jun  3 18:06:37 2023

@author: jintonic
"""

import copy
import glob
import os
import pickle

import numpy as np
import pandas as pd
import torch
import umap
from sklearn.ensemble import IsolationForest

from mlutil.mlbase import BasicMLPDataSet, BasicMLPRegressor

SEED = 42


class OOFFeatureCreator:
    def __init__(self, processors, header_columns=[], experiment_name="default"):
        self.base_processors = processors
        self.header_columns = header_columns
        self.save_dir = os.path.join(
            os.environ["DATASET_ROOT_DIR"], "artifact/oof_encoder"
        )
        self._init_dir()

    def fit_transform(self, fold_generator):
        features = [[] for _ in range(len(self.base_processors))]
        for fold_index, (train, valid) in enumerate(fold_generator):
            for processor_index, processor in enumerate(self.base_processors):
                model = copy.copy(processor)
                model_path = self._get_processor_path(
                    processor=processor, fold_index=fold_index
                )
                model.fit(self._preprocess(train))
                model.save(model_path)
                transformed = model.transform(self._preprocess(valid))
                temp = valid[self.header_columns].copy()
                temp[transformed.columns] = transformed.values
                features[processor_index].append(temp)

        features = list(map(lambda l: pd.concat(l, axis=0), features))
        features = pd.concat(features, axis=1)
        return features

    def transform(self, features):
        processors = self._load_processors()
        transformed = [[] for i in range(len(processors))]
        for index, each_processors in enumerate(processors.values()):
            for model in each_processors:
                transformed[index].append(model.transform(self._preprocess(features)))

        transformed = list(map(lambda l: sum(l) / len(l), transformed))
        transformed = pd.concat(transformed, axis=1)
        temp = features[self.header_columns].copy()
        temp[transformed.columns] = transformed.values
        transformed = temp
        return transformed

    def _preprocess(self, features):
        return features.drop(columns=self.header_columns)

    def _get_processor_path(self, processor, fold_index):
        model_name = self._get_processor_name(processor)
        return os.path.join(
            self.save_dir, f"{model_name}/{model_name}_{fold_index}.pickle"
        )

    def _get_processor_name(self, processor):
        return processor.__class__.__name__.lower()

    def _load_processors(self):
        """
        Note: inference時にも呼ぶ必要はない
        """
        processors = {}
        for processor in self.base_processors:
            name = self._get_processor_name(processor)
            files = glob.glob(os.path.join(self.save_dir, name, f"{name}_*"))
            files.sort()
            assert len(files) >= 1
            processors[name] = list(map(lambda f: processor.load(f), files))
        return processors

    def _init_dir(self):
        try:
            for i in self.base_processors:
                name = self._get_processor_name(i)
                dir_ = os.path.join(self.save_dir, name)
                if not os.path.exists(dir_):
                    os.makedirs(dir_)
        # Note: kaggle notebook用
        except OSError:
            pass


class BaseProcessor:
    target_col = "y"

    def __init__(self):
        pass

    def fit(self, train):
        raise NotImplementedError()

    def transform(self, valid):
        raise NotImplementedError()

    def _preprocess(self, features):
        return features.drop(columns=self.target_col)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    def load(self, path):
        with open(path, mode="rb") as f:
            return pickle.load(f)


class TargetEncoder(BaseProcessor):
    def __init__(self, encode_cols):
        self.encode_cols = encode_cols

    def fit(self, train):
        self.encode_dict = {}
        for col in self.encode_cols:
            temp = train[[col, self.target_col]].groupby(col)[self.target_col].mean()
            self.encode_dict[col] = temp

    def transform(self, valid):
        result = pd.DataFrame()
        for col in self.encode_cols:
            encode = lambda x: self._get_encoded_value(col, x)
            result[col] = np.vectorize(encode)(valid[col].values)
        return result

    def _get_encoded_value(self, col, x):
        try:
            return self.encode_dict[col][x]
        except KeyError:
            return self.encode_dict[col].mean()


class AnomalyDetector(BaseProcessor):
    def __init__(self, n_iter=200):
        self.n_iter = n_iter

    def fit(self, train):
        self.model = IsolationForest(n_estimators=self.n_iter, random_state=SEED)
        self.model.fit(self._preprocess(train))

    def transform(self, valid):
        score = self.model.score_samples(self._preprocess(valid))
        df = pd.DataFrame(score, columns=["anomaly_score"])
        return df

    def _preprocess(self, features):
        df = super()._preprocess(features)
        df = df.fillna(-1)
        return df.values


class UMap(BaseProcessor):
    def fit(self, train):
        self.model = umap.UMAP()
        self.model.fit(self._preprocess(train))

    def transform(self, valid):
        values = self.model.transform(self._preprocess(valid))
        df = pd.DataFrame(values, columns=["umap_x", "umap_y"])
        return df

    def _preprocess(self, features):
        df = super()._preprocess(features)
        df = df.fillna(-1)
        return df.values


class DAE(BaseProcessor):
    def __init__(self, **dae_params):
        self.model = DAEModel(**dae_params)
        self.base_dir = self.model.model_dir

    def fit(self, train):
        self.model.model_dir = self.base_dir
        self.model.fit(self._preprocess(train), save_model=False)

    def transform(self, valid):
        return self.model.estimate(self._preprocess(valid))

    def _preprocess(self, features):
        df = super()._preprocess(features)
        df = df.fillna(-1)
        return df

    def save(self, path):
        self.model.model_dir = path.replace(os.path.basename(path), "")
        self.model.save_model()

    def load(self, path):
        self.model.model_dir = path.replace(os.path.basename(path), "")
        self.model.load_model()
        return self


class DAEModel(BasicMLPRegressor):
    regression = True

    def __init__(
        self,
        metric="rmse",
        loss="rmse",
        optimizer="radam",
        lr_scheduler="CosineAnnealingWarmRestarts",
        loss_kwargs={},
        optimizer_kwargs={},
        lr_scheduler_kwargs={},
        n_layer=3,
        compression_rate=0.8,
        swap_prob=0.1,
        learning_rate=1e-3,
        epochs=15,
        batch_size=128,
        early_stopping_rounds=None,
        dropout_rate=0.2,
        reg_alpha=None,
        reg_lambda=1e-4,
        weight_clip_value=1.0,
        use_best_epoch=False,
        model_dir="./model/dummy",
        learning_curve_dir="./figure/dummy",
    ):
        self.n_layer = n_layer
        self.compression_rate = compression_rate
        self.swap_prob = swap_prob
        super().__init__(
            header_columns=[],
            ignore_columns=[],
            metric=metric,
            loss=loss,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            loss_kwargs=loss_kwargs,
            optimizer_kwargs=optimizer_kwargs,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size,
            early_stopping_rounds=early_stopping_rounds,
            dropout_rate=dropout_rate,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            weight_clip_value=weight_clip_value,
            use_best_epoch=use_best_epoch,
            model_dir=model_dir,
            learning_curve_dir=learning_curve_dir,
        )

    def estimate(self, features):
        pred = self.predict(features)
        cols = [f"hidden_{i}" for i in range(pred.shape[1])]
        return pd.DataFrame(pred, columns=cols)

    def _predict(self, model, features):
        data_set = self._get_data_set(
            self._preprocess_data(features, target_col=None),
            target_col=None,
            shuffle=False,
        )
        data_loader = data_set.get_data_loader()
        model.eval()
        model.to(self._get_device())
        preds = []
        with torch.no_grad():
            for (*features,) in data_loader:
                pred = model.get_encoded_vector(*features)
                if not self.regression:
                    pred = torch.sigmoid(pred)
                preds.append(pred)
        preds = torch.cat(preds)
        preds = preds.detach().to("cpu").numpy()
        model.to("cpu")
        self._clear_gpu_cache()
        return preds

    def _get_data_set(self, df, target_col, shuffle):
        return DAEDataset(
            df=df,
            batch_size=self.batch_size,
            shuffle=shuffle,
            swap_prob=self.swap_prob,
            infer=target_col is None,
        )

    def _get_model(self, train_set):
        return DAEModule(
            feature_dim=train_set.feature_dim,
            n_layer=self.n_layer,
            compression_rate=self.compression_rate,
            dropout_rate=self.dropout_rate,
        )

    def get_empty_model(self):
        params = self._load_params(self._get_params_path())
        return DAEModule(**params.__dict__)

    def _preprocess_data(self, df, target_col):
        return super()._preprocess_data(df, target_col=None)


class DAEDataset(BasicMLPDataSet):
    def __init__(self, df, batch_size, shuffle, swap_prob, infer):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.swap_prob = swap_prob
        self.infer = infer
        self._set_feature_columns(df)
        self._set_feature_array(df)
        self._set_data_length(df)
        self._set_steps_per_epoch()

    def __getitem__(self, idx):
        features = self._get_tensor(self.features[idx, ...], dtype=torch.float)
        return features

    def _set_feature_columns(self, df):
        self.feature_dim = len(df.columns)

    def _set_feature_array(self, df):
        self.features = df.values

    def _set_data_length(self, df):
        self.data_length = len(df)

    def get_data_loader(self):
        return torch.utils.data.DataLoader(
            self,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=0,
            collate_fn=self._add_swap_noise,
            # pin_memory=True,
            # persistent_workers=True,
        )

    def _add_swap_noise(self, batch):
        features = torch.stack(batch)
        shuffled = []
        for i in range(features.shape[1]):
            shuffle_index = torch.randperm(features.shape[0])
            shuffled.append(features[:, i][shuffle_index].reshape(-1, 1))
        shuffled = torch.cat(shuffled, dim=1)

        prob = torch.full(features.shape, self.swap_prob)
        dist = torch.distributions.bernoulli.Bernoulli(prob)
        sample = dist.sample().to(torch.bool)
        sample = self._to_device(sample)

        features = torch.where(sample, shuffled, features)
        if self.infer:
            return (features,)
        else:
            return features, copy.deepcopy(features)


class DAEModule(torch.nn.Module):
    def __init__(self, feature_dim, n_layer, compression_rate, dropout_rate):
        super().__init__()
        self.feature_dim = feature_dim
        self.n_layer = n_layer
        self.compression_rate = compression_rate
        self.dropout_rate = dropout_rate

        self.activation = torch.nn.GELU()
        self.dropout = torch.nn.Dropout(dropout_rate)
        hidden_dim = int(feature_dim * compression_rate)
        self.dense_head = torch.nn.Linear(feature_dim, hidden_dim)
        self.dense_hidden = torch.nn.ModuleList(
            [torch.nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layer - 1)]
        )
        self.dense_tail = torch.nn.Linear(hidden_dim, feature_dim)

    def forward(self, x):
        x = self.dense_head(x)
        x = self.activation(x)
        x = self.dropout(x)
        for i in range(len(self.dense_hidden)):
            x = self.dense_hidden[i](x)
            x = self.activation(x)
            x = self.dropout(x)
        x = self.dense_tail(x)
        return x

    def get_encoded_vector(self, x):
        hidden = []
        x = self.dense_head(x)
        x = self.activation(x)
        x = self.dropout(x)
        hidden.append(copy.deepcopy(x))

        for i in range(len(self.dense_hidden)):
            x = self.dense_hidden[i](x)
            x = self.activation(x)
            x = self.dropout(x)
            hidden.append(copy.deepcopy(x))

        hidden = torch.cat(hidden, dim=1)
        return hidden
