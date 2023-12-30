# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 08:43:39 2023

@author: jintonic
"""

import json
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from tqdm import tqdm


class ABSFeatureGenerator:
    cat_encoder_file_name = "cat_encoder.pickle"
    feature_file_name = "features.parquet"
    feature_metadata_file_name = "feature_metadata.json"

    def __init__(self, feature_dir_name=None, cat_encoder_dir_name=None, infer=False):
        self.feature_dir_name = feature_dir_name
        self.cat_encoder_dir_name = cat_encoder_dir_name
        self.infer = infer
        self.cat_cols = []
        self.artifact_root_dir = os.path.join(
            os.environ["DATASET_ROOT_DIR"], "artifact/feature"
        )
        self._init_dir()

    def __call__(
        self, df: pd.DataFrame, save_features=False, verbose=True
    ) -> pd.DataFrame:
        if not save_features and verbose:
            print("save_features is False. features will not saved.")
        self._set_property(df)
        df = self.calc_features()
        if self.feature_dir and save_features:
            self._save_features()
        self._remove_property()
        if verbose:
            print(f"{len(df.columns)} features are generated")
        return df

    def calc_features(self) -> pd.DataFrame:
        raise NotImplementedError()

    def encode_cat_features(self, cols=None):
        if not cols:
            cols = [i for i in self.df.columns if self.df[i].dtype == "object"]
        if not self.cat_encoder_dir:
            raise ValueError("cat_encoder_dir is not set.")
        if self.infer:
            self.cat_encoder = self._load_cat_encoder()
        else:
            self.cat_encoder = self._fit_cat_encoder(cols)

        self.df[cols] = self.cat_encoder.transform(self.df[cols].values)
        self.df[cols] = self.df[cols].fillna(self.df[cols].max() + 1.0)
        self.cat_cols = cols

    def drop_duplicated_columns(self):
        drop_cols = []
        all_cols = self.df.columns
        for left_index in tqdm(range(len(all_cols))):
            if all_cols[left_index] in drop_cols:
                continue
            else:
                if len(self.df[all_cols[left_index]].unique()) == 1:
                    drop_cols.append(all_cols[left_index])
                    continue
            for right_index in range(left_index + 1, len(all_cols)):
                if all_cols[right_index] in drop_cols:
                    continue
                if (
                    self.df[all_cols[left_index]] != self.df[all_cols[right_index]]
                ).sum() == 0:
                    drop_cols.append(all_cols[right_index])
        print(f"{len(drop_cols)} features dropped due to duplication.")
        self.df = self.df.drop(columns=drop_cols)

    def reduce_memory_usage(self, verbose=True):
        numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
        start_mem = self.df.memory_usage().sum() / 1024**2
        for col in tqdm(self.df.columns):
            col_type = self.df[col].dtypes
            if col_type in numerics:
                c_min = self.df[col].min()
                c_max = self.df[col].max()
                if str(col_type)[:3] == "int":
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        self.df[col] = self.df[col].astype(np.int8)
                    elif (
                        c_min > np.iinfo(np.int16).min
                        and c_max < np.iinfo(np.int16).max
                    ):
                        self.df[col] = self.df[col].astype(np.int16)
                    elif (
                        c_min > np.iinfo(np.int32).min
                        and c_max < np.iinfo(np.int32).max
                    ):
                        self.df[col] = self.df[col].astype(np.int32)
                    elif (
                        c_min > np.iinfo(np.int64).min
                        and c_max < np.iinfo(np.int64).max
                    ):
                        self.df[col] = self.df[col].astype(np.int64)
                else:
                    if (
                        c_min > np.finfo(np.float16).min
                        and c_max < np.finfo(np.float16).max
                    ):
                        self.df[col] = self.df[col].astype(np.float16)
                    elif (
                        c_min > np.finfo(np.float32).min
                        and c_max < np.finfo(np.float32).max
                    ):
                        self.df[col] = self.df[col].astype(np.float32)
                    else:
                        self.df[col] = self.df[col].astype(np.float64)
        end_mem = self.df.memory_usage().sum() / 1024**2
        reduction = 100 * (start_mem - end_mem) / start_mem
        if verbose:
            print(
                "Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)".format(
                    end_mem, reduction
                )
            )

    def _fit_cat_encoder(self, cols):
        encoder = OrdinalEncoder(
            dtype=np.float32,
            handle_unknown="use_encoded_value",
            unknown_value=np.nan,
            encoded_missing_value=np.nan,
        )
        encoder.fit(self.df[cols].values)
        return encoder

    def _count_encoding(self, col):
        if self.df[col].dtype == "object":
            counts = self.df[col].value_counts()
            self.df[f"{col}_count"] = self.df[col].map(counts)
        else:
            raise ValueError(f'Column "{col}" is not object type.')

    def _fill_cat_na(self):
        cat_cols = [i for i in self.df.columns if self.df[i].dtype == "object"]
        self.df[cat_cols] = self.df[cat_cols].fillna("NaN")

    def _set_property(self, df: pd.DataFrame):
        self.df = df

    def _remove_property(self):
        del self.df

    def _init_dir(self):
        try:
            if self.feature_dir_name:
                self.feature_dir = os.path.join(
                    self.artifact_root_dir, self.feature_dir_name
                )
                if not os.path.exists(self.feature_dir):
                    os.makedirs(self.feature_dir)
            else:
                self.feature_dir = None
                print("feature_dir is not set.")
            if self.cat_encoder_dir_name:
                self.cat_encoder_dir = os.path.join(
                    self.artifact_root_dir, self.cat_encoder_dir_name
                )
                if not os.path.exists(self.cat_encoder_dir):
                    os.makedirs(self.cat_encoder_dir)
            else:
                self.cat_encoder_dir = None
        except OSError:
            pass

    def _get_feature_metadata(self):
        return {"cat_cols": self.cat_cols}

    def _save_features(self):
        self.df.to_parquet(self._get_feature_path())
        if hasattr(self, "cat_encoder"):
            self._save_cat_encoder()
        self._save_matadata()

    def _load_features(self):
        if self.feature_dir:
            feature_path = self._get_feature_path()
            metadata_path = self._get_feature_metadata_path()
            if os.path.exists(feature_path):
                self.df = pd.read_parquet(feature_path)
                with open(metadata_path) as f:
                    metadata = json.load(f)
                    self.cat_cols = metadata["cat_cols"]
                return True
            else:
                return False
        else:
            return False

    def _get_feature_path(self):
        return os.path.join(self.feature_dir, self.feature_file_name)

    def _get_feature_metadata_path(self):
        return os.path.join(self.feature_dir, self.feature_metadata_file_name)

    def _get_cat_encoder_path(self):
        return os.path.join(self.cat_encoder_dir, self.cat_encoder_file_name)

    def _save_matadata(self):
        with open(self._get_feature_metadata_path(), "w") as f:
            json.dump(self._get_feature_metadata(), f, indent=4)

    def _save_cat_encoder(self):
        with open(self._get_cat_encoder_path(), "wb") as f:
            pickle.dump(self.cat_encoder, f)

    def _load_cat_encoder(self):
        with open(self._get_cat_encoder_path(), mode="rb") as f:
            return pickle.load(f)


class TimeSeriesFeatureGenerator(ABSFeatureGenerator):
    def __init__(
        self,
        groupby_col=None,
        feature_dir_name=None,
        cat_encoder_dir_name=None,
        infer=False,
    ):
        self.id_col = groupby_col
        super().__init__(
            feature_dir_name=feature_dir_name,
            cat_encoder_dir_name=cat_encoder_dir_name,
            infer=infer,
        )

    def lag(self, column, window, data=None, data_key=None):
        return self._pick_column(column, data, data_key).shift(window)

    def ma(self, column, window, require_lag=None, data=None, data_key=None):
        lag = self._get_require_lag(require_lag)
        trans = lambda s: s.shift(lag).rolling(window).mean()
        return self._pick_column(column, data, data_key).transform(trans)

    def mstd(self, column, window, require_lag=None, data=None, data_key=None):
        lag = self._get_require_lag(require_lag)
        trans = lambda s: s.shift(lag).rolling(window).std()
        return self._pick_column(column, data, data_key).transform(trans)

    def diff(self, column, window, require_lag=None, data=None, data_key=None):
        lag = self._get_require_lag(require_lag)
        trans = lambda s: s.shift(lag).diff(window)
        return self._pick_column(column, data, data_key).transform(trans)

    def pct_change(self, column, window, require_lag=None, data=None, data_key=None):
        lag = self._get_require_lag(require_lag)
        trans = lambda s: s.shift(lag).pct_change(window, fill_method=None)
        picked = self._pick_column(column, data, data_key)
        org = picked.shift(lag)
        pct = picked.transform(trans)
        pct[pct == np.inf] = org[pct == np.inf]
        return pct

    def ma_diff_coef(self, column, window, require_lag=None, data=None, data_key=None):
        lag = self._get_require_lag(require_lag)
        trans = lambda s: s.shift(lag).rolling(window).mean().diff()
        return self._pick_column(column, data, data_key).transform(trans)

    def rank(self, column, key, require_lag=None, data=None, data_key=None):
        lag = self._get_require_lag(require_lag)
        temp = self.df.copy() if data is None else data.copy()
        temp_key = self.id_col if data is None else data_key
        temp["_rank"] = temp.groupby(key)[column].rank(pct=True, method="first")
        temp["_rank"] = self._pick_column("_rank", temp, temp_key).shift(lag)
        return temp["_rank"]

    def _pick_column(self, column, data=None, data_key=None):
        if data is None:
            if self.id_col:
                return self.df.groupby(self.id_col)[column]
            else:
                temp = self.df.copy()
                temp["_dummy_id"] = -1
                return temp.groupby("_dummy_id")[column]
        else:
            if data_key:
                return data.groupby(data_key)[column]
            else:
                temp = data.copy()
                temp["_dummy_id"] = -1
                return temp.groupby("_dummy_id")[column]

    def _get_require_lag(self, require_lag):
        return require_lag if require_lag else 0


class AggFeatureGenerator(ABSFeatureGenerator):
    def __init__(
        self,
        base_groupby_columns,
        sum_columns=[],
        nunique_columns=[],
        mean_columns=[],
        std_columns=[],
        min_columns=[],
        max_columns=[],
        count_columns=[],
        quantile_columns=[],
        config_dir="./data/feature_config/default/",
        infer=False,
    ):
        self.base_groupby_columns = base_groupby_columns
        self.sum_columns = sum_columns
        self.nunique_columns = nunique_columns
        self.mean_columns = mean_columns
        self.std_columns = std_columns
        self.min_columns = min_columns
        self.max_columns = max_columns
        self.count_columns = count_columns
        self.quantile_columns = quantile_columns
        self.config_dir = config_dir
        self.infer = infer
        if not infer:
            self._init_dir()

    def calc_features(self):
        self.groupby = self._groupby(self.base_groupby_columns)
        self.features = []
        self._sum()
        self._nunique()
        self._mean()
        self._std()
        self._min()
        self._max()
        self._count()
        self._quantile()

        self._rename_duplicated_features()
        features = self._concat_features()
        features = self._shape_features(features)
        print(
            f"{len(features.columns) - len(self.base_groupby_columns)} features generated."
        )
        return features

    def _groupby(self, columns):
        return self.df.groupby(columns)

    def _agg(self, columns, method, groupby_columns=None):
        if groupby_columns:
            if isinstance(groupby_columns, str):
                groupby_columns = [groupby_columns]
            groupby = self._groupby(self.base_groupby_columns + groupby_columns)
        else:
            groupby = self.groupby

        if isinstance(columns, str):
            columns = [columns]

        if isinstance(method, str):
            df = groupby[columns].agg(method)
            rename_cols = {i: f"{i}_{method}" for i in df.columns}
        else:
            df = groupby[columns].apply(method)
            rename_cols = {i: f"{i}_{method.__name__}" for i in df.columns}

        df = df.rename(columns=rename_cols)
        if groupby_columns:
            df = self._transpose_groupby(df)

        return df

    def _transpose_groupby(self, df):
        groupby_columns = [
            i for i in df.index.names if i not in ["session", "level_group"]
        ]
        feature_columns = df.columns
        df = df.reset_index()
        df["key"] = ""
        for i in groupby_columns:
            df["key"] += f"{i}_"
            df["key"] += df[i].astype(str)

        # applyで複数列返す方法わからんかった
        sessions = []
        level_groups = []
        features = []
        for (session, level_group), group in df.groupby(["session", "level_group"]):
            index = []
            for col in feature_columns:
                index.append((group["key"] + f"_{col}").values)
            features.append(
                pd.Series(
                    group[feature_columns].values.T.flatten(),
                    index=np.concatenate(index),
                )
            )
            sessions.append(session)
            level_groups.append(level_group)

        index = pd.MultiIndex.from_arrays(
            [sessions, level_groups], names=("session", "level_group")
        )
        return pd.DataFrame(features, index=index)

    def _nunique(self):
        self.features.append(self._agg(columns=self.nunique_columns, method="nunique"))

    def _sum(self):
        self.features.append(self._agg(columns=self.sum_columns, method="sum"))

    def _mean(self):
        self.features.append(self._agg(columns=self.mean_columns, method="mean"))

    def _std(self):
        self.features.append(self._agg(columns=self.std_columns, method="std"))

    def _max(self):
        self.features.append(self._agg(columns=self.max_columns, method="max"))

    def _min(self):
        self.features.append(self._agg(columns=self.min_columns, method="min"))

    def _count(self):
        self.features.append(self._agg(columns=self.count_columns, method="count"))

    def _quantile(self, q=[0.1, 0.25, 0.5, 0.75, 0.9]):
        def _method(q):
            f = lambda s: s.quantile(q)
            f.__name__ = f"quantile_{q}"
            return f

        for i in q:
            self.features.append(
                self._agg(columns=self.quantile_columns, method=_method(i))
            )

    def _concat_features(self):
        return pd.concat(self.features, axis=1)

    def _shape_features(self, features):
        if self.infer:
            feature_names = self._load_feature_names()
            miss_cols = [i for i in feature_names if i not in features.columns]
            features[miss_cols] = np.nan
            features = features[feature_names]
        else:
            drop_cols = features.nunique()
            drop_cols = drop_cols[drop_cols < 2]
            drop_cols = drop_cols.index
            features = features.drop(columns=drop_cols)
            self._save_feature_names(features)

        features = features.reset_index()
        return features

    def _rename_duplicated_features(self):
        appeared_columns = []
        for index, df in enumerate(self.features):
            for col in df.columns:
                if col in appeared_columns:
                    print(f'Column "{col}" duplicates.')
                    count = 1
                    while True:
                        new_col = f"{col}_{count}"
                        if new_col in appeared_columns:
                            count += 1
                            continue
                        else:
                            self.features[index] = self.features[index].rename(
                                columns={col: new_col}
                            )
                            break
                    appeared_columns.append(new_col)
                else:
                    appeared_columns.append(col)

    def _save_feature_names(self, features):
        config = {"feature_names": features.columns.to_list()}
        with open(os.path.join(self.config_dir, "feature_names.json"), "w") as f:
            json.dump(config, f, indent=4)

    def _load_feature_names(self):
        with open(os.path.join(self.config_dir, "feature_names.json")) as f:
            config = json.load(f)
        return config["feature_names"]

    def _init_dir(self):
        if not os.path.exists(self.config_dir):
            os.makedirs(self.config_dir)
