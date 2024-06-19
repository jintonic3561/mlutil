#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 16:25:35 2022

@author: system01
"""

# import multiprocessing as mp
# import concurrent.futures as cf
import copy
import datetime as dt
import gc
import glob
import inspect
import math
import os
import pickle
from collections import OrderedDict, namedtuple
from typing import Callable, Optional, Tuple

import catboost as cb
import lightgbm as lgb
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.metrics as sk_metrics
import torch
from tqdm import tqdm

try:
    import settings as args
except ModuleNotFoundError:
    from . import settings as args


class MLBaseMeta(type):
    # name: class名, bases: 親クラスの配列, attrs: attributesのdict
    def __new__(cls, name, bases, attrs):
        # 自身の場合をハンドルしないと無限ループする(正直よくわかってない)
        if name != "MLBase":
            # 子クラスでの定義を強制
            base_names = list(map(lambda X: X.__name__, bases))
            if "MLBase" in base_names:
                for i in ["fit", "_predict", "_load_model"]:
                    if not attrs.get(i):
                        message = f'{name} has no attribute "{i}". Define this in the class.'
                        raise AttributeError(message)

            # modelを返す関数の返り値をself properyにセット
            # modelに渡すDataFrameからignore_columnsをdrop
            if fit := attrs.get("fit"):
                fit = cls.set_model_to_self_attr(fit)
                fit = cls.drop_ignore_columns(drop_target=False)(fit)
                attrs["fit"] = fit
            if predict := attrs.get("_predict"):
                predict = cls.drop_ignore_columns(drop_target=True)(predict)
                attrs["_predict"] = predict

        return super().__new__(cls, name, bases, attrs)

    @classmethod
    def set_model_to_self_attr(cls, func):
        def wrapper(self, *args, **kwargs):
            res = func(self, *args, **kwargs)
            if res is not None:
                self.model = [res]
                return res

        return wrapper

    @classmethod
    def drop_ignore_columns(cls, drop_target):
        def wrapper(func):
            def _drop_columns(self, df, columns):
                ignore_columns = [i for i in columns if i in df.columns]
                if drop_target and self.target_col in df.columns:
                    ignore_columns.append(self.target_col)
                return df.drop(columns=ignore_columns)

            def _wrapper(self, *args, **kwargs):
                # tupleはimutableなのでlistに変換してからdrop処理
                args = list(args)
                for i in range(len(args)):
                    if isinstance(args[i], pd.DataFrame):
                        args[i] = _drop_columns(self, df=args[i], columns=self.ignore_columns)

                for key, value in kwargs.items():
                    if isinstance(value, pd.DataFrame):
                        value = _drop_columns(self, df=value, columns=self.ignore_columns)
                        kwargs[key] = value

                args = tuple(args)
                return func(self, *args, **kwargs)

            return _wrapper

        return wrapper


class MLBase(object, metaclass=MLBaseMeta):
    pred_col = "pred"
    target_col = "y"
    model_base_name = ""
    model_file_extention = ""
    regression = None
    metric_type = None

    def __init__(self, header_columns, ignore_columns, experiment_name=None):
        # Note: 両者が同じインスタンスだとtargetが両方に加わってしまう
        self.header_columns = self._to_list(copy.copy(header_columns))
        self.ignore_columns = self._to_list(copy.copy(ignore_columns))
        if self.target_col not in self.header_columns:
            self.header_columns.append(self.target_col)
        self._init_artifact_path(experiment_name)
        assert self.model_base_name
        assert self.model_file_extention
        assert self.regression is not None
        assert self.metric_type is not None

    def fit(
        self,
        train: pd.DataFrame,
        valid: Optional[pd.DataFrame] = None,
        test: Optional[pd.DataFrame] = None,
        save_model: bool = False,
    ) -> any:
        """
        Parameters
        ----------
        train : pd.DataFrame
            Training data.
        valid : Optional[pd.DataFrame]
            Validation data.
        test : Optional[pd.DataFrame]
            Test data. used only showing score.
        save_model : Optional[bool]
            If True, save trained model.
        Returns
        -------
        any
            Return model.
        """
        raise NotImplementedError()

    def _predict(self, model: any, features: pd.DataFrame) -> np.ndarray:
        """
        Parameters
        ----------
        features : pd.DataFrame
            Feature DataFrame for prediction.
        model: any
            each model.
        Returns
        -------
        np.ndarray
            Return a 1D array of the same length as the features.
        """
        raise NotImplementedError()

    def _save_model(self, model: any, path: str):
        """
        Parameters
        ----------
        model : any
        path: str

        Returns
        -------
        None.
        """
        raise NotImplementedError()

    def _load_model(self, path: str) -> any:
        """
        Parameters
        ----------
        path : str

        Returns
        -------
        model
        """
        raise NotImplementedError()

    def predict(self, features):
        preds = []
        for i in self.model:
            preds.append(self._predict(model=i, features=features))
        return self._ensemble_predictions(preds)

    # multiprocessing 動かん
    # def predict(self, features):
    #     preds = mp.Manager().dict()
    #     jobs = []
    #     for i, model in enumerate(self.model):
    #         p = mp.Process(target=self._wrap_predict, args=(i, model, features, preds))
    #         jobs.append(p)
    #         p.start()
    #     for proc in jobs:
    #         proc.join()

    #     preds = list(preds.values())
    #     return self._ensemble_predictions(preds)

    # def _wrap_predict(self, index, model, features, share_dict):
    #     share_dict[index] = self._predict(model=model, features=features)
    #     print(f"Process {index} finished.")

    # concurrent.futures 動かん
    # def predict(self, features):
    #     with cf.ProcessPoolExecutor() as executor:
    #         preds = executor.map(
    #             self._wrap_predict, zip(self.model, [features.copy()] * len(self.model))
    #         )
    #     preds = list(preds)
    #     return self._ensemble_predictions(preds)

    # def _wrap_predict(self, args):
    #     model, features = args
    #     return self._predict(model=model, features=features)

    def estimate(self, features: pd.DataFrame, include_target=True) -> pd.DataFrame:
        """
        Parameters
        ----------
        features : pd.DataFrame
            Feature DataFrame for estimation.

        Returns
        -------
        pd.DataFrame
            prediction Result with header.
        """

        if include_target:
            header = self.header_columns
        else:
            header = [i for i in self.header_columns if i != self.target_col]

        result = features[header].copy()
        result[self.pred_col] = self.predict(features)
        return result

    def evaluate(self, features: pd.DataFrame, return_prediction: bool = False) -> Tuple[float, pd.DataFrame]:
        """
        Parameters
        ----------
        features : pd.DataFrame
            Feature DataFrame for evaluation.
        return_prediction: bool, optional
            if True, return model's prediction. The default is False.

        Returns
        -------
        metric: float
            evaluated value.
        prediction: pd.DataFrame
            model's prediction. return only if return_prediction is True.
        """

        result = self.estimate(features)
        metric = self._calc_metric(result)

        if return_prediction:
            return metric, result
        else:
            return

    def save_model(self):
        for i in self.model:
            path = self._get_new_file_path(
                self.model_dir,
                base_name=self.model_base_name,
                extention=self.model_file_extention,
            )
            self._save_model(model=i, path=path)

    def load_model(self):
        files = glob.glob(os.path.join(self.model_dir, f"*.{self.model_file_extention}"))
        files.sort()
        assert len(files) >= 1
        self.model = list(map(lambda f: self._load_model(f), files))

    def cv(
        self,
        fold_generator: Callable,
        calc_permutation_importance: bool = False,
        permutaion_importance_tiral_num: int = 1,
        save_model: bool = True,
    ) -> namedtuple:
        """
        Note:
        Classificationの場合、閾値最適化や各種指標の算出はoof自体を用いている。
        CVが過剰評価される場合、ぶれが大きい場合などに改善を検討する。

        Parameters
        ----------
        fold_generator : Callable
            The function yields train, valid DataFrame.
        calc_permutation_importance : bool, optional
            If True, calculate Permutation Importance. The default is False.
        permutaion_importance_tiral_num : int, optional
            Calculate Permutaion Importance for each feature a given number of times. The default is 2.
        save_model : bool, optional
            If True, save each models.

        Returns
        -------
        namedtuple('Result', ['metrics': list,
                              'cv_preds': List[pd.DataFrame],
                              'permutaion_importance': pd.DataFrame])
        """

        if self._cached_model():
            model = copy.copy(self.model)
            use_trained_model = True
        else:
            model = []
            use_trained_model = False

        metrics = []
        cv_preds = []
        first = True
        for index, (train, valid) in enumerate(fold_generator):
            gc.collect()
            print(f"CV fold {index}")
            if first:
                first = False
                feature_columns = train.columns
                if calc_permutation_importance:
                    feature_length = len(feature_columns) - len(self.ignore_columns) - 1
                    importance_matrix = [[] for _ in range(feature_length)]
                else:
                    importance_matrix = None

            if use_trained_model and index < len(model):
                self.model = [model[index]]
            else:
                model.append(self.fit(train=train, valid=valid, save_model=save_model))

            del train
            metric, pred = self.evaluate(valid, return_prediction=True)
            del valid
            print(f"CV fold {index} metric: {metric:.4f}")
            pred.insert(0, "cv_id", index)
            metrics.append(metric)
            cv_preds.append(pred)

            if calc_permutation_importance:
                result = self._calc_permutation_importance(data.valid, trial_num=permutaion_importance_tiral_num)
                importance_matrix = list(map(lambda x, y: x + y, importance_matrix, result))

        if calc_permutation_importance:
            permutaion_importance = self._shape_permutaion_matrix(importance_matrix, feature_columns=feature_columns)
        else:
            permutaion_importance = None

        print("CV result:")
        print(f"raw: {[round(i, 4) for i in metrics]}")
        # print(f'mean: {np.array(metrics).mean():.4f}')
        # print(f'std: {np.array(metrics).std():.4f}')
        # print(f"sharpe: {np.array(metrics).mean() / np.array(metrics).std() + 1:.4f}")

        self.model = model
        oof = pd.concat(cv_preds, ignore_index=True, axis=0)
        if save_model:
            self._save_oof_pred(oof)

        Result = namedtuple(
            "Result",
            ["cv_metrics", "oof_metrics", "cv_preds", "permutaion_importance"],
        )

        if self.regression:
            oof_metric = self._calc_metric(oof)
            return Result(metrics, oof_metric, cv_preds, permutaion_importance)
        else:
            self.plot_lift_chart(oof)
            self.plot_prob_dist(oof)
            clf_metrics = self.calc_classification_metrics(oof)
            return Result(metrics, clf_metrics, cv_preds, permutaion_importance)

    def plot_learning_curve(
        self,
        metric_dict: dict,
        title: Optional[str] = "Learning Curve",
        save_fig: bool = True,
    ):
        """
        Parameters
        ----------
        metric_dict : dict
            Dictionary which key is a name of metric and value is an array of metric.
        title : Optional[str]
            Figure title. The default is None.
        save_fig : bool
            If you want save figure, specify this argument True. The default is False.
        Returns
        -------
        None.

        """

        fig, ax = plt.subplots()
        first = True
        count = 0
        appeared_metrics = {}
        for key, value in metric_dict.items():
            for metric, array in value.items():
                if metric[0] == "_":
                    metric = metric[1:]
                plot_label = f"{key}'s {metric}"
                count += 1
                if join_ax := appeared_metrics.get(metric):
                    join_ax.plot(array, label=plot_label, color=matplotlib.cm.Set1.colors[count])
                else:
                    if first:
                        ax.plot(
                            array,
                            label=plot_label,
                            color=matplotlib.cm.Set1.colors[count],
                        )
                        ax.set_ylabel(metric)
                        appeared_metrics[metric] = ax
                        first = False
                    else:
                        ax_t = ax.twinx()
                        ax_t.plot(
                            array,
                            label=plot_label,
                            color=matplotlib.cm.Set1.colors[count],
                        )
                        ax_t.set_ylabel(metric)
                        ax_t.spines["right"].set_position(("axes", 1 + (count - 2) * 0.15))
                        appeared_metrics[metric] = ax_t

        handlers = []
        labels = []
        for i in appeared_metrics.values():
            handler, label = i.get_legend_handles_labels()
            handlers += handler
            labels += label
        ax.legend(handlers, labels, bbox_to_anchor=(1.1, 1), loc="upper left", borderaxespad=0)
        if title is not None:
            plt.title(title)
        if save_fig:
            path = self._get_new_file_path(self.learning_curve_dir, base_name="lc", extention="png")
            plt.savefig(path)

        plt.show(block=False)
        plt.clf()

    def plot_corr_coef(self, result, pred_col, target_col):
        corr_coef = np.corrcoef(result[pred_col], result[target_col])[0][1]
        corr_coef = round(corr_coef, 3)
        min_ = result[target_col].min()
        max_ = result[pred_col].max()
        plt.scatter(result[pred_col], result[target_col])
        plt.xlabel(pred_col)
        plt.ylabel(target_col)
        plt.title(f"Prediction vs Target of {target_col}")
        plt.text(
            x=max_,
            y=min_,
            s=f"r = {corr_coef}",
            horizontalalignment="right",
            verticalalignment="bottom",
        )

        path = self._get_new_file_path(self.learning_curve_dir, "scatter", "png")
        plt.savefig(path)
        plt.show(block=False)
        plt.clf()

        return corr_coef

    def plot_lift_chart(self, result, bins=15):
        target_cols, pred_cols = self._get_target_pred_columns(result)
        df = result.copy()
        for index, (target_col, pred_col) in enumerate(zip(target_cols, pred_cols)):
            df["bin"] = pd.qcut(df[pred_col], bins, labels=False, duplicates="drop")
            temp = df.groupby("bin")[[pred_col, target_col]].mean().sort_index()
            plt.plot(temp.index.values, temp[pred_col], label="pred")
            plt.plot(temp.index.values, temp[target_col], label="target")
            plt.xlabel("pred_bin")
            plt.ylabel("prob")
            plt.title(f"Lift Chart for class {index}")
            plt.legend()

            path = self._get_new_file_path(self.learning_curve_dir, "lift_chart", "png")
            plt.savefig(path)
            plt.show(block=False)
            plt.clf()

    def plot_prob_dist(self, result, bins=50):
        target_cols, pred_cols = self._get_target_pred_columns(result)
        for index, (target_col, pred_col) in enumerate(zip(target_cols, pred_cols)):
            labels = result[target_col].sort_values().unique()
            for label_index, label in enumerate(labels):
                df = result[result[target_col] == label]
                plt.hist(
                    df[pred_col],
                    bins=bins,
                    label=label,
                    color=matplotlib.cm.Set1.colors[label_index],
                    alpha=0.7,
                )
            plt.xlabel("prob")
            plt.ylabel("density")
            plt.title(f"Distribution for class {index}")
            plt.legend()

            path = self._get_new_file_path(self.learning_curve_dir, "prob_dist", "png")
            plt.savefig(path)
            plt.show(block=False)
            plt.clf()

    # TODO: 多値分類への対応
    def calc_classification_metrics(self, result, optimize_method="f1", average="binary"):
        best_threshold = self._find_best_threshold(result, method=optimize_method, average=average)
        target = result[self.target_col]
        pred = (result[self.pred_col] > best_threshold).astype(int)
        cm = sk_metrics.confusion_matrix(target, pred)
        f1 = sk_metrics.f1_score(target, pred, average=average)
        precision = sk_metrics.precision_score(target, pred, average=average)
        recall = sk_metrics.recall_score(target, pred, average=average)
        print("Confusion matrix:")
        print(cm)
        print(f"f1 score: {f1:.3f}")
        print(f"precision: {precision:.3f}")
        print(f"recall: {recall:.3f}")

        Result = namedtuple("Result", ["best_threshold", "f1", "precision", "recall"])
        return Result(best_threshold, f1, precision, recall)

    # TODO: 多値分類への対応
    def _find_best_threshold(self, result, method, average):
        obj_func = lambda f, x,: f(
            result[self.target_col],
            (result[self.pred_col] > x).astype(int),
            average=average,
        )
        if method == "f1":
            objective = lambda thr: obj_func(sk_metrics.f1_score, thr)
        elif method == "precision":
            objective = lambda thr: obj_func(sk_metrics.precision_score, thr)
        elif method == "recall":
            objective = lambda thr: obj_func(sk_metrics.recall_score, thr)
        else:
            raise NotImplementedError()

        thrs = np.arange(0.0, 1.0, 0.005)
        scores = np.vectorize(objective)(thrs)
        best_threshold = thrs[np.argmax(scores)]
        return best_threshold

    def _calc_metric(self, result):
        if self.regression:
            corr_coef = self.plot_corr_coef(result, pred_col=self.pred_col, target_col=self.target_col)
            if self.metric_type == "corr_coef":
                metric = corr_coef
            elif self.metric_type == "rmse":
                metric = sk_metrics.mean_squared_error(result[self.target_col], result[self.pred_col], squared=False)
            elif self.metric_type == "mae":
                metric = sk_metrics.mean_absolute_error(result[self.target_col], result[self.pred_col])
            else:
                raise NotImplementedError()
        else:
            self.plot_lift_chart(result)
            self.plot_prob_dist(result)
            target_cols, pred_cols = self._get_target_pred_columns(result)
            if self.metric_type == "log_loss":
                metric = sk_metrics.log_loss(result[target_cols], result[pred_cols])
            elif self.metric_type == "auc":
                metric = sk_metrics.roc_auc_score(result[target_cols], result[pred_cols])
            elif self.metric_type == "f1_macro":
                best_threshold = self._find_best_threshold(result, method="f1", average="macro")
                target = result[self.target_col]
                pred = (result[self.pred_col] > best_threshold).astype(int)
                metric = sk_metrics.f1_score(target, pred, average="macro")
                print(f"Best threshold: {best_threshold:.4f}")
            else:
                raise NotImplementedError()
        return metric

    def _get_target_pred_columns(self, result):
        target_cols = [i for i in result.columns if f"{self.target_col}_" in i]
        pred_cols = [i for i in result.columns if f"{self.pred_col}_" in i]
        if not target_cols:
            target_cols = [self.target_col]
        if not pred_cols:
            pred_cols = [self.pred_col]
        return target_cols, pred_cols

    def _calc_permutation_importance(self, valid: pd.DataFrame, trial_num: int):
        original_score = self.evaluate(valid)
        importance_matrix = []
        for column in tqdm(valid.columns):
            if column in self.ignore_columns + [self.target_col]:
                continue

            importance_array = []
            for _ in range(trial_num):
                np.random.shuffle(valid[column].values)
                score = self.evaluate(valid)
                permutaion_importance = original_score - score
                importance_array.append(permutaion_importance)

            importance_matrix.append(importance_array)

        return importance_matrix

    def _shape_permutaion_matrix(self, importance_matrix, feature_columns):
        importance_matrix = np.array(importance_matrix)
        result = pd.DataFrame()
        result["feature_name"] = [i for i in feature_columns if i not in self.ignore_columns + [self.target_col]]
        result["permutaion_importance_mean"] = importance_matrix.mean(axis=1)
        result["permutaion_importance_std"] = importance_matrix.std(axis=1)
        result["lower_limit_score"] = result["permutaion_importance_mean"] - result["permutaion_importance_std"]

        return result

    def _to_list(self, arg):
        if arg is None:
            return []
        else:
            return arg if isinstance(arg, list) else [arg]

    def _preprocess_data(self, df, train=True):
        return df
        # if train:
        #     return detect_outliers(df, target_column=self.target_col, iqr_coef=2)
        # else:
        #     return df

    def _ensemble_predictions(self, preds):
        return np.stack(preds, axis=0).mean(axis=0)

    def _init_artifact_path(self, experiment_name, default_root_dir="/work/data/"):
        if not experiment_name:
            today = dt.datetime.today().strftime("%Y%m%d%H%M%S")
            experiment_name = today

        # for kaggle setting
        if "DATASET_ROOT_DIR" in os.environ:
            root_dir = os.environ["DATASET_ROOT_DIR"]
        else:
            root_dir = default_root_dir

        self.artifact_root_dir = os.path.join(root_dir, "artifact", experiment_name)
        self.model_dir = os.path.join(self.artifact_root_dir, "model")
        self.learning_curve_dir = os.path.join(self.artifact_root_dir, "figure")
        self.oof_dir = os.path.join(self.artifact_root_dir, "oof_pred")

        for i in [self.model_dir, self.learning_curve_dir, self.oof_dir]:
            if not os.path.exists(i):
                try:
                    os.makedirs(i)
                except OSError as e:
                    print(e)

    def _get_new_file_path(self, dir_, base_name, extention):
        files = glob.glob(os.path.join(dir_, f"{base_name}_*.{extention}"))
        if files:
            count = list(
                map(
                    lambda f: int(os.path.basename(f).split(".")[0].split("_")[-1]),
                    files,
                )
            )
            max_count = max(count)
            return os.path.join(dir_, f"{base_name}_{str(max_count + 1)}.{extention}")
        else:
            return os.path.join(dir_, f"{base_name}_1.{extention}")

    def _cached_model(self):
        try:
            self.load_model()
            print(f"Pre-trained {len(self.model)} models loaded.")
            return True
        except AssertionError:
            return False

    def _save_oof_pred(self, oof):
        oof.to_csv(os.path.join(self.oof_dir, "oof_pred.csv"), index=False)


class BasicLGBRegressor(MLBase):
    # Note: Don't change these parameters.
    model_base_name = "lgb"
    model_file_extention = "txt"
    regression = True
    metric_type = ""

    def __init__(
        self,
        header_columns,
        ignore_columns,
        categorical_columns=[],
        train_params={},
        metric="corr_coef",
        loss="rmse",
        corrcoef_reg_alpha=1e-3,
        early_stopping_rounds=None,
        experiment_name=None,
        show_train_learning_curve=True,
        show_valid_learning_curve=True,
        show_test_learning_curve=False,
        use_gpu=True,
    ):
        """
        Parameters
        ----------
        header_columns: List[str]
            Feature columns used for prediction header.
        ignore_columns: List[str]
            Feature columns that ignore when fitting model.
        categorical_columns: List[str], optional
            Categorical feature columns. The default is None.
        train_params: Dict[str, any], optional
            LGB parameter for fitting model. The default is None.
        metric: str, optional
            Validation metric function. You can choice from ['rmse', 'corr_coef']. The default is 'corr_coef'.
        loss: str, optional
            Loss function for fit. You can choice from ['rmse', 'mae', 'corr_coef']. The default is 'rmse'.
        corrcoef_reg_alpha: float, optional
            Regularization coefficient used for corrcoef loss. The default is 1e-3.
        early_stopping_rounds: int, optional
            LGB early_stopping_rounds parameter.
        experiment_name: str, optional
            used for artifact saving directory name.
        show_train_learning_curve: bool, optional
            If True, plot learning curve of training data. The default is True.
        show_valid_learning_curve: bool, optional
            If True, plot learning curve of validation data. The default is True.
        show_test_learning_curve: bool, optional
            If True, plot learning curve of test data. The default is False.
        use_gpu: bool. optional
            If True, use GPU for training. The default is True.
        """

        super().__init__(header_columns=header_columns, ignore_columns=ignore_columns)
        self.categorical_columns = categorical_columns if categorical_columns else "auto"
        self.corrcoef_reg_alpha = corrcoef_reg_alpha
        self.early_stopping_rounds = early_stopping_rounds
        self.use_best_iteration = self.early_stopping_rounds is not None
        self.show_train_learning_curve = show_train_learning_curve
        self.show_valid_learning_curve = show_valid_learning_curve
        self.show_test_learning_curve = show_test_learning_curve

        self.core_params = {
            "task": "train",
            "objective": "regression",
            "metric": "None",
            "device": "gpu" if use_gpu else "cpu",
            "boosting_type": "gbdt",
            "verbosity": -1,
        }
        self._set_train_params(params=train_params)
        self._init_artifact_path(experiment_name=experiment_name)
        self._init_metric(metric)
        self._init_loss(loss)
        self.weights = None

    def fit(self, train, valid=None, test=None, save_model=False):
        params = dict(**self.core_params, **self.train_params)
        early_stopping_rounds = None if valid is None else self.early_stopping_rounds

        evals_result = {}
        valid_sets = []
        valid_names = []
        # Note: trainはcvでも一回しか使われないため、上書きする
        train = self.make_datasets(train)
        if valid is not None and self.show_valid_learning_curve:
            valid_sets.append(self.make_datasets(valid.copy(), reference=train))
            valid_names.append("valid")
        if test is not None and self.show_test_learning_curve:
            valid_sets.append(self.make_datasets(test.copy(), reference=train))
            valid_names.append("test")
        if self.show_train_learning_curve:
            valid_sets.append(train)
            valid_names.append("train")

        model = lgb.train(
            params=params,
            train_set=train,
            valid_sets=valid_sets,
            valid_names=valid_names,
            num_boost_round=self.n_iter,
            feval=self.metric,
            fobj=self.loss,
            early_stopping_rounds=early_stopping_rounds,
            callbacks=[LgbProgressBarCallback()],
            evals_result=evals_result,
        )

        if len(evals_result) != 0:
            self.plot_learning_curve(evals_result)

        if save_model:
            path = self._get_new_file_path(
                dir_=self.model_dir,
                base_name=self.model_base_name,
                extention=self.model_file_extention,
            )
            self._save_model(model, path)

        return model

    def _predict(self, model, features):
        df = self._preprocess_data(features, train=False)
        num_iteration = model.best_iteration if self.use_best_iteration else None
        return model.predict(df, num_iteration=num_iteration)

    def _save_model(self, model, path):
        num_iteration = model.best_iteration if self.use_best_iteration else None
        model.save_model(path, num_iteration=num_iteration)

    def _load_model(self, path):
        return lgb.Booster(model_file=path)

    def make_datasets(self, df, reference=None):
        train = reference is None
        df = self._date_to_int(df)
        df = self._preprocess_data(df, train=train)
        y = df[self.target_col].copy()
        df.drop(columns=self.target_col, inplace=True)
        params = {"header": True}

        ds = lgb.Dataset(
            df,
            y,
            params=params,
            categorical_feature=self.categorical_columns,
            weight=self.weights if train else None,
            reference=reference,
            free_raw_data=False,
        )
        return ds

    def get_feature_importance(self):
        importance = []
        for i in self.model:
            df = pd.DataFrame()
            df["feature"] = i.feature_name()
            df["importance"] = i.feature_importance()
            df = df.sort_values("importance", ascending=False)
            importance.append(df)
        return importance

    # lgb.Datasetがignore_columnに対しても型チェックを行い、date型に対応していないため
    def _date_to_int(self, df):
        for i in df.columns:
            if df[i].dtype == "datetime64[ns]":
                df[i] = df[i].apply(lambda date: int(date.strftime("%y%m%d%H%M%S")))
                df[i] = df[i].astype(("int32"))

        return df

    def _rmse_metric(self, pred: np.ndarray, ds: lgb.Dataset):
        y = ds.get_label()
        rmse = math.sqrt(np.square(y - pred).sum() / len(y))
        # eval_name, eval_result, is_higher_better
        return "rmse", rmse, False

    def _mae_metric(self, pred: np.ndarray, ds: lgb.Dataset):
        y = ds.get_label()
        mae = np.abs(y - pred).sum() / len(y)
        # eval_name, eval_result, is_higher_better
        return "mae", mae, False

    def _corr_coef_metric(self, pred: np.ndarray, ds: lgb.Dataset):
        y = ds.get_label()
        corr_coef = np.corrcoef(y, pred)[0][1]
        # eval_name, eval_result, is_higher_better
        return "corr_coef", corr_coef, True

    def _corr_coef_loss(self, pred: np.ndarray, ds: lgb.Dataset):
        y = ds.get_label()

        # 分散0対策
        if (pred == 0).sum() == len(pred):
            grad = pred - y
            hess = np.ones(len(grad))
            return grad, hess

        pred_tensor = torch.tensor(pred.reshape(1, pred.shape[0]), requires_grad=True)
        y_tensor = torch.tensor(y.reshape(1, y.shape[0]))
        cat = torch.cat([pred_tensor, y_tensor], axis=0)
        corr_coef = torch.corrcoef(cat)[0][1]
        corr_coef = torch.nan_to_num(corr_coef)
        mean_diff = torch.abs(torch.mean(pred_tensor) - torch.mean(y_tensor))
        std_diff = torch.abs(torch.std(pred_tensor) - torch.std(y_tensor))
        loss = -corr_coef + self.corrcoef_reg_alpha * (mean_diff + std_diff)

        # calculate gradient and convert to numpy
        loss_grads = torch.autograd.grad(loss, pred_tensor)[0]
        loss_grads = loss_grads.to("cpu").detach().numpy()

        # return gradient and ones instead of Hessian diagonal
        return loss_grads[0], np.ones(loss_grads.shape)[0]

    def _init_metric(self, metric):
        """
        self.metricは学習中にLGBインスタンスが使用する関数。
        MLBase._calc_metricはevaluate関数内で使われる可視化を含む評価関数。
        """

        if metric == "corr_coef":
            self.metric = self._corr_coef_metric
            self.metric_type = "corr_coef"
        elif metric == "rmse":
            self.metric = self._rmse_metric
            self.metric_type = "rmse"
        elif metric == "mae":
            self.metric = self._mae_metric
            self.metric_type = "mae"
        else:
            if isinstance(metric, Callable):
                self.metric = metric
                print("Information: Custom metric set. Did you implement MLBase._calc_metric function?")
            else:
                raise ValueError(f'Invalid metric "{metric}".')

    def _init_loss(self, loss):
        if loss == "rmse":
            self.core_params["objective"] = "regression"
            self.loss = None
        elif loss == "mae":
            self.core_params["objective"] = "regression_l1"
            self.loss = None
        elif loss == "corr_coef":
            self.loss = self._corr_coef_loss
        else:
            if isinstance(loss, Callable):
                self.loss = loss
            else:
                raise ValueError("Invalid loss function.")

    def _set_train_params(self, params):
        self.train_params = copy.copy(args.lgb_default_train_params)
        if params:
            self.train_params = {**self.train_params, **params}
        self.n_iter = self.train_params.pop("n_iter")


class BasicLGBClassifier(BasicLGBRegressor):
    regression = False
    metric_type = "log_loss"

    def __init__(
        self,
        header_columns,
        ignore_columns,
        num_class,
        categorical_columns=[],
        train_params={},
        metric="auc",
        loss="log_loss",
        early_stopping_rounds=None,
        experiment_name=None,
        show_train_learning_curve=True,
        show_valid_learning_curve=True,
        show_test_learning_curve=False,
        use_gpu=True,
    ):
        """
        Parameters
        ----------
        header_columns: List[str]
            Feature columns used for prediction header.
        ignore_columns: List[str]
            Feature columns that ignore when fitting model.
        num_class: int
            The Number of target class.
        categorical_columns: List[str], optional
            Categorical feature columns. The default is None.
        train_params: Dict[str, any], optional
            LGB parameter for fitting model. The default is None.
        metric: str, optional
            Validation metric function. You can choice from ['log_loss', 'auc'].
            The default is 'auc'.
        loss: str, optional
            Loss function for fit. You can choice from ['log_loss', 'balanced'].
            The default is 'log_loss'.
        early_stopping_rounds: int, optional
            LGB early_stopping_rounds parameter.
        experiment_name: str, optional
            used for artifact saving directory name.
        show_train_learning_curve: bool, optional
            If True, plot learning curve of training data. The default is True.
        show_valid_learning_curve: bool, optional
            If True, plot learning curve of validation data. The default is True.
        show_test_learning_curve: bool, optional
            If True, plot learning curve of test data. The default is False.
        use_gpu: bool. optional
            If True, use GPU for training. The default is True.
        """

        self.num_class = num_class
        super().__init__(
            header_columns=header_columns,
            ignore_columns=ignore_columns,
            categorical_columns=categorical_columns,
            train_params=train_params,
            metric=metric,
            loss=loss,
            early_stopping_rounds=early_stopping_rounds,
            experiment_name=experiment_name,
            show_train_learning_curve=show_train_learning_curve,
            show_valid_learning_curve=show_valid_learning_curve,
            show_test_learning_curve=show_test_learning_curve,
            use_gpu=use_gpu,
        )
        if self.num_class > 2:
            self.train_params["num_class"] = self.num_class

    def make_datasets(self, df, reference=None):
        if self.use_weights and reference is None:
            self.weights = self._calc_weights(df)
        return super().make_datasets(df, reference=reference)

    def _calc_weights(self, train):
        counts = train[self.target_col].value_counts()
        total = counts.sum()
        weights_parser = {}
        for i in counts.index:
            weight = (self.num_class / 2) * (1 - counts[i] / total)
            weights_parser[i] = weight

        return np.vectorize(lambda y: weights_parser[y])(train[self.target_col])

    def _log_loss_metric(self, pred: np.ndarray, ds: lgb.Dataset):
        y = ds.get_label()
        try:
            log_loss = sk_metrics.log_loss(y, pred)
        # yのclass数が足りない場合
        except ValueError:
            log_loss = np.nan
        # eval_name, eval_result, is_higher_better
        return "log_loss", log_loss, False

    # TODO: 多クラス分類対応
    def _auc_metric(self, pred: np.ndarray, ds: lgb.Dataset):
        fpr, tpr, thresholds = sk_metrics.roc_curve(ds.get_label(), pred, pos_label=1)
        # eval_name, eval_result, is_higher_better
        return "auc", sk_metrics.auc(fpr, tpr), True

    def _init_metric(self, metric):
        if metric == "log_loss":
            self.metric = self._log_loss_metric
            self.metric_type = "log_loss"
        elif metric == "auc":
            self.metric = self._auc_metric
            self.metric_type = "auc"
        elif metric == "f1_macro":
            # 学習中のmetricはaucにし、評価時にf1とする
            self.metric = self._auc_metric
            self.metric_type = "f1_macro"
        else:
            if isinstance(metric, Callable):
                self.metric = metric
                print("Information: Custom metric set. Did you implement MLBase._calc_metric function?")
            else:
                raise ValueError(f'Invalid metric "{metric}".')

    def _init_loss(self, loss):
        self.use_weights = False

        if not loss or loss == "log_loss":
            self.loss = None
            if self.num_class == 2:
                self.core_params["objective"] = "binary"
            else:
                self.core_params["objective"] = "multiclass"
        elif loss == "balanced":
            self.loss = None
            self.use_weights = True
            if self.num_class == 2:
                self.core_params["objective"] = "binary"
            else:
                self.core_params["objective"] = "multiclass"
        else:
            if isinstance(loss, Callable):
                self.loss = loss
            else:
                raise ValueError("Invalid loss function.")


class LgbProgressBarCallback(object):
    def __init__(self, description: Optional[str] = None):
        self.description = description
        self.pbar = tqdm(leave=True, position=0)

    def __call__(self, env: lgb.callback.CallbackEnv):
        # 初回だけProgressBarを初期化する
        is_first_iteration: bool = env.iteration == env.begin_iteration

        if is_first_iteration:
            total: int = env.end_iteration - env.begin_iteration
            self.pbar.reset(total=total)
            self.pbar.set_description(self.description, refresh=False)

        # valid_setsの評価結果を更新
        if len(env.evaluation_result_list) > 0:
            # OrderedDictにしないと表示順がバラバラになって若干見にくい
            postfix = OrderedDict(
                [(f"{entry[0]}:{entry[1]}", f"{entry[2]:.4f}") for entry in env.evaluation_result_list]
            )
            self.pbar.set_postfix(ordered_dict=postfix, refresh=False)

        # 進捗を1進める
        self.pbar.update(1)
        self.pbar.refresh()


class BasicCBRegressor(MLBase):
    model_base_name = "cb"
    model_file_extention = "cbm"
    regression = True
    metric_type = "corr_coef"

    def __init__(
        self,
        header_columns,
        ignore_columns,
        categorical_columns=[],
        train_params={},
        metric="corr_coef",
        loss="rmse",
        corrcoef_reg_alpha=1e-3,
        early_stopping_rounds=None,
        experiment_name=None,
        show_train_learning_curve=True,
        show_valid_learning_curve=True,
        show_test_learning_curve=False,
    ):
        """
        Parameters
        ----------
        header_columns: List[str]
            Feature columns used for prediction header.
        ignore_columns: List[str]
            Feature columns that ignore when fitting model.
        categorical_columns: List[str], optional
            Categorical feature columns. The default is None.
        train_params: Dict[str, any], optional
            LGB parameter for fitting model. The default is None.
        metric: str, optional
            Validation metric function. You can choice from ['rmse', 'corr_coef']. The default is 'corr_coef'.
        loss: str, optional
            Loss function for fit. You can choice from ['rmse', 'mae', 'corr_coef']. The default is 'rmse'.
        corrcoef_reg_alpha: float, optional
            Regularization coefficient used for corrcoef loss. The default is 1e-3.
        early_stopping_rounds: int, optional
            LGB early_stopping_rounds parameter.
        experiment_name: str, optional
            used for artifact saving directory name.
        show_train_learning_curve: bool, optional
            If True, plot learning curve of training data. The default is True.
        show_valid_learning_curve: bool, optional
            If True, plot learning curve of validation data. The default is True.
        show_test_learning_curve: bool, optional
            If True, plot learning curve of test data. The default is False.
        """

        super().__init__(header_columns=header_columns, ignore_columns=ignore_columns)
        self.categorical_columns = categorical_columns
        self.corrcoef_reg_alpha = corrcoef_reg_alpha
        self.early_stopping_rounds = early_stopping_rounds
        self.show_train_learning_curve = show_train_learning_curve
        self.show_valid_learning_curve = show_valid_learning_curve
        self.show_test_learning_curve = show_test_learning_curve

        self._set_train_params(params=train_params)
        self._init_artifact_path(experiment_name=experiment_name)
        self._init_metric(metric)
        self._init_loss(loss)

    def fit(self, train, valid=None, test=None, save_model=False):
        train = self.make_dataset(train)
        if valid is not None:
            valid_pool = self.make_dataset(valid.copy())
        if test is not None:
            raise NotImplementedError

        model = self._get_model()
        model.fit(
            train,
            eval_set=valid_pool,
            use_best_model=True,
            # plot=True,
        )

        # TODO: plot_learning_curve

        if save_model:
            path = self._get_new_file_path(
                dir_=self.model_dir,
                base_name=self.model_base_name,
                extention=self.model_file_extention,
            )
            self._save_model(model, path)

        return model

    def _predict(self, model, features):
        df = self._preprocess_data(features, train=False)
        return model.predict(df)

    def _save_model(self, model, path):
        model.save_model(path)

    def _load_model(self, path):
        if "early_stopping_rounds" in self.train_params.keys():
            _ = self.train_params.pop("early_stopping_rounds")
        model = self._get_model()
        model.load_model(path)
        return model

    def make_dataset(self, df, train=True):
        df = self._preprocess_data(df, train=train)
        y = df[self.target_col].copy()
        df.drop(columns=self.target_col, inplace=True)
        return cb.Pool(df, y, cat_features=self.categorical_columns)

    def _get_model(self):
        return cb.CatBoostRegressor(**self.train_params)

    def _init_metric(self, metric):
        """
        TODO: 未実装
        self.metricは学習中にインスタンスが使用する関数。
        MLBase._calc_metricはevaluate関数内で使われる可視化を含む評価関数。
        """
        if metric == "corr_coef":
            self.metric = None
            self.metric_type = "corr_coef"
            raise NotImplementedError
        elif metric == "rmse":
            self.train_params["eval_metric"] = "RMSE"
            self.metric = None
            self.metric_type = "rmse"
        elif metric == "mae":
            self.train_params["eval_metric"] = "MAE"
            self.metric = None
            self.metric_type = "mae"
        else:
            if isinstance(metric, Callable):
                self.metric = metric
                print("Information: Custom metric set. Did you implement MLBase._calc_metric function?")
            else:
                raise ValueError(f'Invalid metric "{metric}".')

    def _init_loss(self, loss):
        if loss == "rmse":
            self.train_params["loss_function"] = "RMSE"
            self.loss = None
        elif loss == "mae":
            self.train_params["loss_function"] = "MAE"
            self.loss = None
        elif loss == "corr_coef":
            self.loss = None
            raise NotImplementedError
        else:
            if isinstance(loss, Callable):
                self.loss = loss
            else:
                raise ValueError("Invalid loss function.")

    def _set_train_params(self, params):
        self.train_params = copy.copy(args.cb_default_train_params)
        if self.early_stopping_rounds:
            self.train_params["early_stopping_rounds"] = self.early_stopping_rounds
        if self.categorical_columns:
            self.train_params["cat_features"] = self.categorical_columns
        if params:
            self.train_params = {**self.train_params, **params}

    def _preprocess_data(self, df, train=True):
        df = super()._preprocess_data(df, train)
        if self.categorical_columns:
            df[self.categorical_columns] = df[self.categorical_columns].fillna(-1).astype("int32")
        return df


class BasicCBClassifier(BasicCBRegressor):
    regression = False
    metric_type = "log_loss"

    def __init__(
        self,
        header_columns,
        ignore_columns,
        num_class,
        categorical_columns=[],
        train_params={},
        metric="auc",
        loss="log_loss",
        early_stopping_rounds=None,
        experiment_name=None,
        show_train_learning_curve=True,
        show_valid_learning_curve=True,
        show_test_learning_curve=False,
    ):
        """
        Parameters
        ----------
        header_columns: List[str]
            Feature columns used for prediction header.
        ignore_columns: List[str]
            Feature columns that ignore when fitting model.
        categorical_columns: List[str], optional
            Categorical feature columns. The default is None.
        train_params: Dict[str, any], optional
            LGB parameter for fitting model. The default is None.
        metric: str, optional
            Validation metric function. You can choice from ['rmse', 'corr_coef']. The default is 'corr_coef'.
        loss: str, optional
            Loss function for fit. You can choice from ['rmse', 'mae', 'corr_coef']. The default is 'rmse'.
        corrcoef_reg_alpha: float, optional
            Regularization coefficient used for corrcoef loss. The default is 1e-3.
        early_stopping_rounds: int, optional
            LGB early_stopping_rounds parameter.
        experiment_name: str, optional
            used for artifact saving directory name.
        show_train_learning_curve: bool, optional
            If True, plot learning curve of training data. The default is True.
        show_valid_learning_curve: bool, optional
            If True, plot learning curve of validation data. The default is True.
        show_test_learning_curve: bool, optional
            If True, plot learning curve of test data. The default is False.
        """

        if num_class > 2:
            # TODO
            raise NotImplementedError
        self.num_class = num_class
        super().__init__(
            header_columns=header_columns,
            ignore_columns=ignore_columns,
            categorical_columns=categorical_columns,
            train_params=train_params,
            metric=metric,
            loss=loss,
            early_stopping_rounds=early_stopping_rounds,
            experiment_name=experiment_name,
            show_train_learning_curve=show_train_learning_curve,
            show_valid_learning_curve=show_valid_learning_curve,
            show_test_learning_curve=show_test_learning_curve,
        )

    def _get_model(self):
        return cb.CatBoostClassifier(**self.train_params)

    def _predict(self, model, features):
        df = self._preprocess_data(features, train=False)
        return model.predict_proba(df)[:, 1]

    def _init_metric(self, metric):
        if metric == "auc":
            self.train_params["eval_metric"] = "AUC"
            self.metric_type = "auc"
        elif metric == "log_loss":
            self.train_params["eval_metric"] = "Logloss"
            self.metric_type = "log_loss"
        else:
            raise NotImplementedError

    def _init_loss(self, loss):
        if loss == "log_loss":
            self.train_params["loss_function"] = "Logloss"
        else:
            raise NotImplementedError


class BasicMLPRegressor(MLBase):
    model_base_name = "mlp"
    model_file_extention = "pth"
    params_base_name = "params"
    params_file_extention = "bin"
    regression = True
    metric_type = "corr_coef"

    def __init__(
        self,
        header_columns,
        ignore_columns,
        categorical_columns=[],
        cat_emb_dims=[],
        metric="corr_coef",
        loss="rmse",
        optimizer="radam",
        lr_scheduler="CosineAnnealingWarmRestarts",
        loss_kwargs={},
        optimizer_kwargs={},
        lr_scheduler_kwargs={},
        learning_rate=1e-3,
        epochs=30,
        batch_size=128,
        num_layers=4,
        early_stopping_rounds=None,
        checkpoint_metric="metric",
        dropout_rate=0.3,
        reg_alpha=1e-3,
        reg_lambda=1e-3,
        reg_gamma=None,
        corrcoef_reg_alpha=1e-3,
        weight_clip_value=1.0,
        use_best_epoch=True,
        experiment_name=None,
        show_train_learning_curve=True,
        show_valid_learning_curve=True,
        show_test_learning_curve=False,
        approximate_train_metrics=True,
    ):
        """
        Parameters
        ----------
        header_columns: List[str], optional
            Feature columns used for prediction header.
        ignore_columns: List[str], optional
            Feature columns that ignore when fitting model.
        categorical_columns: List[str], optional
            Categorical feature columns. The default is [].
        cat_emb_dims: List[int], optional
            Categorical feature's embedding dimensions. The default is [].
        metric: Union[str, Callable], optional
            Validation metric functions. You can choice ['rmse', 'corr_coef'].
            The default is 'corr_coef'.
        loss: Union[str, Callable], optional
            Loss function for fit. You can choice ['rmse', 'mae', 'corr_coef'].
            The default is 'rmse'.
        optimizer: Union[str, torch.Optimizer], optional
            Optimizer for fit. You can choice ['adam', 'radam', 'sgd'].
            The default is 'radam'.
        lr_scheduler: Union[str, torch.optim.lr_scheduler], optional
            Learning rate scheduler for fit. You can choice from ['CosineAnnealingWarmRestarts'].
            The default is 'CosineAnnealingWarmRestarts'.
        loss_kwargs: dict, optional
            Keyword arguments used for loss. The default is {}.
        optimizer_kwargs: dict, optional
            Keyword arguments used for optimizer. The default is {}.
        scheduler_kwargs: dict, optional
            Keyword arguments used for lr_scheduler. The default is {}.
        learning_rate: float, optional
            The leaning rate of optimizer. The default is 1e-3.
        epochs: int, optional
            Training epochs. The default is 5.
        batch_size: int, optional
            Training batch size. The default is 128.
        num_klayers: int, optional
            The number of layers. The default is 4.
        early_stopping_rounds: int, optional
            LGB early_stopping_rounds parameter. The default is None. This function is not yet implemented.
        checkpoint_metric: str, optional
            The metric used for choose best model. You can choose from ['loss', 'metric']. The default is 'metric'.
        dropout_rate: float, optional
            dropout_rate rate. The value should be (0, 1). The default is 0.3.
        reg_alpha: float, optional
            The L1 regularization coefficient. The default is 1e-3.
        reg_lambda: float, optional
            The L2 regularization coefficient. The default is 1e-3.
        reg_gamma: float, optional
            The standard deviation regularization coefficient.
            This may be useful for problems with very low signal-to-noise ratios.
            The default is None.
        corrcoef_reg_alpha: float, optional
            The regularization coefficient used for corrcoef loss. The default is 1e-3.
        weight_clip_value: float, optional
            The clip value of model paramater. The default is 1.0.
        use_best_epoch: bool, optional
            If True, the most high scored epoch's model used. The default is True.
        experiment_name: str, optional
            used for artifact saving directory name.
        show_train_learning_curve: bool, optional
            If True, plot learning curve of training data. The default is True.
        show_valid_learning_curve: bool, optional
            If True, plot learning curve of validation data. The default is True.
        show_test_learning_curve: bool, optional
            If True, plot learning curve of test data. The default is False.
        approximate_train_metrics: bool, optional
            If True, approximate train metric and loss using batch mean. The default is True.
        """

        super().__init__(header_columns=header_columns, ignore_columns=ignore_columns)
        self.categorical_columns = categorical_columns
        self.cat_emb_dims = cat_emb_dims
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss_kwargs = loss_kwargs
        self.optimizer_kwargs = optimizer_kwargs
        self.lr_scheduler_kwargs = lr_scheduler_kwargs
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.early_stopping_rounds = early_stopping_rounds
        self.checkpoint_metric = checkpoint_metric
        self.dropout_rate = dropout_rate
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.reg_gamma = reg_gamma
        self.corrcoef_reg_alpha = corrcoef_reg_alpha
        self.weight_clip_value = weight_clip_value
        self.use_best_epoch = use_best_epoch
        self.show_train_learning_curve = show_train_learning_curve
        self.show_valid_learning_curve = show_valid_learning_curve
        self.show_test_learning_curve = show_test_learning_curve
        self.approximate_train_metrics = approximate_train_metrics

        self._init_artifact_path(experiment_name=experiment_name)
        self._init_metric(metric)
        self._init_loss(loss)

        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    def fit(self, train, valid=None, test=None, save_model=False):
        train_set, train_loader = self._get_ds_and_loader(train, shuffle=True)
        steps_per_epoch = train_set.steps_per_epoch
        if valid is not None:
            _, valid_loader = self._get_ds_and_loader(valid, shuffle=False)
        if test is not None:
            _, test_loader = self._get_ds_and_loader(test, shuffle=False)

        model = self._get_model(train_set=train_set)
        model.to(self._get_device())
        optimizer = self._init_optimizer(model=model)
        lr_scheduler = self._init_lr_cheduler(optimizer)

        self._reset_metric_container(valid=valid is not None, test=test is not None)
        self._reset_checkpoint()

        for epoch in range(self.epochs):
            self._on_epoch(
                model=model,
                train_loader=train_loader,
                valid_loader=None if valid is None else valid_loader,
                test_loader=None if test is None else test_loader,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                epoch=epoch,
                steps_per_epoch=steps_per_epoch,
            )

        if self.metric_record:
            self.plot_learning_curve(self.metric_record)

        if self.use_best_epoch:
            model = self.best_model
            self._reset_checkpoint(model_only=True)
            best_metric = self.best_metric * (-1 if self.checkpoint_metric == "loss" else 1)
            print(f"Best epoch: {self.best_epoch}")
            print(f"Best metric: {best_metric:.3f}")

        if save_model:
            model_path = self._get_new_file_path(
                self.model_dir,
                base_name=self.model_base_name,
                extention=self.model_file_extention,
            )
            params_path = self._get_new_file_path(
                self.model_dir,
                base_name=self.params_base_name,
                extention=self.params_file_extention,
            )
            self._save_model(model, path=model_path)
            self._save_params(model, path=params_path)

        model.to("cpu")
        self._clear_gpu_cache()
        return model

    def _get_ds_and_loader(self, df, shuffle):
        data_set = self._get_data_set(
            self._preprocess_data(df, target_col=self.target_col),
            target_col=self.target_col,
            shuffle=shuffle,
        )
        data_loader = data_set.get_data_loader()
        return data_set, data_loader

    def _on_epoch(
        self,
        model,
        train_loader,
        valid_loader,
        test_loader,
        optimizer,
        lr_scheduler,
        epoch,
        steps_per_epoch,
    ):
        cum_loss = 0.0
        cum_metric = 0.0
        batch_count = 0
        pbar = self._reset_pbar(epoch=epoch, steps_per_epoch=steps_per_epoch)

        model.train()
        for *features, labels in train_loader:
            loss, metric = self._train_batch(*features, labels=labels, model=model, optimizer=optimizer)

            cum_loss += loss
            cum_metric += metric
            batch_count += 1
            self._update_pbar(
                pbar=pbar,
                cum_loss=cum_loss,
                cum_metric=cum_metric,
                batch_count=batch_count,
            )

        self._eval_epoch(
            model=model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            test_loader=test_loader,
            cum_loss=cum_loss,
            cum_metric=cum_metric,
            epoch=epoch,
            steps_per_epoch=steps_per_epoch,
        )
        self._update_lr_scheduler(lr_scheduler=lr_scheduler)

    def _eval_epoch(
        self,
        model,
        train_loader,
        valid_loader,
        test_loader,
        cum_loss,
        cum_metric,
        epoch,
        steps_per_epoch,
    ):
        # self.predictの引数はpd.DataFrameなので直接eval
        model.eval()

        if self.approximate_train_metrics:
            train_loss = cum_loss / steps_per_epoch
            train_metric = cum_metric / steps_per_epoch
        else:
            train_loss, train_metric = self._eval_valid_set(train_loader, model=model)

        self._log_metrics(
            epoch=epoch,
            loss=train_loss,
            metric=train_metric,
            header="train",
            print_=False,
            append_record=self.show_train_learning_curve,
        )

        if valid_loader is not None:
            valid_loss, valid_metric = self._eval_valid_set(valid_loader, model=model)
            if self.use_best_epoch:
                self._checkpoint(loss=valid_loss, metric=valid_metric, epoch=epoch, model=model)
            self._log_metrics(
                epoch=epoch,
                loss=valid_loss,
                metric=valid_metric,
                header="valid",
                print_=True,
                append_record=self.show_valid_learning_curve,
            )

        if test_loader is not None:
            test_loss, test_metric = self._eval_valid_set(test_loader, model=model)
            self._log_metrics(
                epoch=epoch,
                loss=test_loss,
                metric=test_metric,
                header="test",
                print_=True,
                append_record=self.show_test_learning_curve,
            )

    def _eval_valid_set(self, data_loader, model):
        preds = []
        labels = []
        with torch.no_grad():
            for *batch_features, batch_labels in data_loader:
                preds.append(model(*batch_features))
                labels.append(batch_labels)

        preds = torch.cat(preds)
        labels = torch.cat(labels)
        loss = self.loss(preds, labels).item()
        metric = self.metric(preds, labels)
        return loss, metric

    def _update_lr_scheduler(self, lr_scheduler):
        if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            try:
                valid_loss = self.metric_record["valid"]["loss"]
            except KeyError:
                raise KeyError("Validation data is not given.")
            lr_scheduler.step(valid_loss)
        else:
            lr_scheduler.step()

    def _train_batch(self, *features, labels, model, optimizer):
        optimizer.zero_grad()
        preds = model(*features)
        loss = self.loss(preds, labels)
        metric = self.metric(preds, labels)
        if self.reg_alpha:
            loss = self._regularization(model, loss, l2=False)
        if self.reg_lambda:
            loss = self._regularization(model, loss, l2=True)
        if self.reg_gamma:
            loss = self._mean_convergence_regularization(preds, loss)
        if math.isnan(metric):
            print("metric is nan.")
            metric = 0.0
        loss.backward()
        if self.weight_clip_value:
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.weight_clip_value)
        optimizer.step()

        return loss.item(), metric

    def _reset_pbar(self, epoch, steps_per_epoch):
        pbar = tqdm(leave=True, position=0, total=steps_per_epoch)
        pbar.set_description(f"epoch {epoch}", refresh=False)
        return pbar

    def _update_pbar(self, pbar, cum_loss, cum_metric, batch_count):
        postfix = OrderedDict(
            [
                ("train_loss", f"{(cum_loss / batch_count):.3f}"),
                (
                    f"train_{self._get_metric_name()}",
                    f"{(cum_metric / batch_count):.3f}",
                ),
            ]
        )
        pbar.set_postfix(ordered_dict=postfix, refresh=False)
        pbar.update(1)
        pbar.refresh()

    def _reset_metric_container(self, valid=True, test=True):
        metric_name = self._get_metric_name()
        metric_record = {}
        if self.show_train_learning_curve:
            metric_record["train"] = {"loss": [], metric_name: []}
        if self.show_valid_learning_curve and valid:
            metric_record["valid"] = {"loss": [], metric_name: []}
        if self.show_test_learning_curve and test:
            metric_record["test"] = {"loss": [], metric_name: []}

        self.metric_record = metric_record

    def _get_metric_name(self):
        name = self.metric.__name__
        if name[0] == "_":
            name = name[1:]
        return name

    def _log_metrics(self, epoch, loss, metric, header, print_=True, append_record=True):
        metric_name = self._get_metric_name()
        if print_:
            print(f"\nepoch {epoch}   {header}_loss: {loss:.4f}   {header}_{metric_name}: {metric:.4f}")
        if append_record:
            self.metric_record[header]["loss"].append(loss)
            self.metric_record[header][metric_name].append(metric)

    def _reset_checkpoint(self, model_only=False):
        self.best_model = None
        if not model_only:
            self.best_metric = -1e18
            self.best_epoch = 0

    def _checkpoint(self, loss, metric, epoch, model):
        if self.checkpoint_metric == "loss":
            objective = -loss
        elif self.checkpoint_metric == "metric":
            objective = metric

        if objective > self.best_metric:
            self.best_metric = objective
            self.best_model = copy.deepcopy(model)
            self.best_epoch = epoch

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
            # Note: * アンパックはfor の後に直接使えない
            for (*features,) in data_loader:
                pred = model(*features)
                if not self.regression:
                    pred = torch.sigmoid(pred)
                preds.append(pred)
        preds = torch.cat(preds)
        preds = preds.detach().to("cpu").numpy()
        model.to("cpu")
        self._clear_gpu_cache()
        return preds

    def save_model(self):
        for i in self.model:
            model_path = self._get_new_file_path(
                self.model_dir,
                base_name=self.model_base_name,
                extention=self.model_file_extention,
            )
            params_path = self._get_new_file_path(
                self.model_dir,
                base_name=self.params_base_name,
                extention=self.params_file_extention,
            )

            self._save_model(model=i, path=model_path)
            self._save_params(model=i, path=params_path)

    def _save_model(self, model, path):
        torch.save(model.state_dict(), path)

    def _save_params(self, model, path):
        kwargs = {}
        params = inspect.signature(model.__init__).parameters
        for i in params:
            kwargs[i] = getattr(model, i)
        params = MLPParamaters(**kwargs)
        params.save(path)

    def load_model(self):
        model_files = self._get_file_list(self.model_file_extention)
        params_files = self._get_file_list(self.params_file_extention)
        assert len(model_files) == len(params_files)
        model = []
        for model_path, params_path in zip(model_files, params_files):
            model.append(self._load_model(model_path, params_path))
        self.model = model

    def _load_model(self, model_path, params_path):
        model = self.get_empty_model(params_path)
        model.load_state_dict(torch.load(model_path, map_location=self._get_device()))
        return model

    def _load_params(cls, params_path):
        return MLPParamaters.load(params_path)

    def get_empty_model(self, params_path):
        params = self._load_params(params_path)
        return BasicMLPModule(**params.__dict__)

    def _get_file_list(self, extention):
        files = glob.glob(os.path.join(self.model_dir, f"*.{extention}"))
        files.sort()
        assert len(files) >= 1
        return files

    def _init_metric(self, metric):
        if metric == "rmse":
            self.metric = self._rmse_metric
            self.metric_type = "rmse"
        elif metric == "mae":
            self.metric = self._mae_metric
            self.metric_type = "mae"
        elif metric == "corr_coef":
            self.metric = self._corr_coef_metric
            self.metric_type = "corr_coef"
        else:
            if isinstance(metric, Callable):
                self.metric = metric
            else:
                raise ValueError(f'Invalid metric "{metric}".')

    def _init_loss(self, loss):
        if loss == "rmse":
            self.loss = self._rmse
        elif loss == "mae":
            self.loss = self._mae
        elif loss == "corr_coef":
            self.loss = self._corr_coef_loss
        else:
            if isinstance(loss, Callable):
                self.loss = loss
            else:
                raise ValueError(f'Invalid loss function "{loss}".')

    def _init_optimizer(self, model):
        params = model.parameters()
        if self.optimizer == "adam":
            return torch.optim.Adam(params, lr=self.learning_rate, **self.optimizer_kwargs)
        elif self.optimizer == "radam":
            return torch.optim.RAdam(params, lr=self.learning_rate, **self.optimizer_kwargs)
        elif self.optimizer == "sgd":
            return torch.optim.SGD(params, lr=self.learning_rate, **self.optimizer_kwargs)
        else:
            raise ValueError(f'Invalid optimizer "{self.optimizer}".')

    def _init_lr_cheduler(self, optimizer):
        scheduler = torch.optim.lr_scheduler
        if self.lr_scheduler == "CosineAnnealingWarmRestarts":
            if not self.lr_scheduler_kwargs.get("T_0"):
                if self.epochs < 10:
                    t_0 = max(self.epochs // 2, 1)
                else:
                    t_0 = 10
                self.lr_scheduler_kwargs["T_0"] = t_0
            return scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, **self.lr_scheduler_kwargs)
        elif self.lr_scheduler == "ReduceLROnPlateau":
            return scheduler.ReduceLROnPlateau(optimizer=optimizer, **self.lr_scheduler_kwargs)
        else:
            raise ValueError(f'Invalid lr_scheduler "{self.lr_scheduler}".')

    def _rmse(self, pred, label, eps=1e-6):
        criterion = torch.nn.MSELoss()
        return torch.sqrt(criterion(pred, label) + eps)

    def _mae(self, pred, label):
        criterion = torch.nn.L1Loss()
        return criterion(pred, label)

    def _corr_coef_loss(self, pred, label):
        x = torch.cat([pred, label], axis=1)
        x = torch.t(x)
        corr_coef = torch.corrcoef(x)[0][1]
        mean_diff = torch.abs(torch.mean(pred) - torch.mean(label))
        std_diff = torch.abs(torch.std(pred) - torch.std(label))
        return -corr_coef + self.corrcoef_reg_alpha * (mean_diff + std_diff)

    def _rmse_metric(self, pred, label):
        return self._rmse(pred, label).item()

    def _mae_metric(self, pred, label):
        return self._mae(pred, label).item()

    def _corr_coef_metric(self, pred, label):
        x = torch.cat([pred, label], axis=1)
        x = torch.t(x)
        return torch.corrcoef(x)[0][1].item()

    def _regularization(self, model, loss, l2=True):
        # Note: do not use in-place operation to require_grad tensor.
        reg_loss = torch.tensor(0.0, requires_grad=True)
        count = 0.0
        for w in model.parameters():
            count += 1.0
            if l2:
                reg_loss = reg_loss + torch.norm(w) ** 2 / w.nelement()

            else:
                reg_loss = reg_loss + torch.norm(w, 1) / w.nelement()

        reg_loss = reg_loss / count
        loss = loss + (self.reg_lambda if l2 else self.reg_alpha) * reg_loss
        return loss

    def _mean_convergence_regularization(self, preds, loss):
        return loss - self.reg_gamma * torch.std(preds)

    def _get_model(self, train_set):
        return BasicMLPModule(
            feature_dim=train_set.feature_dim,
            cat_dims=train_set.cat_dims,
            cat_emb_dims=self.cat_emb_dims,
            dropout_rate=self.dropout_rate,
            num_layers=self.num_layers,
        )

    def _get_data_set(self, df, target_col, shuffle):
        return BasicMLPDataSet(
            df,
            target_col=target_col,
            batch_size=self.batch_size,
            shuffle=shuffle,
            categorical_columns=self.categorical_columns,
        )

    def _preprocess_data(self, df, target_col):
        if target_col:
            labels = df[target_col]
            features = df.drop(columns=target_col)
        else:
            features = df.copy()

        cat_cols = self.categorical_columns
        num_cols = self._get_numerical_columns(
            df,
            cat_cols=cat_cols,
            target_col=target_col,
            ignore_cols=self.ignore_columns,
        )
        features[num_cols] = features[num_cols].fillna(features[num_cols].mean()).fillna(-1).astype("float32")
        features[cat_cols] = features[cat_cols].fillna(-1).astype("int32")

        if target_col:
            features[target_col] = labels
            features = features.dropna(subset=[target_col])

        return features

    def _clear_gpu_cache(self):
        if torch.cuda.is_available():
            with torch.no_grad():
                torch.cuda.memory.empty_cache()

    @classmethod
    def _get_numerical_columns(cls, df, cat_cols, target_col, ignore_cols=[]):
        ignore_columns = cat_cols + ignore_cols + ([target_col] if target_col else [])
        condition = lambda col: col not in ignore_columns and str(df[col].dtype) != "object"
        return [i for i in df.columns if condition(i)]

    @classmethod
    def _get_device(self, debug=False):
        if debug:
            return torch.device("cpu")
        else:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BasicMLPClassifier(BasicMLPRegressor):
    regression = False
    metric_type = "log_loss"

    def __init__(
        self,
        header_columns,
        ignore_columns,
        num_class,
        categorical_columns=[],
        cat_emb_dims=[],
        metric="auc",
        loss="log_loss",
        optimizer="radam",
        lr_scheduler="CosineAnnealingWarmRestarts",
        loss_kwargs={},
        optimizer_kwargs={},
        lr_scheduler_kwargs={},
        learning_rate=1e-3,
        epochs=30,
        batch_size=128,
        num_layers=4,
        early_stopping_rounds=None,
        checkpoint_metric="metric",
        dropout_rate=0.3,
        reg_alpha=1e-3,
        reg_lambda=1e-3,
        reg_gamma=None,
        weight_clip_value=1.0,
        use_best_epoch=True,
        experiment_name=None,
        show_train_learning_curve=True,
        show_valid_learning_curve=True,
        show_test_learning_curve=False,
        approximate_train_metrics=True,
    ):
        """
        Parameters
        ----------
        header_columns: List[str], optional
            Feature columns used for prediction header.
        ignore_columns: List[str], optional
            Feature columns that ignore when fitting model.
        num_class: int
            The Number of target class.
        categorical_columns: List[str], optional
            Categorical feature columns. The default is [].
        cat_emb_dims: List[int], optional
            Categorical feature's embedding dimensions. The default is [].
        metric: Union[str, Callable], optional
            Validation metric functions. You can choice ['auc'].
            The default is 'auc'.
        loss: Union[str, Callable], optional
            Loss function for fit. You can choice ['log_loss', 'focal_loss'].
            The default is 'log_loss'.
        optimizer: Union[str, torch.Optimizer], optional
            Optimizer for fit. You can choice ['adam', 'radam', 'sgd'].
            The default is 'radam'.
        lr_scheduler: Union[str, torch.optim.lr_scheduler], optional
            Learning rate scheduler for fit. You can choice from ['CosineAnnealingWarmRestarts'].
            The default is 'CosineAnnealingWarmRestarts'.
        loss_kwargs: dict, optional
            Keyword arguments used for loss. The default is {}.
        optimizer_kwargs: dict, optional
            Keyword arguments used for optimizer. The default is {}.
        scheduler_kwargs: dict, optional
            Keyword arguments used for lr_scheduler. The default is {}.
        learning_rate: float, optional
            The leaning rate of optimizer. The default is 1e-3.
        epochs: int, optional
            Training epochs. The default is 30.
        batch_size: int, optional
            Training batch size. The default is 128.
        num_klayers: int, optional
            The number of layers. The default is 4.
        early_stopping_rounds: int, optional
            LGB early_stopping_rounds parameter. The default is None. This function is not yet implemented.
        checkpoint_metric: str, optional
            The metric used for choose best model. You can choose from ['loss', 'metric']. The default is 'metric'.
        dropout_rate: float, optional
            dropout_rate rate. The value should be (0, 1). The default is 0.3.
        reg_alpha: float, optional
            The L1 regularization coefficient. The default is 1e-3.
        reg_lambda: float, optional
            The L2 regularization coefficient. The default is 1e-3.
        reg_gamma: float, optional
            The standard deviation regularization coefficient.
            This may be useful for problems with very low signal-to-noise ratios.
            The default is None.
        weight_clip_value: float, optional
            The clip value of model paramater. The default is 1.0.
        use_best_epoch: bool, optional
            If True, the most high scored epoch's model used. The default is True.
        experiment_name: str, optional
            used for artifact saving directory name.
        show_train_learning_curve: bool, optional
            If True, plot learning curve of training data. The default is True.
        show_valid_learning_curve: bool, optional
            If True, plot learning curve of validation data. The default is True.
        show_test_learning_curve: bool, optional
            If True, plot learning curve of test data. The default is False.
        approximate_train_metrics: bool, optional
            If True, approximate train metric and loss using batch mean. The default is True.
        """

        self.num_class = num_class
        super().__init__(
            header_columns=header_columns,
            ignore_columns=ignore_columns,
            categorical_columns=categorical_columns,
            cat_emb_dims=cat_emb_dims,
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
            num_layers=num_layers,
            early_stopping_rounds=early_stopping_rounds,
            checkpoint_metric=checkpoint_metric,
            dropout_rate=dropout_rate,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            reg_gamma=reg_gamma,
            weight_clip_value=weight_clip_value,
            use_best_epoch=use_best_epoch,
            experiment_name=experiment_name,
            show_train_learning_curve=show_train_learning_curve,
            show_valid_learning_curve=show_valid_learning_curve,
            show_test_learning_curve=show_test_learning_curve,
            approximate_train_metrics=approximate_train_metrics,
        )

    def _init_metric(self, metric):
        if metric == "auc":
            self.metric = self._auc
            self.metric_type = "auc"
        else:
            if isinstance(metric, Callable):
                self.metric = metric
            else:
                raise ValueError(f'Invalid metric "{metric}".')

    def _init_loss(self, loss):
        if loss == "log_loss":
            self.loss = self._log_loss
        elif loss == "focal_loss":
            self.loss = self._focal_loss
        else:
            if isinstance(loss, Callable):
                self.loss = loss
            else:
                raise ValueError(f'Invalid loss function "{loss}".')

    def _log_loss(self, pred, label, eps=1e-7):
        label_smoothing = self.loss_kwargs.get("label_smoothing")
        if label_smoothing:
            logit = torch.nn.Sigmoid()(pred)
            logit = logit.clamp(eps, 1.0 - eps)
            loss = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)(logit, label)
        else:
            loss = torch.nn.BCEWithLogitsLoss()(pred, label)
        return loss

    def _focal_loss(self, pred, label, eps=1e-7):
        gamma = self.loss_kwargs.get("gamma")
        if not gamma:
            gamma = 1.0
        logit = torch.nn.Sigmoid()(pred)
        logit = logit.clamp(eps, 1.0 - eps)
        loss = -((1 - logit) ** gamma) * label * torch.log(logit) - logit**gamma * (1 - label) * torch.log(1 - logit)
        return loss.mean()

    # TODO: 多クラス分類対応
    def _auc(self, pred, label):
        fpr, tpr, thresholds = sk_metrics.roc_curve(
            label.detach().cpu().numpy(), pred.detach().cpu().numpy(), pos_label=1
        )
        return sk_metrics.auc(fpr, tpr)


class BasicMLPDataSet(torch.utils.data.Dataset):
    def __init__(
        self,
        df,
        target_col,
        batch_size,
        shuffle,
        categorical_columns=[],
        ignore_columns=[],
    ):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self._set_feature_columns(
            df,
            target_col=target_col,
            cat_cols=categorical_columns,
            ignore_cols=ignore_columns,
        )
        self._set_feature_array(df, target_col=target_col)
        self._set_data_length(df)
        self._set_steps_per_epoch()
        self._set_cat_dims()

    def __len__(self):
        return self.data_length

    def __getitem__(self, idx):
        num_features = self._get_tensor(self.num_features[idx, ...], dtype=torch.float)
        cat_features = self._get_tensor(self.cat_features[idx, ...], dtype=torch.int)
        if self.targets is None:
            return num_features, cat_features
        else:
            targets = self._get_tensor(self.targets[idx, ...], dtype=torch.float)
            return num_features, cat_features, targets

    def get_data_loader(self):
        # Note: num_workers=2, persistent_workers=Trueを設定すると若干速度が上がるが、
        #       良くわからんエラーが出るようになる。以下のForumが多分原因を示している。
        #       https://discuss.pytorch.org/t/w-cudaipctypes-cpp-22-producer-process-has-been-terminated-before-all-shared-cuda-tensors-released-see-note-sharing-cuda-tensors/124445
        if hasattr(self, "_collate_fn"):
            collate_fn = self._collate_fn
        else:
            collate_fn = None

        return torch.utils.data.DataLoader(
            self,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=0,
            collate_fn=collate_fn,
            # pin_memory=True,
            # persistent_workers=True,
        )

    def _set_feature_columns(self, df, target_col, cat_cols, ignore_cols):
        self.cat_cols = cat_cols
        self.num_cols = BasicMLPRegressor._get_numerical_columns(
            df, cat_cols=cat_cols, target_col=target_col, ignore_cols=ignore_cols
        )
        self.target_col = target_col
        self.feature_dim = len(self.cat_cols) + len(self.num_cols)

    def _set_feature_array(self, df, target_col):
        self.num_features = df[self.num_cols].values
        self.cat_features = df[self.cat_cols].values
        self.targets = df[[target_col]].values if target_col else None

    def _set_cat_dims(self):
        self.cat_dims = []
        for i in range(len(self.cat_cols)):
            self.cat_dims.append(len(np.unique(self.cat_features[:, i])))

    def _set_data_length(self, df):
        self.data_length = len(df)

    def _set_steps_per_epoch(self):
        self.steps_per_epoch = int(self.data_length / self.batch_size) + 1

    def _get_tensor(self, array, dtype):
        tensor = torch.tensor(array, dtype=dtype)
        tensor = self._to_device(tensor)
        return tensor

    def _to_device(self, tensor):
        return tensor.to(BasicMLPRegressor._get_device(), non_blocking=True)


class BasicMLPModule(torch.nn.Module):
    def __init__(
        self,
        feature_dim,
        cat_dims,
        cat_emb_dims,
        dropout_rate,
        num_layers,
        output_dim=1,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.cat_dims = cat_dims
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers
        self.output_dim = output_dim

        self.cat_emb_dims = self._get_cat_emb_dims(cat_emb_dims, cat_dims)
        self.dropout = torch.nn.Dropout(self.dropout_rate)
        self.embeddings = torch.nn.ModuleList(
            [self._construct_cat_emb(cat_dim, emb_dim) for cat_dim, emb_dim in zip(self.cat_dims, self.cat_emb_dims)]
        )
        self.fc = self._construct_fc(input_dim=self._get_encoded_dim(), num_layers=self.num_layers)

    def _construct_cat_emb(self, cat_dim, emb_dim, padding_idx=None):
        return torch.nn.Embedding(num_embeddings=cat_dim, embedding_dim=emb_dim, padding_idx=padding_idx)

    def _construct_fc(self, input_dim, num_layers):
        fc = torch.nn.Sequential()
        dim = copy.deepcopy(input_dim)
        for i in range(num_layers - 1):
            out_dim = dim // 2
            if out_dim <= self.output_dim:
                break

            fc.add_module(f"fc_{i}", torch.nn.Linear(dim, out_dim))
            dim = out_dim

        fc.add_module(f"fc_{i + 1}", torch.nn.Linear(dim, self.output_dim))
        return fc

    def forward(self, num_features, cat_features):
        x = self._feature_encoder(num_features, cat_features)
        x = self.fc(x)
        return x

    def _feature_encoder(self, num_features, cat_features):
        # batch_size * feature_dim
        embs = []
        for i in range(len(self.cat_dims)):
            embs.append(self.embeddings[i](cat_features[:, i]))
        x = torch.cat([num_features] + embs, dim=1)
        x = self.dropout(x)
        return x

    def _get_encoded_dim(self):
        return self.feature_dim + sum(self.cat_emb_dims) - len(self.cat_emb_dims)

    def _get_cat_emb_dims(self, cat_emb_dims, cat_dims):
        if cat_emb_dims:
            return cat_emb_dims
        else:
            return [max(2, int(np.log2(i))) for i in cat_dims]


class MLPParamaters:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def save(self, path):
        with open(path, "wb") as p:
            pickle.dump(self, p)

    @staticmethod
    def load(path):
        with open(path, "rb") as p:
            obj = pickle.load(p)
        return obj


class BasicSKLearnRegressor(MLBase):
    model_base_name = "sklearn"
    model_file_extention = "pickle"
    regression = True
    metric_type = "corr_coef"

    def __init__(
        self,
        model_class,
        model_kwargs,
        header_columns,
        ignore_columns,
        loss="rmse",
        metric="corr_coef",
        experiment_name=None,
    ):
        """
        Parameters
        ----------
        header_columns: List[str]
            Feature columns used for prediction header.
        ignore_columns: List[str]
            Feature columns that ignore when fitting model.
        loss: str, optional
            Loss function for fit. The default is 'rmse'.
        metric: str, optional
            Validation metric function. The default is 'corr_coef'.
        experiment_name: str, optional
            used for artifact path.
        """

        super().__init__(header_columns=header_columns, ignore_columns=ignore_columns)
        self.model_class = model_class
        self.model_kwargs = model_kwargs
        self._init_artifact_path(experiment_name=experiment_name)
        self._init_loss(loss)
        self._init_metric(metric)

    def fit(self, train, valid=None, test=None, save_model=False):
        model = self.model_class(**self.model_kwargs)
        model.fit(
            self._preprocess_data(train.drop(columns=self.target_col)).values,
            train[self.target_col].values,
        )
        if valid is not None:
            self._calc_valid_metrics(model=model, valid=valid)
        if test is not None:
            self._calc_valid_metrics(model=model, valid=test)
        if save_model:
            path = self._get_new_file_path(
                dir_=self.model_dir,
                base_name=self.model_base_name,
                extention=self.model_file_extention,
            )
            self._save_model(model, path)

        return model

    def _predict(self, model, features):
        df = self._preprocess_data(features, train=False)
        if self.target_col in df.columns:
            df = df.drop(columns=[self.target_col])
        return model.predict(df.values)

    def _preprocess_data(self, df, train=True):
        return super()._preprocess_data(df, train=train)

    def _save_model(self, model, path):
        with open(path, "wb") as f:
            pickle.dump(model, f)

    def _load_model(self, path):
        with open(path, mode="rb") as f:
            return pickle.load(f)

    def _calc_valid_metrics(self, model, valid):
        pred = self._predict(
            model=model,
            features=self._preprocess_data(valid.drop(columns=self.target_col)),
        )
        label = valid[self.target_col].values
        loss = self.loss(pred=pred, label=label)
        metric = self.metric(pred=pred, label=label)
        print(f"loss: {loss}, metric: {metric}")
        return loss, metric

    def _rmse(self, pred: np.ndarray, label: np.ndarray):
        return math.sqrt(np.square(label - pred).sum() / len(label))

    def _mae(self, pred: np.ndarray, label: np.ndarray):
        return np.abs(label - pred).sum() / len(label)

    def _corr_coef(self, pred: np.ndarray, label: np.ndarray):
        return np.corrcoef(label, pred)[0][1]

    def _init_metric(self, metric):
        if metric == "corr_coef":
            self.metric = self._corr_coef
            self.metric_type = "corr_coef"
        elif metric == "rmse":
            self.metric = self._rmse
            self.metric_type = "rmse"
        elif metric == "mae":
            self.metric = self._mae
            self.metric_type = "mae"
        else:
            if isinstance(metric, Callable):
                self.metric = metric
                print("Information: Custom metric set. Did you implement MLBase._calc_metric function?")
            else:
                raise ValueError(f'Invalid metric "{metric}".')

    def _init_loss(self, loss):
        if loss == "rmse":
            self.loss = self._rmse
        elif loss == "mae":
            self.loss = self._mae
        else:
            if isinstance(loss, Callable):
                self.loss = loss
            else:
                raise ValueError("Invalid loss function.")


class BasicSKLearnClassifier(BasicSKLearnRegressor):
    regression = False

    def __init__(
        self,
        model_class,
        model_kwargs,
        header_columns,
        ignore_columns,
        loss="log_loss",
        metric="auc",
        model_dir=None,
    ):
        """
        Parameters
        ----------
        header_columns: List[str]
            Feature columns used for prediction header.
        ignore_columns: List[str]
            Feature columns that ignore when fitting model.
        loss: str, optional
            Loss function for fit. The default is 'log_loss'.
        metric: str, optional
            Validation metric function. The default is 'auc'.
        model_dir: str, optional
            Model saving or saved directory.
        """

        super().__init__(
            model_class=model_class,
            model_kwargs=model_kwargs,
            header_columns=header_columns,
            ignore_columns=ignore_columns,
            loss=loss,
            metric=metric,
            model_dir=model_dir,
        )

    # TODO: 多クラス分類
    def _predict(self, model, features):
        df = self._preprocess_data(features, train=False)
        if self.target_col in df.columns:
            df = df.drop(columns=[self.target_col])
        return model.predict_proba(df.values)[:, 1]

    def _log_loss(self, pred, label):
        try:
            log_loss = sk_metrics.log_loss(label, pred)
        # yのclass数が足りない場合
        except ValueError:
            log_loss = np.nan
        return log_loss

    # TODO: 多クラス分類対応
    def _auc_metric(self, pred, label):
        fpr, tpr, thresholds = sk_metrics.roc_curve(label, pred, pos_label=1)
        auc = sk_metrics.auc(fpr, tpr)
        return auc

    def _init_metric(self, metric):
        if metric == "log_loss":
            self.metric = self._log_loss
        elif metric == "auc":
            self.metric = self._auc_metric
        else:
            if isinstance(metric, Callable):
                self.metric = metric
            else:
                raise ValueError(f'Invalid metric "{metric}".')

    def _init_loss(self, loss):
        if loss == "log_loss":
            self.loss = self._log_loss
        else:
            if isinstance(loss, Callable):
                self.loss = loss
            else:
                raise ValueError("Invalid loss function.")


if __name__ == "__main__":
    import random

    random.seed(42)
    np.random.seed(42)

    from mlutil.data import load_california_housing, train_test_split

    data = load_california_housing()
    data["cat"] = np.random.randint(0, 2, len(data))
    data["y"] = pd.cut(data["y"], 3, labels=False)
    data = data.sample(frac=1)
    data = data.dropna(how="any")
    data = train_test_split(data, test_size=0.2)
    # model = BasicMLPRegressor(header_columns=[],
    #                           ignore_columns=[],
    #                           categorical_columns=['cat'],
    #                           cat_emb_dims=[2],
    #                           learning_rate=1e-4,
    #                           epochs=2,
    #                           loss='rmse',
    #                           reg_lambda=1e-3,
    #                           model_dir='./model/test'
    #                             )
    # model = BasicLGBClassifier(header_columns=[],
    #                             ignore_columns=[],
    #                             num_class=3,
    #                             categorical_columns=['cat'],
    #                             metric='acc',
    #                             loss='balanced',
    #                             use_gpu=False,
    #                             model_dir='./model/test'
    #                             )
    model = BasicLGBRegressor(
        header_columns=[],
        ignore_columns=[],
        categorical_columns=["cat"],
        use_gpu=False,
        model_dir="./model/test",
    )

    # model.cv(k_fold(data.train, 2), save_model=True)
    # model.fit(data.train, data.test, save_model=True)
    # result = model.evaluate(data.test)

    model.load_model()
    result = model.evaluate(data.test)
