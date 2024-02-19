import copy
import os

import optuna


class Tuner:
    def __init__(
        self,
        submitter,
        model_class,
        n_trials=100,
        optimize_direction="minimize",
        dry_run=False,
        known_params_list=[],
        known_values_list=[],
        initial_trial_params=[],
        params_distribution={},
        fix_params=[],
        seed=408,
    ):
        self.submitter = submitter
        self.model_class = model_class
        self.n_trials = n_trials if not dry_run else 2
        self.optimize_direction = optimize_direction
        self.dry_run = dry_run
        self.known_params_list = known_params_list
        self.known_values_list = known_values_list
        self.initial_trial_params = initial_trial_params
        self.params_distribution = params_distribution
        self.fix_params = fix_params
        self.submitter.seed_everything(seed)

    def run(self):
        best_params = self._study()
        return best_params

    def sample_params(self, trial) -> dict:
        """
        n_iter, lr等固定値を含めてパラメータを返す
        """
        raise NotImplementedError()

    def get_metric(self, submitter, res):
        """
        submitter.experiment()の結果からmetricを取得する
        """
        return submitter.get_metrics(res)["cv_mean"]

    def _objective(self, trial):
        params = self.sample_params(trial)
        res = self._experiment(params)
        return res

    def _study(self):
        study = optuna.create_study(direction=self.optimize_direction)
        self._initial_trial(study)
        self._set_kwown_trial(study)
        study.optimize(self._objective, n_trials=self.n_trials)
        best_params = study.best_trial.params
        print("Best params:", best_params)
        self._save_result(study)
        return best_params

    def _experiment(self, params):
        submitter = self._get_submitter(params)
        res = submitter.experiment(
            retrain_all_data=False, dry_run=False, return_only=True, save_model=False
        )
        res = self.get_metric(submitter, res)
        return res

    def _set_kwown_trial(self, study):
        for params, value in zip(self.known_params_list, self.known_values_list):
            known_trial = optuna.trial.create_trial(
                params=params, value=value, distributions=self.params_distribution
            )
            study.add_trial(known_trial)

    def _save_result(self, study):
        path = os.path.join(
            os.environ["DATASET_ROOT_DIR"],
            "artifact/tuning/",
            f"{self.submitter.submission_comment}_tuning_result.csv",
        )
        dir_name = os.path.dirname(path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        study.trials_dataframe().to_csv(path, index=False)

    def _initial_trial(self, study):
        for params in self.initial_trial_params:
            res = self._experiment(params)
            params = {k: v for k, v in params.items() if k not in self.fix_params}
            self.known_params_list.append(params)
            self.known_values_list.append(res)

    def _get_submitter(self, model_params):
        """
        TODO
        last foldのみ使う、データをsampleingするなど、1周を軽くする場合の実装
        """
        model = self.model_class(experiment_name="dummy", train_params=model_params)
        submitter = copy.copy(self.submitter)
        submitter.model = model
        return submitter


class LGBTuner(Tuner):
    def __init__(
        self,
        submitter,
        model_class,
        n_trials=100,
        optimize_direction="minimize",
        dry_run=False,
        known_params_list=[],
        known_values_list=[],
        initial_trial_params=[],
        params_distribution={
            "max_depth": optuna.distributions.CategoricalDistribution(
                [-1, 3, 5, 7, 9, 11]
            ),
            "num_leaves": optuna.distributions.IntUniformDistribution(10, 256),
            "min_data_in_leaf": optuna.distributions.IntUniformDistribution(10, 100),
            "subsample": optuna.distributions.FloatDistribution(0.1, 1),
            "colsample_bytree": optuna.distributions.FloatDistribution(0.1, 1),
            "reg_alpha": optuna.distributions.FloatDistribution(0, 10),
            "reg_lambda": optuna.distributions.FloatDistribution(0, 10),
        },
        fix_params=["n_iter", "learning_rate", "random_state", "subsample_freq"],
        seed=408,
    ):
        super().__init__(
            submitter=submitter,
            model_class=model_class,
            n_trials=n_trials,
            optimize_direction=optimize_direction,
            dry_run=dry_run,
            known_params_list=known_params_list,
            known_values_list=known_values_list,
            initial_trial_params=initial_trial_params,
            params_distribution=params_distribution,
            fix_params=fix_params,
            seed=seed
        )

    def _experiment(self, params):
        if self.dry_run:
            params["n_iter"] = 2
        return super()._experiment(params)

    def sample_params(self, trial) -> dict:
        params = {
            "n_iter": 200,
            "learning_rate": 1e-1,
            "random_state": self.seed,
            "subsample_freq": 1,
            "max_depth": trial.suggest_categorical("max_depth", [-1, 3, 5, 7, 9, 11]),
            "num_leaves": trial.suggest_int("num_leaves", 10, 256),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 100),
            "subsample": trial.suggest_float("subsample", 0.1, 1),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1),
            "reg_alpha": trial.suggest_float("reg_alpha", 0, 10),
            "reg_lambda": trial.suggest_float("reg_lambda", 0, 10),
            # "extra_trees": trial.suggest_categorical("extra_trees", [True, False]),
        }
        return params


class CBTuner(Tuner):
    def __init__(
        self,
        submitter,
        model_class,
        optimize_direction="minimize",
        n_trials=100,
        dry_run=False,
        known_params_list=[],
        known_values_list=[],
        initial_trial_params=[],
        params_distribution={
            "depth": optuna.distributions.IntDistribution(4, 10),
            "l2_leaf_reg": optuna.distributions.FloatDistribution(0, 10),
            "random_strength": optuna.distributions.FloatDistribution(0, 100),
            "bagging_temperature": optuna.distributions.FloatDistribution(0, 100),
            "colsample_bylevel": optuna.distributions.FloatDistribution(0.1, 1),
            "langevin": optuna.distributions.CategoricalDistribution([True, False]),
        },
        fix_params=["iterations", "random_state", "learning_rate"],
        seed=408,
    ):
        super().__init__(
            submitter=submitter,
            model_class=model_class,
            n_trials=n_trials,
            optimize_direction=optimize_direction,
            dry_run=dry_run,
            known_params_list=known_params_list,
            known_values_list=known_values_list,
            initial_trial_params=initial_trial_params,
            params_distribution=params_distribution,
            fix_params=fix_params,
            seed=seed
        )

    def _experiment(self, params):
        if self.dry_run:
            params["iterations"] = 2
        return super()._experiment(params)

    def sample_params(self, trial) -> dict:
        params = {
            "iterations": 1000,
            "learning_rate": 0.03,
            "random_state": self.seed,
            "depth": trial.suggest_int("depth", 3, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0, 10),
            "random_strength": trial.suggest_float("random_strength", 0, 500),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0, 500),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.1, 1),
            "langevin": trial.suggest_categorical("langevin", [True, False]),
        }
        return params