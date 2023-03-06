import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.model_selection import cross_validate, GridSearchCV

rmse_loss = make_scorer(mean_squared_error, squared=False, greater_is_better=False)

##############################################
##############################################
##############################################

class ModelTuner:
    def __init__(self, models, X, y, cv, loss):
        self.init_models = models.copy()
        self.models = models.copy()
        self.X_train = X.copy()
        self.y_train = y.copy()
        self.cv = cv
        self.loss = loss
        self.train_scores = {}
        self.valid_scores = {}
        
    def tune_model(self, model_name, grid, verbose=1, n_jobs=-1):
        self.models[model_name] = self.init_models[model_name]
        search = GridSearchCV(
            self.models[model_name],
            param_grid=grid,
            scoring=self.loss,
            cv=self.cv,
            return_train_score=True,
            verbose=verbose,
            n_jobs=n_jobs,
        )
        search.fit(self.X_train, self.y_train)
        self.models[model_name] = search.best_estimator_
        
        rename_metric = {
            "mean_train_score": "mean_train_RMSE",
            "mean_test_score": "mean_valid_RMSE",
        }        
        
        cv_results = pd.DataFrame(search.cv_results_)
        columns = [f"param_{param}" for param, val in grid.items() if len(val)>1]
        columns.extend(rename_metric.keys())
        cv_results = cv_results[columns].rename(columns=rename_metric)
        cv_results[list(rename_metric.values())] = -cv_results[list(rename_metric.values())]
        cv_results = cv_results.sort_values(by="mean_valid_RMSE", ignore_index=True)

        train_score = cv_results.mean_train_RMSE.iloc[0]
        valid_score = cv_results.mean_valid_RMSE.iloc[0]
        self.train_scores[model_name] = train_score
        self.valid_scores[model_name] = valid_score
        
        print("="*40)
        print(f"Model: {search.best_estimator_}")
        print(f"Train RMSE: {train_score:.5f}")
        print(f"Valid RMSE: {valid_score:.5f}")
        print("="*40)

        return cv_results
    
    def collate_results(self):
        train_results = pd.DataFrame.from_dict(
            self.train_scores,
            orient="index",
            columns=["mean_train_RMSE"],
        )
        valid_results = pd.DataFrame.from_dict(
            self.valid_scores,
            orient="index",
            columns=["mean_valid_RMSE"],
        )
        results = pd.concat(
            [train_results, valid_results],
            axis=1,
        ).sort_values(by="mean_valid_RMSE")
        
        return results
    
    def get_models(self, model_names=None):
        if not model_names:
            model_names = self.models.keys()
            
        models = [(name, self.models[name]) for name in model_names]
        
        return models
    
    def run_cv(self, model, n_jobs=-1):
        model = self.models[model] if model in self.models else model
        scores = cross_validate(
            model,
            X=self.X_train,
            y=self.y_train,
            cv=self.cv,
            return_train_score=True,
            scoring=self.loss,
            n_jobs=n_jobs,
        )
        train_scores = scores["train_score"]
        valid_scores = scores["test_score"]
        
        print(f"Model: {model}")
        print(f"Train RMSE: {-np.mean(train_scores):.5f}")
        print(f"Valid RMSE: {-np.mean(valid_scores):.5f}")

    
    def model_predict(self, model, X):
        model = self.models[model] if model in self.models else model
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(X)
        y_pred = pd.DataFrame({
            "Id": range(1461, 1461+len(y_pred)),
            "SalePrice": np.expm1(y_pred),
        })

        return y_pred
    
    def perf_boxplot(self, add_models=None, cv=None, n_jobs=-1):
        models = self.models.copy()
        if add_models:
            for name, model in add_models:
                models[name] = model
            
        perf_df = pd.DataFrame()
        for name, model in models.items():
            valid_scores = cross_validate(
                model,
                X=self.X_train,
                y=self.y_train,
                cv=cv if cv else self.cv,
                scoring=self.loss,
                n_jobs=n_jobs,
            )["test_score"]
            
            model_perf = pd.DataFrame({"Model": name, "Validation RMSE": -valid_scores})
            perf_df = pd.concat([perf_df, model_perf])
        
        min_median = perf_df.groupby("Model").median().min()[0]
        
        plt.figure(figsize=(18, 8))
        
        sns.boxplot(
            x=perf_df["Validation RMSE"],
            y=perf_df["Model"],
            flierprops={"alpha": 0.5},
        )
        plt.axvline(x=min_median, ls="--", color="coral")
        plt.ylabel("")
        plt.title("Model Performance", fontsize="large")
        plt.tight_layout()