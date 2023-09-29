from comet_ml import Experiment
import optuna
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2)

# Experiment 1
experiment1 = Experiment(api_key="hcUmYvDjaVusNZMhAoxq7sVDR",
                         project_name="cometml-intro2",
                         workspace="maju116")
experiment1.set_name("Logistic Regression Experiment")
experiment1.add_tags(["LogisticRegression", "IrisDataset"])

clf1 = LogisticRegression(max_iter=200)
clf1.fit(X_train, y_train)

train_accuracy1 = clf1.score(X_train, y_train)
test_accuracy1 = clf1.score(X_test, y_test)

experiment1.log_metric("train_accuracy", train_accuracy1)
experiment1.log_metric("test_accuracy", test_accuracy1)
experiment1.end()


# Hyperparameter Optimization for Experiment 2
def objective(trial):
    experiment2 = Experiment(api_key="hcUmYvDjaVusNZMhAoxq7sVDR",
                             project_name="cometml-intro2",
                             workspace="maju116")
    experiment2.set_name("Random Forest Hyperparameter Optimization Trial")
    experiment2.add_tags(["RandomForest", "IrisDataset", "Optimization", "Trial"])

    n_estimators = trial.suggest_int('n_estimators', 2, 150)
    max_depth = trial.suggest_int('max_depth', 1, 32, log=True)
    min_samples_split = trial.suggest_float('min_samples_split', 0.1, 1)
    min_samples_leaf = trial.suggest_float('min_samples_leaf', 0.1, 0.5)
    max_features = trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2'])

    clf = RandomForestClassifier(n_estimators=n_estimators,
                                 max_depth=max_depth,
                                 min_samples_split=min_samples_split,
                                 min_samples_leaf=min_samples_leaf,
                                 max_features=max_features)

    score = cross_val_score(clf, X_train, y_train, n_jobs=-1, cv=3).mean()

    experiment2.log_parameters({
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "min_samples_leaf": min_samples_leaf,
        "max_features": max_features
    })
    experiment2.log_metric("cross_val_score", score)
    experiment2.end()

    return score


study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100, timeout=600)

best_params = study.best_params
best_score = study.best_value

# Logging the overall best model's results
experiment_overall = Experiment(api_key="hcUmYvDjaVusNZMhAoxq7sVDR",
                                project_name="cometml-intro2",
                                workspace="maju116")
experiment_overall.set_name("Random Forest Best Model")
experiment_overall.add_tags(["RandomForest", "IrisDataset", "Optimized"])

clf2 = RandomForestClassifier(**best_params)
clf2.fit(X_train, y_train)

train_accuracy2 = clf2.score(X_train, y_train)
test_accuracy2 = clf2.score(X_test, y_test)

experiment_overall.log_parameters(best_params)
experiment_overall.log_metric("best_cross_val_score", best_score)
experiment_overall.log_metric("train_accuracy", train_accuracy2)
experiment_overall.log_metric("test_accuracy", test_accuracy2)
experiment_overall.end()
