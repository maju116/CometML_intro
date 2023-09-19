from comet_ml import Experiment
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# Initialize multiple experiments
experiment1 = Experiment(api_key="hcUmYvDjaVusNZMhAoxq7sVDR",
                        project_name="cometml-intro",
                        workspace="maju116")
experiment2 = Experiment(api_key="hcUmYvDjaVusNZMhAoxq7sVDR",
                        project_name="cometml-intro",
                        workspace="maju116")

data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2)

clf1 = LogisticRegression(max_iter=200)
clf1.fit(X_train, y_train)

train_accuracy1 = clf1.score(X_train, y_train)
test_accuracy1 = clf1.score(X_test, y_test)

experiment1.log_metric("train_accuracy", train_accuracy1)
experiment1.log_metric("test_accuracy", test_accuracy1)

experiment1.end()

clf2 = DecisionTreeClassifier()
clf2.fit(X_train, y_train)

train_accuracy2 = clf2.score(X_train, y_train)
test_accuracy2 = clf2.score(X_test, y_test)

experiment2.log_metric("train_accuracy", train_accuracy2)
experiment2.log_metric("test_accuracy", test_accuracy2)

experiment2.end()
