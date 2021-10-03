import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from cuml.ensemble import RandomForestClassifier as cuRFC
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.model_selection import train_test_split

import optuna
import numpy as np
# Load the data

df = pd.read_csv('UCI_Credit_Card.csv')
df = df.rename(columns={'default.payment.next.month': 'def_pay',
                        'PAY_0': 'PAY_1'})

features = ['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_1', 'PAY_2',
       'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2',
       'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1',
       'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
X = df[features].copy()
X.columns

#%%

# create the target variable
y = df['def_pay'].copy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)


# this means we will train on 80% of the data and test on the remaining 20%.

# %%

def objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 100, 1200)
    split_criterion = trial.suggest_categorical("criterion", ["gini", "entropy"])
    max_depth = trial.suggest_int('max_depth', 3, 30)
    # min_samples_split = trial.suggest_int('min_samples_split', 2, 100)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 100)
    max_features = trial.suggest_float('max_features', 0.1, 0.9)
    bootstrap = trial.suggest_categorical("bootstrap", ["True", "False"])
    # clf = cuRFC(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, max_features=max_features, bootstrap=bootstrap)
    clf = cuRFC(n_estimators=n_estimators, split_criterion=split_criterion, max_depth=max_depth)

    gbm = clf.fit(np.float32(X_train), np.float32(y_train))
    predictions = gbm.predict(np.float32(X_test))
    accuracy = accuracy_score(y_test, y_pred=predictions)
    return accuracy


study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

trial = study.best_trial

print('Number of finished trials: ', len(study.trials))
print("Best trial: ", study.best_trial.params)
