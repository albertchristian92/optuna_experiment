import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from cuml.ensemble import RandomForestClassifier as cuRFC
from sklearn.ensemble import RandomForestClassifier as skruf
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.model_selection import train_test_split
import numpy as np

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


forest_upsampled = skruf(bootstrap= True,max_depth= 2, max_features= 'auto', n_estimators= 452)


gbm = forest_upsampled.fit(np.float32(X_train), np.float32(y_train))
predictions = gbm.predict(np.float32(X_test))
accuracy = accuracy_score(y_test, y_pred=predictions)

print(accuracy)
