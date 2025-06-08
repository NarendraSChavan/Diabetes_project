# -*- coding: utf-8 -*-


import pandas as pd

diabetes = pd.read_csv('https://github.com/YBI-Foundation/Dataset/raw/main/Diabetes.csv')

diabetes.head()

diabetes.info()

diabetes.describe()

diabetes.columns

y = diabetes['diabetes']
X = diabetes[['pregnancies', 'glucose', 'diastolic', 'triceps', 'insulin', 'bmi',
       'dpf', 'age']]

diabetes.shape

diabetes['diabetes'].value_counts()

diabetes.groupby('diabetes').mean()

y=diabetes['diabetes']

y.shape

y

X=diabetes[['pregnancies','glucose','diastolic','triceps','insulin','bmi','dpf','age']]

X=diabetes.drop('diabetes',axis=1)

X.shape

X

from sklearn.preprocessing import MinMaxScaler

mm=MinMaxScaler()

X=mm.fit_transform(X)

X

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,stratify=y,random_state=2529)

X_train.shape,X_test.shape,y_train.shape,y_test.shape

from sklearn.linear_model import LogisticRegression

lr=LogisticRegression()

lr.fit(X_train,y_train)

y_pred=lr.predict(X_test)

y_pred.shape

y_pred

lr.predict_proba(X_test)

from sklearn.metrics import confusion_matrix,classification_report

print(confusion_matrix(y_test,y_pred))

print(classification_report(y_test,y_pred))

X_new = diabetes.sample(1)

X_new

X_new.shape

X_new = X_new.drop('diabetes',axis=1)

X_new

X_new.shape

X_new=mm.fit_transform(X_new)

y_pred_new= lr.predict(X_new)

y_pred_new

lr.predict_proba(X_new)

print('Predicted and Actual Class is Zero(0)that is Non-Diabetic')