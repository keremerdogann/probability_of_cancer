import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
import seaborn as sns
import statsmodels.api as sm
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

df = pd.read_csv("cancer-probabilities.csv")
df = df.set_index("Sr No.")

#print(df)

df = df.dropna()


Y2 = df["Probability of Cancer"]


#KANSER OLMA OLASILIGINI 0 1 ŞEKLİNDE DEĞİŞTİRME
df['Probability of Cancer'] = pd.cut(df['Probability of Cancer'], bins=[0, 0.5, 1], labels=[0, 1])

Y = df["Probability of Cancer"]

X = df.drop(columns=["Probability of Cancer"])

print(X)

print(Y)


print(df["Smoking Habit"].unique())

#üstünlükleri ayarlıyoruz ( we set up which comes first or last)

smoke_hab = {
'Occasional' : 1 ,
'Moderate' : 2,
'Heavy' : 3
}

drink_hab = {
    'Occasional' : 1,
    'Moderate' : 2,
    'Frequent' : 3
}

biking_hab = {
    'Low':1,
    'Medium':2,
    'High':3
}

walking_hab = {
    'Low':1,
    'Medium':2,
    'High':3
}

jogging_hab = {
    'Low': 1,
    'Medium': 2,
    'High': 3
}

X["Smoking Habit"] = X["Smoking Habit"].replace(smoke_hab)
X["Drinking Habit"] = X["Drinking Habit"].replace(drink_hab)
X["Biking Habit"] = X["Biking Habit"].replace(biking_hab)
X["Walking Habit"] = X["Walking Habit"].replace(walking_hab)
X["Jogging Habit"] = X["Jogging Habit"].replace(jogging_hab)

print("AFTER REPLACE")

print(X)

x_train , x_test , y_train , y_test = train_test_split(X,Y,test_size=0.70,random_state=0)

x_train2 , x_test2 , y_train2 , y_test2 = train_test_split(X,Y2,test_size=0.70,random_state=0)

log_r = LogisticRegression(random_state=0)

log_r.fit(x_train,y_train)

y_pred = log_r.predict(x_test)

y_real = y_test

cm=confusion_matrix(y_real,y_pred)

print(cm)

lr = LinearRegression()

lr.fit(x_train2,y_train2)

y_real2 = y_test2

y_pred2 = lr.predict(x_test2)

real_values = pd.DataFrame(y_test2)
pred_values = pd.DataFrame(y_pred2)

print(real_values)
print(pred_values)

X_train_sm = sm.add_constant(x_train2)  # Eğitim verisi için
X_test_sm = sm.add_constant(x_test2)    # Test verisi için

model = sm.OLS(y_train2, X_train_sm).fit()
print("\n OLS Model Özeti:")
print(model.summary())
















