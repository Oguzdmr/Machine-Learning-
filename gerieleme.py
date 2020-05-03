# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#1. Kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#2. Veri Önisleme

#2.1 Veri Yukleme 


veriler = pd.read_csv('odev_tenis.csv')



#encoder Kategorik Veri -> Numeric
outlook =veriler.iloc[:,0:1].values
print(outlook)

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le = LabelEncoder()
outlook[:,0]=le.fit_transform(outlook[:,0])
print(outlook)


ohe = OneHotEncoder(categories='auto')
outlook=ohe.fit_transform(outlook).toarray()
print(outlook)


play =veriler.iloc[:,-1:].values
print(play)

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le = LabelEncoder()
play2=le.fit_transform(play[:,0])
print(play)



windy =veriler.iloc[:,3:4].values
print(windy)
le = LabelEncoder()
windy2=le.fit_transform(windy[:,0])
print(windy2)


#numpy dizilerin dataframe e dönüşümü
sonuc = pd.DataFrame(data=outlook,index =range(14),columns=['rainy','overcast','sunny'])
print(sonuc)

temp =veriler.iloc[:,1:3].values
sonuc2=pd.DataFrame(data=temp,index=range(14),columns=['emperature','humidity'])
print(sonuc2)

play =  veriler.iloc[:,-1].values
print(play)

sonuc3=pd.DataFrame(data=windy2,index = range(14),columns=['windy'])
print(sonuc3)

sonuc4=pd.DataFrame(data=play2,index = range(14),columns=['play'])
print(sonuc4)



#dataframe birleştirme işlemi
s=pd.concat([sonuc,sonuc2],axis=1)
print(s)

s2=pd.concat([s,sonuc3],axis=1)
print(s2)

s3=pd.concat([s2,sonuc4],axis=1)
print(s3)

'''
#verilerin eğitim ve test için bölünmesi
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(s,sonuc3,test_size=0.33,random_state=0)



#verilerin ölçeklendirilmesi(standatlaştırma)
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()
X_train=sc.fit_transform(x_train)

X_test=sc.fit_transform(x_test)



from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)

boy = s2.iloc[:,3:4].values
print(boy)
sol = s2.iloc[:,:3]
sag = s2.iloc[:,4:]

veri = pd.concat([sol,sag],axis=1)
x_train,x_test,y_train,y_test = train_test_split(veri,boy,test_size=0.33,random_state=0)

r2 = LinearRegression()
r2.fit(x_train,y_train)

y_pred = r2.predict(x_test)


import statsmodels.api as sm 

X = np.append(arr = np.ones((22,1)).astype(int),values = veri, axis=1 )
X_l = veri.iloc[:,[0,1,2,3,4,5]].values
r_ols = sm.OLS(endog = boy,exog = X_l).fit()
print(r_ols.summary())


X = np.append(arr = np.ones((22,1)).astype(int),values = veri, axis=1 )
X_l = veri.iloc[:,[0,1,2,3,5]].values
r_ols = sm.OLS(endog = boy,exog = X_l).fit()
print(r_ols.summary())



X = np.append(arr = np.ones((22,1)).astype(int),values = veri, axis=1 )
X_l = veri.iloc[:,[0,1,2,3]].values
r_ols = sm.OLS(endog = boy,exog = X_l).fit()
print(r_ols.summary())




'''






