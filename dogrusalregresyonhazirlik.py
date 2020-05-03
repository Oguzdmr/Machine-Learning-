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


veriler = pd.read_csv('satislar.csv')

print(veriler)

#veri ön isleme

aylar= veriler[['Aylar']]
print(aylar)
satislar = veriler [['Satislar']]
print(satislar)





#verilerin eğitim ve test için bölünmesi
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(aylar,satislar,test_size=0.33,random_state=0)


'''
#verilerin ölçeklendirilmesi(standatlaştırma)
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()
X_train=sc.fit_transform(x_train)

X_test=sc.fit_transform(x_test)

Y_train=sc.fit_transform(y_train)

Y_test=sc.fit_transform(y_test)

'''

 
#model inşası (linear regression)
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train)

tahmin = lr.predict(x_test)

x_train = x_train.sort_index()
y_train = y_train.sort_index()

plt.plot(x_train,y_train)
plt.plot(x_test,lr.predict(x_test))
plt.title('aylara göre satışlar')
plt.xlabel('Aylar')
plt.ylabel('Satışlar')


















