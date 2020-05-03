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


veriler = pd.read_csv('eksikveriler.csv')



#veri ön isleme

boy= veriler[['boy']]

boykilo = veriler [['boy','kilo']]


#eksik veriler

from sklearn.impute import SimpleImputer
im = SimpleImputer(missing_values = np.nan, strategy = 'mean')
Yas =veriler.iloc[:,1:4].values
print(Yas)
im.fit(Yas[:,1:4])
Yas[:,1:4] =im.transform(Yas[:,1:4])
print(Yas)

#encoder Kategorik Veri -> Numeric
ulke =veriler.iloc[:,0:1].values
print(ulke)

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le = LabelEncoder()
ulke[:,0]=le.fit_transform(ulke[:,0])
print(ulke)


ohe = OneHotEncoder(categories='auto')
ulke=ohe.fit_transform(ulke).toarray()
print(ulke)

#numpy dizilerin dataframe e dönüşümü
sonuc = pd.DataFrame(data=ulke,index =range(22),columns=['fr','tr','us'])
print(sonuc)

sonuc2=pd.DataFrame(data=Yas,index=range(22),columns=['boy','kilo','yaş'])
print(sonuc2)

cinsiyet =  veriler.iloc[:,-1].values
print(cinsiyet)

sonuc3=pd.DataFrame(data=cinsiyet,index = range(22),columns=['cinsiyet'])
print(sonuc3)


#dataframe birleştirme işlemi
s=pd.concat([sonuc,sonuc2],axis=1)
print(s)

s2=pd.concat([s,sonuc3],axis=1)
print(s2)



#verilerin eğitim ve test için bölünmesi
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(s,sonuc3,test_size=0.33,random_state=0)



#verilerin ölçeklendirilmesi(standatlaştırma)
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()
X_train=sc.fit_transform(x_train)

X_test=sc.fit_transform(x_test)













