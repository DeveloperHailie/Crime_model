# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 17:23:35 2020

@author: 미정관 406호 걸즈
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn import utils
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import LabelEncoder
tf.random.set_seed(777) #하이퍼파라미터 튜닝을 위해 실행시 마다 변수가 같은 초기값 가지게 하기


## 데이터 로드
df = pd.read_csv('PoliceDepartmentIncidents2016.csv')

## 데이터 분석
print("head")
print(df.head())

print("info")
print(df.info())

print("describe")
print(df.describe())



## 데이터 전처리

# 문자열을 숫자로 변환
e = LabelEncoder()

# DayOfWeek 숫자로 변환
DayOfWeek = df[['DayOfWeek']].values
e.fit(DayOfWeek)
DayOfWeek = e.transform(DayOfWeek)
df['DayOfWeek'] = DayOfWeek
# Time , 07시~11시 아침 / 12~17시 낮 / 18시~00시 저녁 / 01시~06시 새벽
Time = df[['Time']].values
for i in range(0,len(Time)) :
    time = Time[i][0][0:2]
    if (time=='07' or time=='08' or time=='09' or time=='10' or time=='11') :
        Time[i] = 0
    elif (time=='12' or time=='13' or time=='14' or time=='15' or time=='16' or time=='17') :
        Time[i] = 1
    elif (time=='18' or time=='19' or time=='20' or time=='21' or time=='22' or time=='23' or time=='00') :
        Time[i] = 2
    elif (time=='01' or time=='02' or time=='03' or time=='04' or time=='05' or time=='06') :
        Time[i] = 3
df[['Time']] = Time

# 데이터 분류
dataset = df[['DayOfWeek','Time','X','Y','Category']].values
X = dataset[:,0:4].astype(float)
Y_obj = dataset[:,4]
print(X)
print(Y_obj)

# Y 데이터 변환(숫자, 배열)
e.fit(Y_obj)
Y = e.transform(Y_obj)
Y_encoded = tf.keras.utils.to_categorical(Y)

print(X)
print(Y_encoded)

## 모델의 설정
model = Sequential()
model.add(Dense(16,  input_dim=4, activation='relu'))
#model.add(Dense(32, activation='relu'))
model.add(Dense(39, activation='softmax'))

# 모델 컴파일
model.compile(loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy'])

# 모델 실행
model.fit(X, Y_encoded, epochs=50, batch_size=100)

# 결과 출력
print("\n Accuracy: %.4f" % (model.evaluate(X, Y_encoded)[1]))