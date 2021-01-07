import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
import seaborn as sns
import itertools
from sklearn.metrics import f1_score

df=pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
df_test_nolabel = pd.read_csv('test_nolabel.csv')

df.pop('ID')
df.pop('reservation_status')
df.pop('adr')
df.pop('is_canceled')
df['reservation_status_date'] = pd.to_datetime(df['reservation_status_date'])
df = df.sort_values(by=['reservation_status_date'])

df_test['reservation_status_date']=pd.to_datetime(df_test['arrival_date_year'].astype(int).astype(str) + df_test['arrival_date_month'] + df_test['arrival_date_day_of_month'].astype(int).astype(str),format='%Y%B%d')
df_test.pop('ID')
df_test = df_test.sort_values(by=['reservation_status_date'])

# remove outlier
df=df.loc[(df['adults']<20)]
df=df.loc[(df['children']<10)]


for col_name in df.columns:
    if(df[col_name].dtype == 'object'):
        df[col_name]= df[col_name].astype('category')
        df[col_name] = df[col_name].cat.codes
for col_name in df_test.columns:
    if(df_test[col_name].dtype == 'object'):
        df_test[col_name]= df_test[col_name].astype('category')
        df_test[col_name] = df_test[col_name].cat.codes

df=df.fillna(0)
df_test=df_test.fillna(0)

df1 = df.groupby("reservation_status_date").sum()
df = df1['2015-07-01':'2017-03-31']
df_test = df_test.groupby("reservation_status_date").sum()
df_label = pd.read_csv('train_label.csv')
y = np.array(df_label.pop('label'))
y = keras.utils.to_categorical(y, 10)
x=df.valuess
x_train,x_test,y_train,y_test=train_test_split(x,y)

epochs=15

model = keras.Sequential([
    keras.layers.Dense(32,input_shape=(28,),activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='Adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
history=model.fit(x_train, y_train, epochs=epochs,validation_split=0.1,verbose=0)

result = model.predict(x)

result1 = pd.DataFrame(result)
print(result1)
# result2 = pd.concat([df_test_nolabel,result1],axis=1)
# result2.to_csv('result.csv', mode='a', index=False)