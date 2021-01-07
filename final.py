import numpy as np 
import pandas as pd
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
import seaborn as sns
import itertools
from sklearn.metrics import f1_score
from sklearn import preprocessing

enc = preprocessing.OneHotEncoder(categories='auto')

df_train = pd.read_csv('train.csv')
df_label = pd.read_csv('train_label.csv')
df_test = pd.read_csv('test.csv')
df_test_nolabel = pd.read_csv('test_nolabel.csv')
df_train = pd.DataFrame(df_train)
df_test = pd.DataFrame(df_test)

df_train = df_train.drop('reservation_status', axis = 1)
df_train = df_train.drop('is_canceled', axis = 1)
df_train = df_train.drop('adr', axis = 1)
df_train.pop('ID')
df_test.pop('ID')
df_train['reservation_status_date'] = pd.to_datetime(df_train['reservation_status_date'])
df_label['arrival_date'] = pd.to_datetime(df_label['arrival_date'])
df_train = df_train.sort_values(by=['reservation_status_date'])
df_test['reservation_status_date']=pd.to_datetime(df_test['arrival_date_year'].astype(int).astype(str) + df_test['arrival_date_month'] + df_test['arrival_date_day_of_month'].astype(int).astype(str),format='%Y%B%d')
df_test = df_test.sort_values(by=['reservation_status_date'])

for col_name in df_train.columns:
    if(df_train[col_name].dtype == 'object'):
        df_train[col_name]= df_train[col_name].astype('category')
        df_train[col_name] = df_train[col_name].cat.codes

for col_name in df_train.columns:
    if(df_test[col_name].dtype == 'object'):
        df_test[col_name]= df_test[col_name].astype('category')
        df_test[col_name] = df_test[col_name].cat.codes

df_train = df_train.fillna(0)
df_test = df_test.fillna(0)
sectors = df_train.groupby("reservation_status_date").sum()
sectors = sectors['2015-07-01':'2017-03-31']

df_test = df_test.groupby("reservation_status_date").sum()


y=np.array(df_label.pop('label'))
y = keras.utils.to_categorical(y, 10)
x=sectors.values
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
history = model.fit(x_train, y_train, epochs=epochs,validation_split=0.2,verbose=0)
history=history.history
epochs_num=np.arange(1,epochs+1)
epochs_num=np.arange(1,epochs+1)

# plt.figure(0)
# plt.title('accuracy')
# plt.plot(epochs_num,history['accuracy'],label='accuracy')
# plt.plot(epochs_num,history['val_accuracy'],label='val_accuracy')
# plt.legend()
# plt.show()

# plt.figure(0)
# plt.title('loss')
# plt.plot(epochs_num,history['loss'],label='loss')
# plt.plot(epochs_num,history['val_loss'],label='accuracy_loss')
# plt.legend()
# plt.show()
result = model.predict(df_test)
def props_to_onehot(props):
    if isinstance(props, list):
        props = np.array(props)
    a = np.argmax(props, axis=1)
    b = np.zeros((len(a), props.shape[1]))
    b[np.arange(len(a)), a] = 1
    return b
result = props_to_onehot(result)
result1 = [np.where(r==1)[0][0] for r in result]
result1 = pd.DataFrame(result1)
result2 = pd.concat([df_test_nolabel,result1],axis=1)
result2.to_csv('result.csv', mode='a', index=False)