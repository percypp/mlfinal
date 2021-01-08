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
cancelled=df.loc[(df['is_canceled']==1)]
a=100*(cancelled.shape[0]/df.shape[0])
df_test=pd.read_csv('test_adr.csv')
df.pop('reservation_status_date')
df.pop('arrival_date_year')
df.pop('ID')
df_test.pop('arrival_date_year')
df_test.pop('ID')
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
y=np.array(df.pop('is_canceled'))

x=df.values
test=df_test.values
x_train,x_test,y_train,y_test=train_test_split(x,y)

epochs=15

model = keras.Sequential([
    keras.layers.Dense(32,input_shape=(29,),activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(2, activation='softmax')
])
model.compile(optimizer='Adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
history=model.fit(x_train, y_train, epochs=epochs,validation_split=0.1,verbose=0)

def plot_confusion_matrix(cm, classes,
                        normalize=False,
                        title='Confusion matrix',
                        cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

history=history.history
epochs_num=np.arange(1,epochs+1)

plt.figure(0)
plt.title('accuracy')
plt.plot(epochs_num,history['accuracy'],label='accuracy')
plt.plot(epochs_num,history['val_accuracy'],label='val_accuracy')
plt.legend()
plt.show()

plt.figure(0)
plt.title('loss')
plt.plot(epochs_num,history['loss'],label='loss')
plt.plot(epochs_num,history['val_loss'],label='accuracy_loss')
plt.legend()
plt.show()

comp=model.predict(x_test)
comp=np.array([np.argmax(u) for u in comp])
cm = confusion_matrix(y_true=y_test, y_pred=comp)
plot_confusion_matrix(cm=cm,classes=['Non Canceled','Canceled'],title='Confusion Matrix')
plt.show()

plt.figure(figsize=(12,6))
cm = confusion_matrix(y_true=y_test, y_pred=comp)
plot_confusion_matrix(cm=cm,classes=['Non Canceled','Canceled'],title='Confusion Matrix',normalize=True)
plt.show()
print('F1 score '+str(f1_score(y_true=y_test,y_pred=comp)))

result = model.predict(test)