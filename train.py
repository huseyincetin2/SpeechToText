import librosa as lr
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import IPython.display as ipd
import keras
from sklearn.model_selection import train_test_split
import tensorflow as tf
import pickle as pk
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Dropout, Flatten, Dense
import sys


dataset = './dataset/augmented_dataset/augmented_dataset'
pd.DataFrame(os.listdir(dataset), columns=['Files'])

def count(path):
    size = []
    for file in os.listdir(path):
        size.append(len(os.listdir(os.path.join(path, file))))
    return pd.DataFrame(size, columns=['Number Of Sample'], index=os.listdir(path))

tr = count(dataset)

print(tr)

def load(path):
    data = []
    label = []
    sample = []
    for file in os.listdir(path):
        path_ = os.path.join(path, file)
        if os.path.isdir(path_):  # Sadece dizinleri işleme al
            for fil in os.listdir(path_):
                full_path = os.path.join(path_, fil)
                if os.path.isfile(full_path):  # Sadece dosyaları işleme al
                    data_contain, sample_rate = lr.load(full_path, sr=16000)
                    data.append(data_contain)
                    sample.append(sample_rate)
                    label.append(file)
    return data, label, sample

# Örnek kullanım
data, label, sample = load(dataset)
#print(data, label, sample)

code = {}
x = 0
for i in np.unique(label):
    code[i] = x
    x += 1

pd.DataFrame(code.values(), columns=['Value'], index=code.keys())

def get_Name(N):
    for x, y in code.items():
        if y == N:
            return x

for i in range(len(label)):
    label[i] = code[label[i]]
pd.DataFrame(label, columns=['Labels'])

data = np.array(data).reshape(-1, 16000, 1)

label = np.array(label)

X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.1, random_state=44, shuffle =True)
print('X_train shape is ', X_train.shape)
print('X_test shape is ', X_test.shape)
print('y_train shape is ', y_train.shape)
print('y_test shape is ', y_test.shape)

num_class = len(pd.unique(label))
print(num_class)
model = Sequential([
    Conv1D(filters=8, kernel_size=13, activation='relu', input_shape=(16000,1)),
    MaxPooling1D(pool_size=3),
    Dropout(0.3),
    Conv1D(filters=16, kernel_size=11, activation='relu'),
    MaxPooling1D(pool_size=3),
    Dropout(0.3),
    Conv1D(filters=32, kernel_size=9, activation='relu'),
    MaxPooling1D(pool_size=3),
    Dropout(0.3),
    Conv1D(filters=64, kernel_size=7, activation='relu'),
    MaxPooling1D(pool_size=3),
    Dropout(0.3),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(num_class, activation='softmax')
])

tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True,show_dtype=True,dpi=120)
model.compile(optimizer ='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
hist=model.fit(X_train,y_train,epochs=10)
model.save('./model/m2.keras')

model.summary()

loss,acc=model.evaluate(X_test,y_test)
print('Loss is :',loss)
print('ACC is :',acc)

plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
plt.plot(hist['loss'],c='r',marker='*',label='Loss')
plt.title('Overall Loss',fontsize=20)
plt.legend()
plt.subplot(1,2,2)
plt.plot(hist['accuracy'],label='Accuracy')
plt.title('Overall Accuracy',fontsize=20)
plt.legend()

ClassificationReport = classification_report(y_test,preN)
print('Classification Report is : ', ClassificationReport)




