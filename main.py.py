# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 23:39:31 2018

@author: Art
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import talib as tl

class SnaptoCursor(object):
    def __init__(self, ax, x, y):
        self.ax = ax
        self.ly = ax.axvline(color="k", alpha=0.2)  # the vert line
        self.marker, = ax.plot([0],[0], marker="o", color="crimson", zorder=3) 
        self.x = x
        self.y = y
        self.txt = ax.text(0.7, 0.9, '')

    def mouse_move(self, event):
        if not event.inaxes: return
        x, y = event.xdata, event.ydata
        indx = np.searchsorted(self.x, [x])[0]
        x = self.x[indx]
        y = self.y[indx]
        self.ly.set_xdata(x)
        self.marker.set_data([x],[y])
        self.txt.set_text("x=%1.2f, y=%1.2f" % (x, y))
        self.txt.set_position((x,y))
        self.ax.figure.canvas.draw_idle()

def normalize(data, mini, maxi):
    normie = (data.copy() - mini.copy())/(maxi.copy() - mini.copy())
    return normie

def magnitude_removal(data):
    df = data.copy()
    for i in range(0, len(df)):
        if(df[i] == '-'):
            df[i] = df[i-1]
    mag = df.str.extract(r'[\d\.]+([KMB]+)', expand=False).fillna(1).replace(["K","M","B"], [10**3, 10**6, 10**9]).astype(int)
    base = df.replace(r'[KMB]+$', '', regex=True).astype(float)
    return base*mag


coins = np.array(os.listdir("./coins/"))
print(coins[0])
print(len(coins))

for i in range(0, len(coins)):

    dataset = pd.read_csv("./coins/"+coins[i]+"/"+coins[i]+"_raw_daily_data.csv")
    dataset = dataset.dropna()
    dataset = dataset[::-1].reset_index(drop=True)
    close = dataset.loc[:, "Price"]
    close = pd.to_numeric(close, errors="coerce")
    open = dataset.loc[:, "Open"]
    open = pd.to_numeric(open, errors="coerce")
    high = dataset.loc[:, "High"]
    high = pd.to_numeric(high, errors="coerce")
    low = dataset.loc[:, "Low"]
    low = pd.to_numeric(low, errors="coerce")
    vol = dataset.loc[:, "Vol."]
    vol = magnitude_removal(vol)
    
    t = np.arange(close.size)
    s = close[t]
    fig, ax = plt.subplots()
    cursor = SnaptoCursor(ax, t, s)
    cid =  plt.connect("motion_notify_event", cursor.mouse_move)
    ax.plot(t, s,)
    plt.title(coins[i].upper())
    plt.show()
    

    action = close.copy().rename("ACTION")
    action = action*0;
    print(i)
    up = np.loadtxt("./coins/"+coins[i]+"/"+coins[i]+"_up.txt", delimiter='-')
    down = np.loadtxt("./coins/"+coins[i]+"/"+coins[i]+"_down.txt", delimiter='-')
    
    for k in range(0, len(up[:])):
        for e in range(int(up[k,0]), int(up[k,1])+1):
            action[e] = 1;
    for k in range(0, len(down[:])):
        for e in range(int(down[k,0]), int(down[k,1])+1):
            action[e] = 2;
            

            
    ema5 = tl.EMA(close, timeperiod=5).rename("EMA5")
    ema10 = tl.EMA(close, timeperiod=10).rename("EMA10")
    w14 = tl.WILLR(high, low, close, timeperiod=14).rename("W14")
    rsi = tl.RSI(close, timeperiod=14).rename("RSI")
    mdi = tl.MINUS_DI(high, low, close, timeperiod=14).rename("MDI")
    pdi = tl.PLUS_DI(high, low, close, timeperiod=14).rename("PDI")
    natr = tl.NATR(high, low, close, timeperiod=14).rename("NATR")
    aroon = tl.AROONOSC(high, low, timeperiod=14).rename("AROON")
    cross = ema5.copy().rename("CROSS")
    
    for j in range(0, ema5.size):
        if(ema5[j] > ema10[j]):
            cross[j] = 1
        else:
            cross[j] = 0

    data = pd.concat([cross, w14, rsi, mdi, pdi, natr, aroon, action], axis=1)
    data = data.iloc[14:].reset_index(drop=True)
    data.to_csv("./coins/"+coins[i]+"/"+coins[i]+".csv")
    
combined_csv = pd.read_csv("./coins/"+coins[0]+"/"+coins[0]+".csv")
for i in range(1, len(coins)):
    combined_csv = pd.concat([combined_csv, pd.read_csv("./coins/"+coins[i]+"/"+coins[i]+".csv")])    
combined_csv = combined_csv.drop('Unnamed: 0', 1)
combined_csv.to_csv( "combined_csv.csv", index=False )
data= pd.read_csv('combined_csv.csv')

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X = data.iloc[:, 0:len(data.columns)-1].values
y = data.iloc[:,len(data.columns)-1].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
X_train_og = X_train
y_train_og = y_train

# Feature Scaling
from sklearn import preprocessing
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from keras.utils.np_utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# Building the network
classifier = Sequential()
classifier.add(Dense(units = 1024, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dropout(0.4))
classifier.add(Dense(units = 256, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dropout(0.3))
classifier.add(Dense(units = 64, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dropout(0.2))
classifier.add(Dense(units = 3, kernel_initializer = 'uniform', activation = 'softmax'))
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
history = classifier.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size = 100, epochs = 1000)

y_pred = np.round(classifier.predict(X_test))
correct_count = 0
for i in range (0,len(y_pred)):
    if( (y_pred[i] == y_test[i]).all()):
        correct_count = correct_count + 1
test_acc = correct_count/len(y_pred)

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
from ann_visualizer.visualize import ann_viz;
ann_viz(classifier, title="");

print(history.history.keys);
keys = history.history.keys

loss = history.history['loss']
acc = history.history['acc']
val_acc = history.history['val_acc']
val_loss = history.history['val_loss']

plt.plot(loss)
plt.title("Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")

plt.plot(acc)
plt.title("Training Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")

plt.plot(val_acc)
plt.title("Testing Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")

plt.plot(val_loss)
plt.title("Testing Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")

print(history.history.keys())
classifier.save('./eth/eth_model.h5')



