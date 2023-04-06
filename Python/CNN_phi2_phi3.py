# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 10:41:05 2023

@author: arjun
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from scipy.ndimage import gaussian_filter1d
import scipy.constants as const
import pandas as pd
import tensorflow as tf
import os
import h5py

class MonteCarloDropout(tf.keras.layers.Dropout):
  def call(self, inputs):
    return super().call(inputs, training=True)

wdir = os.path.dirname(os.path.abspath(__file__))
os.chdir(wdir)

root_dir = '../'
filename = 'train_set_ph2_ph3_-30k-0k_45fspulse.h5'
filename = root_dir+'TrainingData/'+filename

f = h5py.File(filename, "r")
A = list(f['Spectra'])
phi3 = np.array(f['TOD']).reshape(-1,)
phi2 = np.array(f['GDD']).reshape(-1,)
df = pd.DataFrame({'X':A,'phi2':phi2,'phi3':phi3})
f.close()

dim = int(df['X'][0].shape[0])

df['ph2_norm'] = (df['phi2']-min(df['phi2']))/(max(df['phi2'])-min(df['phi2']))
df['ph3_norm'] = (df['phi3']-min(df['phi3']))/(max(df['phi3'])-min(df['phi3']))

train, test       = train_test_split(df, test_size=0.2)
train_X, train_Y  = np.stack(train['X']),np.array(train[['ph2_norm','ph3_norm']])
val, test         = train_test_split(test, test_size=0.01)
val_X, val_Y      = np.stack(val['X']),np.array(val[['ph2_norm','ph3_norm']])
test_X, test_Y    = np.stack(test['X']),np.array(test[['ph2_norm','ph3_norm']])

train_X = train_X.reshape(-1, dim,dim, 1)
val_X   = val_X.reshape(-1, dim,dim, 1)
test_X  = test_X.reshape(-1, dim,dim, 1)

inputs= tf.keras.Input(shape=(dim,dim,1))

x = tf.keras.layers.Conv2D(filters=96, kernel_size=(4, 4), padding= 'same', activation='relu')(inputs)
x = tf.keras.layers.MaxPool2D(padding='same')(x)
x = tf.keras.layers.Conv2D(filters=96, kernel_size=(4, 4), padding= 'same', activation='relu')(x)
x = tf.keras.layers.MaxPool2D(padding='same')(x)
#x = tf.keras.layers.Conv2D(filters=96, kernel_size=(4, 4), padding= 'same', activation='relu')(x)
#x = tf.keras.layers.MaxPool2D(padding='same')(x)
x = tf.keras.layers.Flatten()(x)

#x = MonteCarloDropout(0.3)(x)
#x = tf.keras.layers.Dense(3000, activation='relu')(x)
x = MonteCarloDropout(0.3)(x)
x = tf.keras.layers.Dense(800, activation='relu')(x)
x = MonteCarloDropout(0.3)(x)
x = tf.keras.layers.Dense(400, activation='relu')(x)
x = MonteCarloDropout(0.3)(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)

outputs = tf.keras.layers.Dense(2, activation='linear')(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

#%%

optimiser = tf.keras.optimizers.Adam(learning_rate=0.0001) #,decay=1e-3)
model.compile(optimizer=optimiser,loss='mse')
es = EarlyStopping(monitor='val_loss', mode='min', patience=10, restore_best_weights=True, verbose=1)

#%%

history = model.fit(x=train_X,y=train_Y,validation_data=(val_X,val_Y),batch_size=256, epochs=30, callbacks= [es])
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

#%% Method with Dropout layer

cmprsr = np.load(root_dir+'cmprsr.npy')
dx = 265- 28.1
cmprsr += dx
Spec_X = np.load(root_dir+'test_images_64x64.npy')
Spec_X = Spec_X.reshape(-1, dim,dim, 1)
N = Spec_X.shape[0]
Spec_X = Spec_X/np.max(Spec_X)
Spec_X = gaussian_filter1d(Spec_X,10,axis=1)
Spec_Y = np.array([model(Spec_X, training=True) for _ in range(100)]).T
ph2, ph3 = Spec_Y[0]*(max(df['phi2'])-min(df['phi2'])) + min(df['phi2']) , Spec_Y[1]*(max(df['phi3'])-min(df['phi3'])) + min(df['phi3'])

phi2,phi2_err = np.mean(ph2,1), np.std(ph2,1)
phi3,phi3_err = np.mean(ph3,1), np.std(ph3,1)

Phi_df = pd.DataFrame({'phi2':phi2,'phi3': phi3,'phi2_error': phi2_err,'phi3_error':phi3_err})
#Phi_df.to_csv(root_dir+'Predictions/phi_pred_with_noise_45fs_test_17_02.csv')

plt.figure(1)
plt.errorbar(cmprsr,phi2, yerr= phi2_err,ls='None',color='black',capsize=5)
plt.plot(cmprsr,phi2,'.r')
plt.grid()
plt.xlabel('Distance between gratings (mm)',fontsize=15)
plt.ylabel('GDD (fs$^2$)',fontsize=15)
plt.tight_layout()
plt.xticks(np.arange(264,266.5,0.5))

plt.figure(2)
plt.errorbar(cmprsr, phi3, yerr= phi3_err ,ls='None',color='black',capsize=5)
plt.plot(cmprsr,phi3,'.r')
plt.grid()
plt.xlabel('Distance between gratings (mm)',fontsize=15)
plt.ylabel('TOD (fs$^3$)',fontsize=15)
plt.tight_layout()
plt.xticks(np.arange(264,266.5,0.5))

#%% Computing a linear fit to the predictions

c0 = 2.9979e8;
wl = 800e-9;
d = 1e-3/1500;
theta = 51.3*np.pi/180;
omega = 2*np.pi*c0/wl;
cc= (2*np.pi*c0/omega/d);

D2_slope = 2*(4*np.pi**2*const.c/omega**3/d**2)*(1-(cc-np.sin(theta))**2)**-1.5
D3_slope = -(D2_slope/2/omega)*((1+(cc*np.sin(theta))-(np.sin(theta))**2)/(1-(cc-np.sin(theta))**2))

D2_slope *= 1e27
D3_slope *= 1e42

def lin_fit_GDD(x,c):
    y = D2_slope*x+c
    return(y)
def lin_fit_TOD(x,c):
    y = D3_slope*x+c
    return(y)

fit2,cov2 = curve_fit(lin_fit_GDD,cmprsr,phi2,sigma=phi2_err,absolute_sigma=True)
fit3,cov3 = curve_fit(lin_fit_TOD,cmprsr,phi3,sigma=phi3_err,absolute_sigma=True)

D2_fit = D2_slope*cmprsr+fit2[0]
plt.figure(1)
plt.plot(cmprsr,D2_fit)

D3_fit = D3_slope*cmprsr+fit3[0]
plt.figure(2)
plt.plot(cmprsr,D3_fit)
