#!/usr/bin/env python
# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import tensorflow as tf
from pandas import DataFrame
import tensorflow.keras.layers as layers
import os
from tensorflow import keras
from tensorflow.keras import regularizers


def get_all_data(path):

    path_name = path
    data_name = os.listdir(path_name)
    len_data = len(data_name)
    data_dir = []
    for i in range(len_data):
        data_path = os.path.join(path_name, data_name[i])
        data_dir.append(data_path)

    return data_dir

def shuffle_data(dataset):
    np.random.seed(1)
    shuffle_index = np.arange(len(dataset))
    np.random.shuffle(shuffle_index)
    return shuffle_index


def generate_train_test_number(shuffle_index):
    
    train_number = shuffle_index[:420]
    validation_number = shuffle_index[420:420+60]
    test_number = shuffle_index[420+60:]

    return train_number, validation_number, test_number

def deal_data(dataset):
    eis_real = dataset[:,1][:,np.newaxis]
    eis_img = -dataset[:,2][:,np.newaxis]
    data_return = np.concatenate((eis_real,eis_img),axis=1).reshape(1,-1)

    return data_return

def generate_y(data_name):
    if 'BD' in data_name:
        y = DataFrame(np.zeros(shape = (1,1)))
    else:
        y = DataFrame(np.ones(shape = (1,1)))
    return y

def generate_name(data_name):
    return np.array(data_name.rsplit('\\',-1)[-1].rsplit('.csv',-1)[0]).reshape(1,1)

def generate_train_test_data(train_number, validation_number, test_number, data_dir):
    train_data = DataFrame(np.zeros(shape = (1,10)))
    validation_data = DataFrame(np.zeros(shape=(1, 10)))
    test_data = DataFrame(np.zeros(shape=(1, 10)))
    train_y = DataFrame(np.zeros(shape = (1,1)))
    validation_y = DataFrame(np.zeros(shape = (1,1)))
    test_y = DataFrame(np.zeros(shape = (1,1)))
    train_name = DataFrame(np.zeros(shape = (1,1)))
    validation_name = DataFrame(np.zeros(shape = (1,1)))
    test_name = DataFrame(np.zeros(shape = (1,1)))

    for i in range(len(train_number)):
        train_dataset = np.array(pd.read_csv(data_dir[train_number[i]],sep='\t',header=None))
        train_dataset = deal_data(train_dataset)
        train_data = pd.concat([train_data,DataFrame(train_dataset)],axis=0,ignore_index=True)
        y = generate_y(data_dir[train_number[i]])
        train_y = pd.concat([train_y,y],axis=0,ignore_index=True)
        name = generate_name(data_dir[train_number[i]])
        train_name = pd.concat([train_name,DataFrame(name)],axis=0,ignore_index=True)

    for i in range(len(validation_number)):
        validation_dataset = np.array(pd.read_csv(data_dir[validation_number[i]],sep='\t',header=None))
        validation_dataset = deal_data(validation_dataset)
        validation_data = pd.concat([validation_data,DataFrame(validation_dataset)],axis=0,ignore_index=True)
        y = generate_y(data_dir[validation_number[i]])
        validation_y = pd.concat([validation_y,y],axis=0,ignore_index=True)
        name = generate_name(data_dir[validation_number[i]])
        validation_name = pd.concat([validation_name, DataFrame(name)], axis=0, ignore_index=True)

    for i in range(len(test_number)):
        test_dataset = np.array(pd.read_csv(data_dir[test_number[i]],sep='\t',header=None))
        test_dataset = deal_data(test_dataset)
        test_data = pd.concat([test_data,DataFrame(test_dataset)],axis=0,ignore_index=True)
        y = generate_y(data_dir[test_number[i]])
        test_y = pd.concat([test_y, y], axis=0, ignore_index=True)
        name = generate_name(data_dir[test_number[i]])
        test_name = pd.concat([test_name, DataFrame(name)], axis=0, ignore_index=True)

    return np.array(train_data.iloc[1:,:]), np.array(validation_data.iloc[1:,:]), np.array(test_data.iloc[1:,:]),\
           np.array(train_y.iloc[1:,:]).reshape(-1,),np.array(validation_y.iloc[1:,:]).reshape(-1,),np.array(test_y.iloc[1:,:]).reshape(-1,),\
           np.array(train_name.iloc[1:,:]),np.array(validation_name.iloc[1:,:]),np.array(test_name.iloc[1:,:])


def norm(x_train,x_validation,x_test):
    x_train = x_train.reshape(-1,51,2)
    x_validation = x_validation.reshape(-1,51,2)
    x_test = x_test.reshape(-1,51,2)

    x_train_0 = x_train[:,:,0].reshape(-1,1)
    x_train_1 = x_train[:,:,1].reshape(-1,1)
    x_validation_0 = x_validation[:,:,0].reshape(-1,1)
    x_validation_1 = x_validation[:,:,1].reshape(-1,1)
    x_test_0 = x_test[:,:,0].reshape(-1,1)
    x_test_1 = x_test[:,:,1].reshape(-1,1)

    train_0_mean = np.mean(x_train_0,axis=0)
    train_0_std = np.std(x_train_0,axis=0)
    train_1_mean = np.mean(x_train_1,axis=0)
    train_1_std = np.std(x_train_1,axis=0)
    norm_x_train_0 = ((x_train_0 - train_0_mean)/train_0_std).reshape(-1,51,1)
    norm_x_train_1 = ((x_train_1 - train_1_mean)/train_1_std).reshape(-1,51,1)
    norm_x_validation_0 = ((x_validation_0 - train_0_mean)/train_0_std).reshape(-1,51,1)
    norm_x_validation_1 = ((x_validation_1 - train_1_mean)/train_1_std).reshape(-1,51,1)
    norm_x_test_0 = ((x_test_0 - train_0_mean)/train_0_std).reshape(-1,51,1)
    norm_x_test_1 = ((x_test_1 - train_1_mean)/train_1_std).reshape(-1,51,1)

    norm_x_train = np.concatenate((norm_x_train_0,norm_x_train_1),axis=2)
    norm_x_validation = np.concatenate((norm_x_validation_0,norm_x_validation_1),axis=2)
    norm_x_test = np.concatenate((norm_x_test_0,norm_x_test_1),axis=2)

    return norm_x_train,norm_x_validation,norm_x_test


path = r'EIS_total_dataset'
data_dir = get_all_data(path)
shuffle_index = shuffle_data(data_dir)
train_number, validation_number, test_number = generate_train_test_number(shuffle_index)

x_train, x_validation, x_test, y_train, y_validation, y_test, train_name, validation_name, test_name = generate_train_test_data(train_number, validation_number, test_number, data_dir)

print(x_train.shape, y_train.shape, x_validation.shape, y_validation.shape, x_test.shape, y_test.shape,train_name.shape, validation_name.shape, test_name.shape)
norm_x_train,norm_x_validation,norm_x_test = norm(x_train,x_validation,x_test)
print(norm_x_train.shape,norm_x_validation.shape,norm_x_test.shape,y_train.shape,y_validation.shape,y_test.shape)

learing_rate = 0.0005
optimizer = tf.keras.optimizers.Adam(learing_rate, beta_1=0.9,
                                     beta_2=0.98, epsilon=1e-9)

model = keras.Sequential([
    layers.Flatten(input_shape=(51,2)),
    layers.Dense(32, activation='relu',kernel_regularizer = regularizers.l2(0.003)),
    layers.Dense(16, activation='relu',kernel_regularizer = regularizers.l2(0.003)),
    layers.Dense(16, activation='relu',kernel_regularizer = regularizers.l2(0.003)),
    layers.Dense(8, activation='relu',kernel_regularizer = regularizers.l2(0.003)),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer=optimizer,
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])
model.summary()

EPOCHS = 200
checkpoint_path = './checkpoint/EIS_diagnose_975/cp-{epoch:04d}.ckpt'
checkpoint_dir = os.path.dirname(checkpoint_path)
latest = tf.train.latest_checkpoint(checkpoint_dir)
model.load_weights(latest)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 save_best_only=True,
                                                 verbose=1)

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=500)

history = model.fit(norm_x_train, y_train, batch_size=512, epochs=EPOCHS,
                    validation_data=(norm_x_validation, y_validation), verbose=1,
                    callbacks=[early_stop, cp_callback],
                    use_multiprocessing=True)
loss, accuracy = model.evaluate(norm_x_test, y_test, batch_size=32, verbose=1)


y_pre_train = model.predict(
    norm_x_train, batch_size=32, verbose=0, steps=None, callbacks=None, max_queue_size=10,
    workers=1, use_multiprocessing=False)


y_pre_validation = model.predict(
    norm_x_validation, batch_size=32, verbose=0, steps=None, callbacks=None, max_queue_size=10,
    workers=1, use_multiprocessing=False)


y_pre_test = model.predict(
    norm_x_test, batch_size=32, verbose=0, steps=None, callbacks=None, max_queue_size=10,
    workers=1, use_multiprocessing=False)

