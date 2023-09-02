import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
from dl.net import TA_keras
from ensemble1.model_factory import get_new_model
from prepare.prepare_ml import  xgb_feature_selection
import os

from colorama import Fore
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from xgboost import XGBClassifier
from tools.tools import score2

from config.config import Config
import sys
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.python.keras.layers import Dropout, Dense, Flatten, Activation, Conv1D, LeakyReLU, Bidirectional, LSTM
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.optimizer_v2.gradient_descent import SGD
from tools.tools import split_seq_df, get_label
import random
import numpy as np
import pandas as pd
import os
from random import seed
import tensorflow as tf
sys.setrecursionlimit(15000)
import numpy as np
import pandas as pd


seq, label = get_fasta('../DeepDNA4mC/A.thaliana.fasta')





data_train1, label_train, record_feature_type = .....
best_features = xgb_feature_selection(data_train1, label_train)
best_features_name = [record_feature_type[i] for i in best_features]
best_features_type = dict(Counter(best_features_name))

n_folds = 5
base_models_name = ["CNN1", "DNN1", "LSTM1"]
base_models = []

base_models_5fold_prediction = np.zeros((data_train1.shape[0], len(base_models_name)+1))
skf = list(StratifiedKFold(n_splits=n_folds).split(data_train1, label_train))
X = data_train1[:,best_features]
y = label_train


for j, model_name in enumerate(base_models_name):
    single_type_model = []
    for i, (train_index, val_index) in enumerate(skf):
        X_train, y_train, X_val, y_val = X[train_index], y[train_index], X[val_index], y[val_index]
        model = get_new_model(model_name, len(X_train[0]))
        model.fit(X_train, y_train)
        y_pred = model.predict_proba(X_val)[:, 1]
        base_models_5fold_prediction[val_index, j] = y_pred
        single_type_model.append(model)
    base_models.append(single_type_model)



config_global = Config()
config_global.model_name = ""
config_global.is_feature_selection = True  
config_global.load_global_pretrain_model = True  
config_global.global_model_save_path = "data/pretrain_model.h5"

XX = get_fasta('data/A.thaliana.fasta')


def create_model(config):
    print('building model...............')
    model = Sequential()
    params = {
        "filters_1": 250,
        "kernel_size_1": 2,
        "filters_2": 100,
        "kernel_size_2": 2,
        "filters_3": 100,
        "kernel_size_3": 3,
        "filters_4": 250,
        "kernel_size_4": 2,
        "filters_5": 250,
        "kernel_size_5": 10,
        "dense_1": 320
    }

    #************************* convolutional layer  ***********************
    model.add(Conv1D(
                            input_shape=(41,4),
                            filters=params["filters_1"],
                            kernel_size=params["kernel_size_1"],
                            padding="valid",
                            activation="linear",
                            strides=1,
                            #W_regularizer = l2(0.01),
                            kernel_initializer='he_normal',
                            name="cov1"))
    model.add(LeakyReLU(alpha=.001))
    model.add(Dropout(0.2))


    model.add(Conv1D(filters=params["filters_2"],
                            kernel_size=params["kernel_size_2"],
                            padding="valid",
                            activation="linear",
                            strides=1, kernel_initializer='he_normal',
                            name = "cov2"))
    model.add(LeakyReLU(alpha=.001))
    model.add(Dropout(0.2))



    model.add(Conv1D(filters=params["filters_3"],
                            kernel_size=params["kernel_size_3"],
                            padding="valid",
                            activation="linear",
                            strides=1, kernel_initializer='he_normal',
                            name="cov3"))
    model.add(LeakyReLU(alpha=.001))
    model.add(Dropout(0.2))


    model.add(Conv1D(filters=params["filters_4"],
                            kernel_size=params["kernel_size_4"],
                            padding="valid",
                            activation="linear",
                            strides=1, kernel_initializer='he_normal',
                            name = "cov4"))
    model.add(LeakyReLU(alpha=.001))
    model.add(Dropout(0.2))



    model.add(Conv1D(filters=params["filters_5"],
                            kernel_size=params["kernel_size_5"],
                            padding="valid",
                            activation="linear",
                            strides=1,
                            kernel_initializer='he_normal',
                            name = "cov5"))
    model.add(LeakyReLU(alpha=.001))
    model.add(Dropout(0.5))


    model.add(Flatten())
    model.add(Dense(units=params["dense_1"], kernel_initializer='he_normal'))
    model.add(LeakyReLU(alpha=.001))
    model.add(Dropout(0.5))
    model.add(Dense(units=1))
    model.add(Activation('sigmoid'))
    return model


def split_seq_df(data,freq=0.2):
    labels = get_label(data)
    labels_index = np.array(range(0, data.shape[0]))
    labels_p = [(True if value == "1" else False) for value in labels]
    labels_n = [(True if value == "0" else False) for value in labels]
    labels_p = set(labels_index[labels_p])
    labels_n = set(labels_index[labels_n])
    labels_p_test = set(random.sample(labels_p, int(len(labels_p) * freq)))
    labels_n_test = set(random.sample(labels_n, int(len(labels_n) * freq)))
    labels_p_train = labels_p.difference(labels_p_test)
    labels_n_train = labels_n.difference(labels_n_test)
    labels_train = list(labels_p_train.union(labels_n_train))
    labels_test = list(labels_p_test.union(labels_n_test))

    random.shuffle(labels_train)
    random.shuffle(labels_test)

    data_train = data.iloc[labels_train,:]
    data_test = data.iloc[labels_test, :]
    return data_train, data_test
def get_label(seqs_df):
    label = seqs_df["label"].values
    return label

class TA_keras():
    def __init__(self,config):
        self.config = config
        self.best_model = None
        self.temp_model = None
        self.global_model_save_path = config.global_model_save_path
        self.bestmodel_path = config.model_save_path+config.model_name+"_.h5"

    def fit(self,X,y,X_train=None,X_dev=None):
        if X_train is None:
            X_train, X_dev = split_seq_df(X)
        print(X_dev)
        y_train = get_label(X_train)
        y_dev = get_label(X_dev)
        
        X_train = one_hot(X_train.iloc[:,1].values)
        X_dev = one_hot(X_dev.iloc[:,1].values)

        if self.config.load_global_pretrain_model:
            print("loading global pretrain model ...")
            model = load_model(self.config.global_model_save_path)
        else:
            model = create_model(self.config)

        sgd = SGD(lr=self.config.learning_rate, momentum=0.9, decay=1e-6, nesterov=True)
        checkpointer = ModelCheckpoint(filepath=self.bestmodel_path, verbose=0, save_best_only=True)
        earlystopper = EarlyStopping(monitor='val_loss', patience=self.config.patience, verbose=0)
        model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

        y_train = tf.strings.to_number(y_train, out_type=tf.float32)
        y_dev = tf.strings.to_number(y_dev, out_type=tf.float32)
        model.fit(X_train,
                  y_train,
                  batch_size=self.config.batch_size,
                  epochs=self.config.num_epochs,
                  shuffle=True,
                  verbose=0,
                  validation_data=(X_dev, y_dev),
                  callbacks=[checkpointer, earlystopper])
        self.best_model = load_model(self.bestmodel_path)
        print('training done!')

    def predict_proba(self,X):
        model = self.best_model
        X = one_hot(X.iloc[:, 1].values)
        pred_prob_test = model.predict(X, verbose=0)
        pred_prob_test = np.squeeze(pred_prob_test)
        return pred_prob_test
XX = get_fasta('data/A.thaliana.fasta')

single_type_model = []
meta_model = LogisticRegression()
for i, (train_index, val_index) in enumerate(skf):
    X_train, y_train, X_val, y_val = XX.iloc[train_index, :], y[train_index], XX.iloc[val_index, :], y[val_index]
    ta = TA_keras(config_global)
    ta.fit(X_train, y_train)
    y_pred = ta.predict_proba(X_val)
    single_type_model.append(ta) 
    base_models_5fold_prediction[val_index, len(base_models_name)] = y_pred
base_models.append(single_type_model)


res = 0
coeff = []
y_trued = np.zeros((data_train1.shape[0],))


x_mate = np.zeros((data_train1.shape[0],))
for i, (train_index, val_index) in enumerate(skf):
    X_train, y_train, X_val, y_val = base_models_5fold_prediction[train_index], y[train_index], base_models_5fold_prediction[val_index], y[val_index]
    
    meta_model = LogisticRegression() 
    meta_model.fit(X_train, y_train)
    pred_prob = meta_model.predict(X_val)
    y_trued[val_index] = y_val
    x_mate[val_index] = meta_model.predict_proba(X_val)[:,1]
    result_temp = score2(y_val, pred_prob)
    print(result_temp)
    res += meta_model.score(X_val, y_val)
print(res/5)

