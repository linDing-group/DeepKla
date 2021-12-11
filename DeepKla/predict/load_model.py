#!/usr/bin/env python
# _*_coding:utf-8_*_

from __future__ import print_function
#import _pickle as cPickle
import cPickle
import numpy as np
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU, SimpleRNN
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.datasets import imdb
from keras.callbacks import EarlyStopping,ModelCheckpoint
from sklearn.model_selection import StratifiedKFold
from keras.engine.topology import Layer, InputSpec
import h5py
from keras.models import model_from_json
from keras.models import load_model
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from keras import initializers
from sklearn import metrics
from keras import backend as K
from keras.layers import *
from keras.models import *
from keras.optimizers import Adam
from keras.regularizers import l1, l2
from keras import initializers, regularizers, constraints
import sys, os
from SEL_CNN_BiGRU_Attention import get_model

def trans(str1):
    a = []
    dic = {'A':1,'B':22,'U':23,'J':24,'Z':25,'O':26,'C':2,'D':3,'E':4,'F':5,'G':6,'H':7,'I':8,'K':9,'L':10,'M':11,'N':12,'P':13,'Q':14,'R':15,'S':16,'T':17,'V':18,'W':19,'Y':20,'X':21}
    for i in range(len(str1)):
        a.append(dic.get(str1[i]))
    return a

def createTrainData(str1):
    sequence_num = []
    label_num = []
    name = []
    f = open(str1).readlines()
    label = 0
    for i in range(0,len(f)-1,2):
        #label = f[i].strip('\n').replace('>','')
        name.append(f[i].strip('\n'))
        label_num.append(label)
        label += 1
        sequence = f[i+1].strip('\n')
        sequence_num.append(trans(sequence))

    return sequence_num,label_num,name

def createTrainTestData(str_path, nb_words=None, skip_top=0,
              maxlen=None, test_split=0.25, seed=113,
              start_char=1, oov_char=2, index_from=3):
    X,labels = cPickle.load(open(str_path, "rb"))

    # np.random.seed(seed)
    # np.random.shuffle(X)
    # np.random.seed(seed)
    # np.random.shuffle(labels)
    if start_char is not None:
        X = [[start_char] + [w + index_from for w in x] for x in X]
    elif index_from:
        X = [[w + index_from for w in x] for x in X]

    if maxlen:
        new_X = []
        new_labels = []
        for x, y in zip(X, labels):
            if len(x) < maxlen:
                new_X.append(x)
                new_labels.append(y)
        X = new_X
        labels = new_labels
    if not X:
        raise Exception('After filtering for sequences shorter than maxlen=' +
                        str(maxlen) + ', no sequence was kept. '
                                      'Increase maxlen.')
    if not nb_words:
        nb_words = max([max(x) for x in X])


    if oov_char is not None:
        X = [[oov_char if (w >= nb_words or w < skip_top) else w for w in x] for x in X]
    else:
        nX = []
        for x in X:
            nx = []
            for w in x:
                if (w >= nb_words or w < skip_top):
                    nx.append(w)
            nX.append(nx)
        X = nX

    X_train = np.array(X[:int(len(X) * (1 - test_split))])
    y_train = np.array(labels[:int(len(X) * (1 - test_split))])

    X_test = np.array(X[int(len(X) * (1 - test_split)):])
    y_test = np.array(labels[int(len(X) * (1 - test_split)):])

    return (X_train, y_train), (X_test, y_test)



fastafile = sys.argv[1]
#model = sys.argv[2]
result = sys.argv[2]
pklfile = sys.argv[3]

a,b,name = createTrainData(fastafile)
t = (a, b)

cPickle.dump(t,open(pklfile,"wb"))
max_features = 23
maxlen = 200
(X_train, y_train), (X_test, y_test) = createTrainTestData(pklfile,nb_words=max_features, test_split=0)

X = sequence.pad_sequences(X_train, maxlen=maxlen)
model=None
model=get_model()
model.load_weights("51bp_rusuan.h5")


loss, acc = model.evaluate(X, y_train)
####################################################
g = open(result,'w')
pre_test_y = model.predict(X)# 输出的是预测概率
for i in range(len(name)):
    g.write(name[i]+'\t'+str(pre_test_y[i][0])+'\n')
g.close()
