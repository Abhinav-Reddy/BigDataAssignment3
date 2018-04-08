from time import time
import pandas as pd
import numpy as np
from gensim.models import KeyedVectors
import re
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

import itertools
import datetime

from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Dense
import keras.backend as K
from keras.optimizers import Adadelta, SGD
from keras.callbacks import ModelCheckpoint
import re
import urllib
import os

import numpy as np
import tensorflow as tf
import pandas as pd

#TRAIN_CSV = "../raw_data/train.csv"
TRAIN_CSV = "~/fulltest/test.csv"
TEST_CSV = "../raw_data/test.csv"
COMPUTE_DATA_PATH = "../computed_data/"
MODELS_PATH = "../models/"
RESULTS_PATH = "../results/"

#LOADS TRAINING AND TEST SET
train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)
seqLength = 13
dict = {}
dict['A'] = 0
dict['C'] = 1
dict['G'] = 2
dict['T'] = 3


embedding_dim = 16
vocabulary = 16
embeddings = np.zeros((vocabulary + 1, embedding_dim))
embeddings.fill(0)
for i in range(vocabulary):
	embeddings[i+1][i] = 1
embeddings[0][0] = 0

for dataTuple in [train_df, test_df]:
	for index, row in dataTuple.iterrows():
		numVector = []
		curStr = row['sequence']
		for i in range(len(curStr)-1):
			curVal = 0
			#if (curStr[i] != curStr[i+1]):
			for j in range(2):
				curVal = (4**j)*curVal + dict[curStr[i+j]]
			numVector.append(curVal+1)
		dataTuple.set_value(index, 'sequence', numVector)

validation_size = 100
xTrain, xValidation, yTrain, yValidation = train_test_split(train_df['sequence'], train_df['label'], test_size=validation_size)


xValidation = np.array(xValidation.tolist())
xTrain = np.array(xTrain.tolist())
print(xValidation)
# Model variables
n_hidden = 16
gradientClippingNorm = 1.25
batch_size = 4
n_epoch = 3

inputSeq = Input(shape=(seqLength,), dtype='int32')
embeddingLayer = Embedding(len(embeddings), embedding_dim, weights=[embeddings], input_length=seqLength, trainable=False)
encodedSeq = embeddingLayer(inputSeq)
lstm = LSTM(n_hidden)
output = lstm(encodedSeq)
output = Dense(1, activation = 'sigmoid')(output)
Predictor = Model([inputSeq], [output])
optimizer = Adadelta(clipnorm=gradientClippingNorm)
#optimizer = SGD(lr=0.01)
Predictor.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])
training_start_time = time()

print("Started training")
PredictorTrained = Predictor.fit(xTrain, yTrain.values, batch_size=batch_size, epochs=10, validation_data=(xValidation, yValidation.values))
	

print("Training time finished.\n{} epochs in {}".format(n_epoch, datetime.timedelta(seconds=time()-training_start_time)))

xTest = np.array(test_df['sequence'].tolist())
predictions = Predictor.predict(xTest)

import pandas as pdn
sub_df = pd.DataFrame(data=predictions,columns={"prediction"})
sub_df.to_csv(path_or_buf="../results/sub.csv", columns={"prediction"}, header=True, index=True, index_label="id")

			
