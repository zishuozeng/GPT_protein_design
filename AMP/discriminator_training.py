from transformers import TFT5EncoderModel, T5Tokenizer
import tensorflow as tf
import numpy as np
import re
import gc
import pandas as pd
import pickle

##########################
##### data embedding #####
##########################
tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False)
embedder = TFT5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50", from_pt=True)

gc.collect()

def chunk_array(ARRAY, chunk_SIZE):
    chunk_SIZE = max(1, chunk_SIZE)
    return [ARRAY[i:i + chunk_SIZE, :] for i in range(0, ARRAY.shape[0], chunk_SIZE)]

def preproc_seq(SEQ, chunk_SIZE, MAX_LEN):
    sequences1 = []
    for i in SEQ:
        splits = [j for j in i]
        tmp = ' '.join(splits)
        sequences1.append(tmp)
    ids = tokenizer.batch_encode_plus(sequences1, max_length = MAX_LEN, add_special_tokens=True,
                                      padding=True, return_tensors="tf")
    input_ids = ids['input_ids']
    input_id_chunks = chunk_array(input_ids, chunk_SIZE)
    features = embedder(input_id_chunks[0])
    features = np.asarray(features.last_hidden_state)
    k = 0
    for i in input_id_chunks[1:]:
        features0 = embedder(i)
        features0 = np.asarray(features0.last_hidden_state)
        features = np.concatenate([features, features0], axis=0)
        k += 1
        print('done', k, 'batch')
    # attention_mask = ids['attention_mask']
    # attention_mask = np.asarray(attention_mask)
    return features

X_train_embedded = preproc_seq(X_train, 20, 50) #(5187, 51, 1024)
X_test_embedded = preproc_seq(X_test, 20, 50) #(571, 51, 1024)

# pickle.dump(X_train_embedded, open('X_train_embedded', 'wb'))
# pickle.dump(X_test_embedded, open('X_test_embedded', 'wb'))

###############################
##### train discriminator #####
###############################
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
import pickle
import numpy as np
from keras.callbacks import EarlyStopping

X_train_preproc = pickle.load(open('X_train_embedded', 'rb'))
y_train = pickle.load(open('y_train', 'rb'))
X_test_preproc = pickle.load(open('X_test_embedded', 'rb'))
y_test = pickle.load(open('y_test', 'rb'))

def tune_model(n_filter, n_conv_maxpool, ks, ps, n_dense_layers, n_dense_nodes, lr, batch_size):
    model = Sequential()
    model.add(Conv2D(n_filter, kernel_size=ks, activation='relu', input_shape=(51, 1024, 1)))
    model.add(MaxPooling2D(pool_size=(ps, ps)))
    for i in range(n_conv_maxpool - 1):
        model.add(Conv2D(n_filter, kernel_size=ks, activation='relu'))
        model.add(MaxPooling2D(pool_size=(ps, ps)))
    model.add(Flatten())
    model.add(Dense(n_dense_nodes, activation='relu'))
    for i in range(n_dense_layers - 1):
        model.add(Dense(n_dense_nodes, activation='relu'))
        model.add(Dense(n_dense_nodes, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    optimizer = keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    callback = EarlyStopping(monitor='loss', patience=5)
    model.fit(X_train_preproc[:4687, :, :], np.array(y_train[:4687]), batch_size=batch_size, epochs=100, callbacks=[callback],
              verbose = 0)  #
    res = model.evaluate(X_train_preproc[4687:, :, :], np.array(y_train[4687:]), verbose=0)[1]
    return res

#can tune the hyperparameters using the tune_model() function
tune_model(n_filter=3, n_conv_maxpool=2, ks=3, ps=2, n_dense_layers=3, n_dense_nodes=1024, lr=0.001, batch_size=2000)


model = Sequential()
model.add(Conv2D(3, kernel_size=3, activation='relu', input_shape=(51, 1024, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(3, kernel_size=3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Conv2D(3, kernel_size=3, activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())

model.add(Dense(512, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

optimizer = keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer,
              loss='binary_crossentropy',
              metrics=['accuracy'])

callback = EarlyStopping(monitor='loss', patience=6)
model.fit(X_train_preproc, np.array(y_train), batch_size=2000, epochs=75, callbacks=[callback], validation_split = 0.1)  #

model.save('model')

#performance on testing set
model.evaluate(X_test_preproc, np.array(y_test), verbose=1)  # [0.4296610653400421, 0.8091068267822266]
