###############################################
######### MDH discriminator training ##########
###############################################
from transformers import TFT5EncoderModel, T5Tokenizer
import gc
import pandas as pd
import numpy as np
import pickle

tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False)
embedder = TFT5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50", from_pt=True)
gc.collect()

def chunk_array(ARRAY, chunk_SIZE):
    #split the input array by batch (chunks)
    chunk_SIZE = max(1, chunk_SIZE)
    return [ARRAY[i:i + chunk_SIZE, :] for i in range(0, ARRAY.shape[0], chunk_SIZE)]


def preproc_seq(SEQ, chunk_SIZE, MAX_LEN):
    '''
    :param SEQ: a list of sequences to be embedded
    :param chunk_SIZE: embed the sequences by batch; batch size
    :param MAX_LEN: maximum length set for embedding
    :return: arrays of the embedded sequences
    '''
    sequences1 = []
    for i in SEQ:
        splits = [j for j in i]
        tmp = ' '.join(splits)
        sequences1.append(tmp)
    ids = tokenizer.batch_encode_plus(sequences1, max_length = MAX_LEN, add_special_tokens=True,
                                      padding='max_length', truncation=True, return_tensors="tf")
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


X_train = pickle.load(open('X_train', 'rb'))
y_train = pickle.load(open('y_train', 'rb'))

X_test = pickle.load(open('X_test', 'rb'))
y_test = pickle.load(open('y_test', 'rb'))


########### data embedding ###########
'''
Note that embedding these 33k sequences takes a long time (several hours),
thus we recommend submitting the job to a server
'''
X_train_embedded = preproc_seq(X_train, 10, 505)
X_test_embedded = preproc_seq(X_test, 10, 505)


########### discriminator training ##############
import pandas as pd
import numpy as np
import pickle
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
import numpy as np
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping

model = Sequential()
model.add(Conv2D(3, kernel_size=3, activation='relu', input_shape=(505, 1024, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(3, kernel_size=3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(3, kernel_size=3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())

model.add(Dense(1024, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
callback = EarlyStopping(monitor='loss', patience=5)

model.fit(x1, np.array(y1), batch_size=128, epochs=10, validation_split=0.1, callbacks=[callback])  #

#please specify path to MDH discriminator
model.save(PATH/TO/DISCRIMINATOR)

### testing
#please specify path to MDH discriminator
model = keras.models.load_model(PATH/TO/DISCRIMINATOR)
score = model.evaluate(X_test_embedded, np.array(y_test), verbose=1) #loss: 0.0154 - accuracy: 0.9946



