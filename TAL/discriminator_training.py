import pandas as pd
import numpy as np
import pickle, gc, time
from transformers import TFT5EncoderModel, T5Tokenizer
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
import numpy as np
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping

######################
### data embedding ###
######################
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

train = pd.read_csv('train.csv').sample(frac=1, random_state=123).reset_index(drop=True)
test = pd.read_csv('test.csv').sample(frac=1, random_state=123).reset_index(drop=True)

X_train = list(train.seq)
y_train = list(train.label)
X_test = list(test.seq)
y_test = list(test.label)

# pickle.dump(X_train, open('X_train', 'wb'))
# pickle.dump(y_train, open('y_train', 'wb'))
# pickle.dump(X_test, open('X_test', 'wb'))
# pickle.dump(y_test, open('y_test', 'wb'))

# del train, test

t = time.time()
X_train_embedded = preproc_seq(X_train, 10, 1000) #max length is 2134; setting max=1000 covers 985/1000 seqs
time.time() -t #275.85
# pickle.dump(X_train_embedded, open('X_train_embedded', 'wb'))
# del X_train_embedded

t = time.time()
X_test_embedded = preproc_seq(X_test, 10, 1000)
time.time() -t #11
# pickle.dump(X_test_embedded, open('X_test_embedded', 'wb'))
# del X_test_embedded

##############################
### discriminator training ###
##############################
model = Sequential()
model.add(Conv2D(3, kernel_size=3, activation='relu', input_shape=(1000, 1024, 1)))
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

model.fit(X, np.array(y), batch_size=128, epochs=10, validation_split=0.1, callbacks=[callback])

model.save('model')

y_test = pickle.load(open('y_test', 'rb'))
X_test = pickle.load(open('X_test_embedded', 'rb')) #

score = model.evaluate(X_test, np.array(y_test), verbose=1) #loss: 0.1452 - accuracy: 0.9700



