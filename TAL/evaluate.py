import pandas as pd
import numpy as np
import keras
from transformers import TFT5EncoderModel, T5Tokenizer
import gc, time, random, pickle

tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False)
embedder = TFT5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50", from_pt=True)
gc.collect()

model = keras.models.load_model('discriminator')

##############################################################
### evaluate (make predictions) on the generated sequences ###
##############################################################
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

def evaluate_gen(DF, chunk_SIZE, MAX_LEN, MODEL):
    SEQ = list(DF.seq)
    preproc = preproc_seq(SEQ, chunk_SIZE, MAX_LEN)
    print('#################### DONE PREPROCESSING ###################')
    pred = list(MODEL.predict(preproc)[:, 0])
    print('#################### DONE PREDICTING ###################')
    DF['prediction'] = pred
    return DF

gen = pd.read_csv('gen.csv')
gen_with_pred = evaluate_gen(gen, 10, 1000, model)

gen_with_pred.to_csv('gen.csv', index=False)



##########################################################################
### evaluate the discrminator resolution on functionally close enzymes ###
##########################################################################
func_close = pd.read_csv('func_close.csv').sample(frac=1, random_state=123).reset_index(drop=True)
X_func_close = list(func_close.seq)
X_func_close_embedded = preproc_seq(X_func_close, 10, 1000)
y_predicted = model.predict(X_func_close_embedded)[:, 0]
func_close['prediction'] = y_predicted # <=========================================working here

func_close.to_csv('func_close.csv', index=False)

y_test = pickle.load(open('y_test', 'rb'))
X_test = pickle.load(open('X_test_embedded', 'rb')) #
y_test_predicted = model.predict(X_test)[:, 0]

test_pos = pd.DataFrame({'label':y_test, 'prediction':y_test_predicted})
test_pos = test_pos[test_pos['label'] == 1]
test_pos['ec3'] = ['1'] * test_pos.shape[0]

func_close_eval = func_close[['ec3', 'prediction']]
func_close_eval['label'] = [0]*func_close_eval.shape[0]
func_close_eval = func_close_eval[['label', 'prediction', 'ec3']]

func_close_eval = pd.concat([test_pos, func_close_eval], axis=0)

func_close_eval.to_csv('func_close.csv', index=False)

