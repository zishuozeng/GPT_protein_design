from transformers import TFT5EncoderModel, T5Tokenizer
import re
import gc
import pandas as pd
import numpy as np
import pickle
import keras
import random
from scipy.stats import poisson

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

#############################################################################
########### evaluate random, ProtGPT2-random, ProtGPT2-finetuned  ###########
#############################################################################
protgpt2_random = pd.read_csv('protgpt2_shorts.csv')
protgpt2_finetuned = pd.read_csv('amp_gen_seqs.csv')

protgpt2_random['group'] = ['ProtGPT2-random'] * protgpt2_random.shape[0]
protgpt2_finetuned['group'] = ['ProtGPT2-finetuned'] * protgpt2_finetuned.shape[0]

amp_pred = pd.concat([protgpt2_random, protgpt2_finetuned], axis = 0)
amp_pred['length'] = [len(i) for i in list(amp_pred.seq)]
amp_pred = amp_pred[amp_pred['length'] < 51] #[5167 rows x 3 columns]

seq_embedded = preproc_seq(list(amp_pred.seq), 20, 51) #

model = keras.models.load_model('model')

pred = model.predict(seq_embedded)[:,0]

amp_pred['prediction'] = pred


### add random peptides ###
'''
summarize aa distribution of real AMPs as sampling weights for random peptide generation
'''

aa_compo = pd.read_csv('aa_compo.csv')
aas = list(aa_compo.aa)

def read_fasta(input_file):
    f = open(input_file, 'r')
    all_text = f.read()
    entries = all_text.split('>')[1:]
    headers = []
    sequences = []
    for i in entries:
        lines = i.split('\n')
        headers.append(lines[0])
        seq = ''
        for j in lines[1:]:
            seq = seq + j
        sequences.append(seq)
    return headers, sequences

headers1, amp = read_fasta('SATPdb_amp.fa') #3011

big_seq = ''
for i in amp:
    big_seq += i

aas = list(aas)
sampling_weights = []
for i in aas:
    sampling_weights.append(big_seq.count(i))

'''
generate 1000 random peptides as control
'''#length follows a poisson distribution with mean being real AMPs' mean
random_peptides = []
lengths = []
for i in range(1000):
    n = poisson.rvs(mu=round(np.mean(amp_pred.length)), size=1)[0]
    selected = random.choices(aas, weights = sampling_weights, k=n)
    concatenated = ''.join(selected)
    random_peptides.append(concatenated)
    lengths.append(n)

random_peptides = pd.DataFrame({'seq':random_peptides, 'group': ['random'] * 1000, 'length':lengths})
random_peptides

'''
predict AMP-likelihood of the 1000 random peptides
'''
seq_embedded = preproc_seq(list(random_peptides.seq), 20, 51) #
model = keras.models.load_model('model')
pred = model.predict(seq_embedded)[:,0]
random_peptides['prediction'] = pred

###add to amp_pred
amp_pred = pd.concat([random_peptides, amp_pred], axis = 0) #[6167 rows x 4 columns]
amp_pred.to_csv('amp_pred.csv', index = False)


