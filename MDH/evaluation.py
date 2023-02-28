import pandas as pd
import numpy as np
import keras
from transformers import TFT5EncoderModel, T5Tokenizer
import gc, time, random, pickle

tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False)
embedder = TFT5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50", from_pt=True)
gc.collect()

#please specify path to MDH discriminator
model = keras.models.load_model(PATH/TO/DISCRIMINATOR)

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

def evaluate_gen(DF, chunk_SIZE, MAX_LEN, MODEL):
    '''
    :param DF: the input dataframe containing sequences (column 'seq') 
    :param chunk_SIZE: embed the sequences by batch; batch size
    :param MAX_LEN: aximum length set for embedding
    :param MODEL: discriminator keras model object
    :return: 
    '''
    SEQ = list(DF.seq)
    preproc = preproc_seq(SEQ, chunk_SIZE, MAX_LEN)
    print('#################### DONE PREPROCESSING ###################')
    pred = list(MODEL.predict(preproc)[:, 0])
    print('#################### DONE PREDICTING ###################')
    DF['prediction'] = pred
    return DF

# gen_noft = pd.read_csv('gen_noft.csv')
# gen765 = pd.read_csv('gen765.csv')
# gen2029 = pd.read_csv('gen2029.csv')
# gen5987 = pd.read_csv('gen5987.csv')
# gen7477 = pd.read_csv('gen7477.csv')
#
# gen_noft_pred = evaluate_gen(gen_noft, 100, 505, model)
# gen765_pred = evaluate_gen(gen765, 100, 505, model)
# gen2029_pred = evaluate_gen(gen2029, 100, 505, model)
# gen5987_pred = evaluate_gen(gen5987, 100, 505, model)
# gen7477_pred = evaluate_gen(gen7477, 100, 505, model)
#
# gen_noft_pred.to_csv('gen_noft.csv', index=False)
# gen765_pred.to_csv('gen765.csv', index=False)
# gen2029_pred.to_csv('gen2029.csv', index=False)
# gen5987_pred.to_csv('gen5987.csv', index=False)
# gen7477_pred.to_csv('gen7477.csv', index=False)

all_gen = pd.read_csv('all_gen.csv')
all_gen = evaluate_gen(all_gen, 100, 505, model)

###################################################################################################
###### also make predictions for EC1.1.1.x and EC1.1.x.x. and EC1.1.1.36 (validation set) #########
###################################################################################################
mdh_close_ec_test = pd.read_csv('mdh_close_ec_test.csv')
mdh_close_ec_test_pred = evaluate_gen(mdh_close_ec_test, 100, 505, model)
mdh_close_ec_test.to_csv('mdh_close_ec_test.csv', index=False)

##################################################
######## generate & evaluate random seqs ########
##################################################
#load aa composition in natural proteomes
#obtained from
#https://proteopedia.org/wiki/index.php/Amino_acid_composition
aa_compo = pd.read_csv('aa_compo.csv')
aas = list(aa_compo.aa)
aa_set = set(aas)
aas.sort()

train = pickle.load(open('train', 'rb')) #
mdh_seqs = list(train[train['label'] == 1].seq)

tmp = ''.join(mdh_seqs)
aa_weights = [tmp.count(i) for i in aas]

mdh_seqs = pickle.load(open('mdh_seqs', 'rb'))
cluster_summary = pickle.load(open('cluster_summary', 'rb'))
rep7477 = list(mdh_seqs[mdh_seqs['seq_id']=='SEQ14897'].seq)[0]

def generate_random_seqs(template_seq, starter_lens, N):
    # take first n aa from template_seq and make up a seq with totally N aa
    # first_n is a list
    first_n = []
    seq = []
    for i in starter_lens:
        starter = template_seq[: i]
        n_needed = N - i
        for j in range(N):
            sampled = random.choices(aas, weights=aa_weights, k=n_needed)
            new_seq = starter + ''.join(sampled)
            first_n.append(i)
            seq.append(new_seq)
    res = pd.DataFrame({'first_n': first_n, 'seq': seq, 'length': [len(template_seq)] * len(seq)})
    return res


gen_random = generate_random_seqs(rep7477, [0, 10, 50, 100], 1000) #
gen_random_pred = evaluate_gen(gen_random, 100, 505, model)
gen_random_pred['group'] = ['random'] * gen_random_pred.shape[0]
gen_random_pred['perplexity'] = [np.nan] * gen_random_pred.shape[0]
# gen_random_pred.to_csv('gen_random.csv', index=False)



#######################################################################
######## put all generated dataframes together and save as one ########
#######################################################################
all_gen = pd.concat([all_gen, gen_random_pred], axis = 0)
all_gen.to_csv('all_gen.csv', index=False)









