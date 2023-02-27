import numpy as np
import re
import gc
import pandas as pd
import tensorflow as tf
import pickle
import random

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

#AMP sequences from https://webs.iiitd.edu.in/raghava/satpdb/down.php
#'antibacterial peptide', because other databases contain other types of antimicrobial peptide (e.g., antifungal)

#non_AMP sequences from UniProt with seq length<50 aa, then pick entries with names not about antimicrobial

headers1, amp = read_fasta('SATPdb_amp.fa') #3011
headers2, non_amp = read_fasta('short_peptides_uniprot.fasta') #13225

#load aa composition in natural proteomes
#obtained from
#https://proteopedia.org/wiki/index.php/Amino_acid_composition
aa_compo = pd.read_csv('aa_compo.csv')
aas = set(list(aa_compo.aa))

#make sure the amino acids are canonical
amp0 = [i for i in amp if set(i).issubset(aas)] #2879
random.shuffle(amp0)

#write finetuning input files
with open('finetune_train.txt', 'w') as f:
    for i in amp0[:2700]:
        f.write('<|endoftext|>\n')
        f.write(i + '\n')

with open('finetune_dev.txt', 'w') as f:
    for i in amp0[2700:]:
        f.write('<|endoftext|>\n')
        f.write(i + '\n')


#prepare data for discriminator training
amp1 = [] #2879
for i in amp:
    if len(i) <=50 and set(i).issubset(aas):
        amp1.append(i)

non_amp1 = [] #11533
for i in range(len(non_amp)):
    tmp = headers2[i].lower()
    if set(non_amp[i]).issubset(aas) and 'antimic' not in tmp and 'antibac' not in tmp and 'toxic' not in tmp and 'defensive' not in tmp:
        non_amp1.append(non_amp[i])


amp_df = pd.DataFrame({'seq': amp1, 'length':[len(i) for i in amp1]})
non_amp_df = pd.DataFrame({'seq': non_amp1, 'length':[len(i) for i in non_amp1]})


#sample non-AMPs so that aa and length distribution are similar
len_counts = pd.DataFrame(amp_df['length'].value_counts())
lens = list(len_counts.index); counts = list(len_counts.length)
sampled_non_amp = [] #2868
for i in range(len(lens)):
    tmp = non_amp_df[non_amp_df['length'] == lens[i]]
    if tmp.shape[0] < counts[i]:
        continue
    else:
        tmp = list(tmp.sample(counts[i], replace = False).seq)
    sampled_non_amp += tmp

makeup = list(non_amp_df.sample(2879-2868).seq)
sampled_non_amp += makeup

data = pd.DataFrame({'seq':list(amp_df.seq) + sampled_non_amp, 'label':[1] *amp_df.shape[0] + [0]*len(sampled_non_amp)})
data = data.sample(frac=1).reset_index(drop=True) #[5758 rows x 2 columns]

X_train = list(data.seq)[:5187]
y_train = list(data.label)[:5187]
X_test = list(data.seq)[5187:]
y_test = list(data.label)[5187:]

pickle.dump(X_train, open('X_train', 'wb'))
pickle.dump(y_train, open('y_train', 'wb'))
pickle.dump(X_test, open('X_test', 'wb'))
pickle.dump(y_test, open('y_test', 'wb'))





