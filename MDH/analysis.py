
import pandas as pd
import numpy as np
import pickle

'''
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# PART I: run hmmscan on the generated sequences finetuned by clusters @@@
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
'''

### generated fasta file for all generated seqs ###
def write_fasta(seq_names, seqs, fname):
    with open(fname, 'w') as f:
        for i in range(len(seqs)):
            f.write('>' + seq_names[i] + '\n')
            f.write(seqs[i] + '\n')
    f.close()

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

all_gen = pd.read_csv('all_gen.csv')

### write fasta file for seqs in 'all_gen' for hmmscan (Pfam domain detection)
write_fasta([str(i) for i in list(all_gen.index)], list(all_gen.seq), 'files_for_analysis/all_gen.fa')

########################################################
### after hmmscan done running, parse hmmscan output ###
########################################################
def parse_hmmscan(FILE):
    with open(FILE, 'r') as f:
        lines = f.readlines()
    f.close()
    seq_domains = {}
    for i in lines:
        if i[0] != '#':
            splits = i.split(' ')
            splits = [j for j in splits if j != '']
            seq_num = int(splits[2])
            domain = ' '.join(splits[18:]).replace('\n', '')
            if seq_num not in list(seq_domains.keys()):
                seq_domains[seq_num] = domain
            else:
                seq_domains[seq_num] = seq_domains[seq_num] + ', ' + domain
    res = pd.DataFrame({'seq_no': list(seq_domains.keys()), 'domain': list(seq_domains.values())})  #
    return res


all_gen['seq_no'] = list(all_gen.index) #[24000 rows x 7 columns]

hmmscan_out = parse_hmmscan('files_for_analysis/hmmscan_out') #[19613 rows x 2 columns]

## count only MDH domains
malate_in = []
for i in list(hmmscan_out.domain):
    if 'lactate/malate dehydrogenase' in i or 'Malate/L-lactate dehydrogenase' in i:
        malate_in.append(1)
    else:
        malate_in.append(0)

hmmscan_out['mdh_domain'] = malate_in #[19613 rows x 3 columns]
hmmscan_out = hmmscan_out[hmmscan_out['mdh_domain'] == 1] #[13970 rows x 3 columns]
hmmscan_out1 = hmmscan_out[['seq_no', 'mdh_domain']].drop_duplicates() #[13970 rows x 2 columns]

all_gen = all_gen.merge(hmmscan_out1, on = 'seq_no', how = 'left') #[24000 rows x 8 columns]

all_gen.to_csv('all_gen.csv', index=False) # <=======update 'all_gen.csv' PART(1)

###calculate 'mdh_ratio' (ratio of seqs containing MDH domain)
groups = []; first_ns = []; mdh_ratio = []
for i in ['random', 'noft', 'cluster765', 'cluster2029', 'cluster5987', 'cluster7477']:
    for j in [0, 10, 50, 100]:
        df0 = all_gen[(all_gen['group'] == i) & (all_gen['first_n'] == j)]
        n0 = df0.shape[0]
        df1 = df0[df0['mdh_domain'] == 1]
        n1 = df1.shape[0]
        groups.append(i)
        first_ns.append(j)
        mdh_ratio.append(round(n1/n0, 2))

mdh_ratio = pd.DataFrame({'group':groups, 'first':first_ns, 'mdh_ratio': mdh_ratio})

'''
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# PART II: evaluate redundancy of the generated sequences @@@
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
'''
import pandas as pd
import os, pickle
import numpy as np

def write_fasta(seq_names, seqs, fname):
    with open(fname, 'w') as f:
        for i in range(len(seqs)):
            f.write('>' + seq_names[i] + '\n')
            f.write(seqs[i] + '\n')
    f.close()

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

def eval_redundancy(seqs):
    seq_names = list(range(len(seqs)))
    seq_names = [str(i) for i in seq_names]
    #write fasta for cd-hit input
    write_fasta(seq_names, seqs, 'files_for_analysis/cdhit_input.fa')
    #execute cd-hit
    os.system('cd-hit -i files_for_analysis/cdhit_input.fa -o files_for_analysis/cdhit_output')
    #collect cd-hit output
    headers, sequences = read_fasta('files_for_analysis/cdhit_output')
    df = pd.DataFrame({'nr':['y'] * len(sequences), 'seq': sequences})
    #remove output files
    os.remove('files_for_analysis/cdhit_input.fa')
    os.remove('files_for_analysis/cdhit_output')
    os.remove('files_for_analysis/cdhit_output.clstr')
    return df

all_gen = pd.read_csv('all_gen.csv') #[24000 rows x 8 columns]
nr_df = pd.DataFrame() #[11161 rows x 4 columns]
for i in ['cluster765', 'cluster2029', 'cluster5987', 'cluster7477']:
    for j in [0, 10, 50, 100]:
        df0 = all_gen[(all_gen['group'] == i) & (all_gen['first_n'] == j)]
        input_seqs = list(df0.seq)
        df = eval_redundancy(input_seqs)
        df['group'] = [i] * df.shape[0]
        df['first_n'] = [j] * df.shape[0]
        nr_df = pd.concat([nr_df, df], axis = 0)

all_gen = all_gen.merge(nr_df, on = ['group', 'first_n', 'seq'], how = 'left') #[24000 rows x 9 columns]

all_gen.to_csv('all_gen.csv', index=False) # <=======update 'all_gen.csv' PART(2)

###summarize ratios of non-redundant seqs by group
groups = []; first_ns = []; nr_ratio = []
for i in ['cluster765', 'cluster2029', 'cluster5987', 'cluster7477']:
    for j in [0, 10, 50, 100]:
        df0 = all_gen[(all_gen['group'] == i) & (all_gen['first_n'] == j)]
        n0 = df0.shape[0]
        df1 = df0[df0['nr'] == 'y']
        n1 = df1.shape[0]
        groups.append(i)
        first_ns.append(j)
        nr_ratio.append(round(n1/n0, 2))

nr_summary = pd.DataFrame({'group':groups, 'first':first_ns, 'nr_ratio': nr_ratio})

groups = []; nr_ratio = []
for i in ['cluster765', 'cluster2029', 'cluster5987', 'cluster7477']:
    df0 = all_gen[all_gen['group'] == i]
    n0 = df0.shape[0]
    df1 = df0[df0['nr'] == 'y']
    n1 = df1.shape[0]
    groups.append(i)
    nr_ratio.append(round(n1 /n0, 2))

nr_summary_lite = pd.DataFrame({'group':groups, 'generated_nr_ratio': nr_ratio})

internal_nr_ratio = pd.DataFrame({'group':['cluster765', 'cluster2029', 'cluster5987', 'cluster7477'],
                                  'internal_nr_ratio': [250/765, 166/2029, 1654/5987, 2408/7477]})

nr_summary_lite = nr_summary_lite.merge(internal_nr_ratio, on = 'group', how = 'left')
#          group  generated_nr_ratio  internal_nr_ratio
# 0   cluster765                0.95               0.33
# 1  cluster2029                0.31               0.08
# 2  cluster5987                0.86               0.28
# 3  cluster7477                0.67               0.32


'''
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# PART III: obtain best hit of each generated seq against all known MDHs @@@
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
'''
import pandas as pd
import os, pickle
import numpy as np

#run blastp
os.system('blastp -query files_for_analysis/all_gen.fa -subject train_sequences.fasta -out files_for_analysis/all_gen_blast_against_16k.out  -outfmt 7  -evalue 0.01')

###parse blastp output to extract best-hit (defined as highest bit-score) identity
##in the blastp program, each query's hits are ranked by bit-score from high to low
fname = 'files_for_analysis/all_gen_blast_against_16k.out'
with open(fname, 'r') as f:
    text = f.read()

chunks = text.split('# BLASTP 2.13.0+\n')
seq_nos = []
identities = []
align_lens = []
c = 0
for i in chunks[1:]:
    split_newline = i.split('\n')
    if len(split_newline) > 4:
        best_hit = split_newline[4]
        split_tab = best_hit.split('\t')
        seq_nos.append(int(split_tab[0]))
        identities.append(float(split_tab[2]))
        align_lens.append(int(split_tab[3]))
    c += 1

f.close()
best_hit = pd.DataFrame({'seq_no':seq_nos, 'identity':identities, 'align_length': align_lens}) #[20018 rows x 3 columns]

all_gen = pd.read_csv('all_gen.csv') #[24000 rows x 9 columns]
all_gen = all_gen.merge(best_hit, on = 'seq_no', how = 'left') #[24000 rows x 11 columns]
all_gen = all_gen.fillna({'identity': 0, 'align_length': 0})
all_gen.to_csv('all_gen.csv', index=False) # <=======update 'all_gen.csv' PART(3)
























