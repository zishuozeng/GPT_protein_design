import pandas as pd
import numpy as np
import pickle
import os

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

def length_interval_count(LIST, MIN, MAX):
    #count how many sequences' lengths are within an interval between MIN and MAX
    counts = 0
    for i in LIST:
        if len(i) >= MIN and len(i) < MAX:
            counts += 1
    return counts

def write_fasta(seq_names, seqs, fname):
    with open(fname, 'w') as f:
        for i in range(len(seqs)):
            f.write('>' + seq_names[i] + '\n')
            f.write(seqs[i] + '\n')
    f.close()

def parse_cdhit(fname):
    #parse the cdhit output
    with open(fname, 'r') as f:
        text = f.read()
    f.close()
    chunks = text.split('>Cluster')
    clusters = {}
    representatives = {}
    for i in range(1, len(chunks[1:]), 1):
        seqs_in_cluster = []
        splits = chunks[i].split('\n')
        splits = [s for s in splits[1:] if s != '']
        for j in splits:
            seq_id = j.split('>')[1].split('...')[0]
            seqs_in_cluster.append(seq_id)
            if '*' in j:
                representatives['cluster' + str(i)] = seq_id
        clusters['cluster' + str(i)] = seqs_in_cluster
    return clusters, representatives

def write_finetune_inputs(SEQS, FILE):
    with open(FILE, 'w') as f:
        for i in SEQS:
            f.write('<|endoftext|>\n')
            f.write(i + '\n')
    f.close()

########################
##### collect data #####
########################
'''
positve data
'''
headers, seqs = read_fasta('train_sequences.fasta')

lens = [len(i) for i in seqs]
np.median(lens) #320
min(lens) #64
max(lens) #505

pickle.dump(seqs, open('/Users/zishuozeng/Desktop/MDH/mdh_seq_train', 'wb')) #16706


headers1, seqs1 = read_fasta('val_sequences.fasta')

lens1 = [len(i) for i in seqs1]
np.median(lens1) #319
min(lens1) #67
max(lens1) #472

pickle.dump(seqs1, open('/Users/zishuozeng/Desktop/MDH/mdh_seq_dev', 'wb')) #213

'''
negative data
'''
uniprot = pd.read_csv('uniprot-reviewed_yes.tab', sep = '\t') #(568002, 8)
uniprot = uniprot.drop(['Entry name', 'Organism', 'Status', 'Protein names'], axis = 1).dropna(axis = 0, how = 'any') #(273877, 4)
uniprot.columns = ['id', 'length', 'ec', 'seq'] #[273877 rows x 4 columns]
uniprot = uniprot[(uniprot['length'] <= 505) & (uniprot['length'] >=64)] #[213921 rows x 4 columns]

ecs = list(uniprot.ec)
ecs1 = []
for i in ecs:
    if ';' in i:
        splits = i.split(';')
        splits = [i.strip() for i in splits]
        if '1.1.1.37' in splits:
            ecs1.append('1.1.1.37')
        else:
            ecs1.append(splits[0])
    else:
        ecs1.append(i)

uniprot['ec'] = ecs1

####### select seqs with EC1.1.1.X and EC1.1.X.X, and valdiation set from ProteinGAN
# to evaluate model capability to classify functionally close proteins ######
ec1 = []; ec2 = []; ec3 = []; ec4 = []
for i in list(uniprot.ec):
    splits = i.split('.')
    ec1.append(splits[0])
    ec2.append(splits[1])
    ec3.append(splits[2])
    ec4.append(splits[3])

is_ec1_1 = []
for i in range(len(ec1)):
    if ec1[i] == '1' and ec2[i] == '1':
        is_ec1_1.append('y')
    else:
        is_ec1_1.append('n')

uniprot['ec1'] = ec1
uniprot['ec2'] = ec2
uniprot['ec3'] = ec3
uniprot['ec4'] = ec4
uniprot['is_ec1_1'] = is_ec1_1

negative_set = uniprot[(uniprot['is_ec1_1'] == 'n') & (uniprot['ec'] != '1.1.1.37')] #[206596 rows x 9 columns]

ec111x = uniprot[(uniprot['ec1']=='1') & (uniprot['ec2']=='1') & (uniprot['ec3']=='1') & (uniprot['ec4']!='37')]
ec111x = ec111x[ec111x['ec4'] != '-'] #[5766 rows x 9 columns]

ec11xx = uniprot[(uniprot['ec1']=='1') & (uniprot['ec2']=='1') & (uniprot['ec3']!='1')] #[543 rows x 9 columns]

mdh_close_ec_test = pd.DataFrame({'is_mdh':[0] * (5766 + 543) + [1] * len(seqs1),
                                  'group': ['EC1.1.1.X'] * 5766 + ['EC1.1.X.X'] * 543 + ['MDH'] *len(seqs1),
                                  'seq': list(ec111x.seq) + list(ec11xx.seq) + seqs1})
mdh_close_ec_test.to_csv('mdh_close_ec_test.csv', index=False)


######## sample negative seqs matching postive length distribution
intervals = list(range(60, 500, 50))
intervals[0] = 64
intervals[-1] = 455

neg_seqs = [] #16706
for i in intervals:
    sub_set = negative_set[(negative_set['length'] >= i) & (negative_set['length'] < i+50)]
    n = length_interval_count(seqs, i, i+50)
    neg_seqs += list(sub_set.sample(n).reset_index(drop=True).seq)

# pickle.dump(neg_seqs, open('neg_seqs', 'wb'))

#############################################
##### prepare training and testing sets #####
#############################################
# neg_seqs = pickle.load(open('neg_seqs', 'rb'))
headers, pos_seqs = read_fasta('train_sequences.fasta')

data = pd.DataFrame({'seq': pos_seqs + neg_seqs, 'label': [1] * len(pos_seqs) + [0] * len(neg_seqs)})
data = data.sample(frac = 1).reset_index(drop=True) #[33412 rows x 2 columns]

#select 5k for training set (2.5k MDH and 2.5k non-MDH), and the rest serves as testing set
X_train = list(data.seq)[:5000]
y_train = list(data.label)[:5000]

X_test = list(data.seq)[5000: ]
y_test = list(data.label)[5000:]

pickle.dump(X_train, open('X_train', 'wb'))
pickle.dump(y_train, open('y_train', 'wb'))
pickle.dump(X_test, open('X_test', 'wb'))
pickle.dump(y_test, open('y_test', 'wb'))


###########################################
### cluster all the known MDH sequences ###
###########################################
headers, seqs = read_fasta('train_sequences.fasta')
headers = ['SEQ'+str(i) for i in list(range(len(headers)))]
write_fasta(headers, seqs, 'cdhit/all_MDHs_for_cdhit.fa')

mdh_seqs = pd.DataFrame({'seq_id': headers, 'seq':seqs}) #[16706 rows x 2 columns]
# pickle.dump(mdh_seqs, open('mdh_seqs', 'wb'))

#run cd-hit <=================================================================working here
os.system('cd-hit -i cdhit/all_MDHs_for_cdhit.fa -o cdhit/cdhit_out -c 0.4 -n 2')

clusters, representatives = parse_cdhit('cdhit/cdhit_out.clstr')
n_seqs = [len(clusters[i]) for i in list(clusters.keys())]
n_seqs.sort()
len(n_seqs) #82
n_seqs.count(1) #39
# [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
# 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4,
# 4, 4, 4, 4, 5, 6, 6, 7, 7, 7, 8, 8, 8, 10, 11, 11, 13, 16, 16, 16, 16, 17,
# 19, 21, 28, 28, 30, 48,
# 765, 2029, 5987, 7477] #765+2029+5987+7477=16258

######################################
### prepare finetuning input files ###
######################################
n_seqs = [len(clusters[i]) for i in list(clusters.keys())]

cluster_summary = pd.DataFrame({'cluster':list(clusters.keys()),
                                'n_seq': n_seqs,
                                'seq_ids': list(clusters.values()),
                                'rep_seq': list(representatives.values())})
pickle.dump(cluster_summary, open('cluster_summary', 'wb'))

selected_clusters = cluster_summary[cluster_summary['n_seq'] >= 765]

cluster7477 = mdh_seqs[mdh_seqs['seq_id'].isin(list(selected_clusters[selected_clusters['n_seq']==7477].seq_ids)[0])]
cluster5987 = mdh_seqs[mdh_seqs['seq_id'].isin(list(selected_clusters[selected_clusters['n_seq']==5987].seq_ids)[0])]
cluster2029 = mdh_seqs[mdh_seqs['seq_id'].isin(list(selected_clusters[selected_clusters['n_seq']==2029].seq_ids)[0])]
cluster765 = mdh_seqs[mdh_seqs['seq_id'].isin(list(selected_clusters[selected_clusters['n_seq']==765].seq_ids)[0])]

write_fasta(list(cluster765.seq_id), list(cluster765.seq), 'cdhit/cluster765.fa')
write_fasta(list(cluster2029.seq_id), list(cluster2029.seq), 'cdhit/cluster2029.fa')
write_fasta(list(cluster5987.seq_id), list(cluster5987.seq), 'cdhit/cluster5987.fa')
write_fasta(list(cluster7477.seq_id), list(cluster7477.seq), 'cdhit/cluster7477.fa')

write_finetune_inputs(list(cluster765.seq), 'cdhit/cluster765.txt')
write_finetune_inputs(list(cluster2029.seq), 'cdhit/cluster2029.txt')
write_finetune_inputs(list(cluster5987.seq), 'cdhit/cluster5987.txt')
write_finetune_inputs(list(cluster7477.seq), 'cdhit/cluster7477.txt')






































