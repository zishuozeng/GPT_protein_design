import pandas as pd
import numpy as np
import pickle, random

############################
##### define functions #####
############################
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
    counts = 0
    for i in LIST:
        if len(i) >= MIN and len(i) < MAX:
            counts += 1
    return counts

def write_finetune_inputs(SEQS, FILE):
    with open(FILE, 'w') as f:
        for i in SEQS:
            f.write('<|endoftext|>\n')
            f.write(i + '\n')
    f.close()


#############################
####### postive data ########
#############################
headers, pos_seqs = read_fasta('TAL.txt')

lens = [len(i) for i in pos_seqs]
np.median(lens) #728.0
max(lens) #2134
min(lens) #483

tal_seqs = pd.DataFrame({'seq_name':headers, 'seq':pos_seqs, 'length':lens})
tal_seqs.to_csv('tal_seqs.csv', index=False)

#write input file for finetuning
write_finetune_inputs(pos_seqs, 'TAL_input.txt')

random.shuffle(pos_seqs)

train_pos_seqs = pos_seqs[:900]
test_pos_seqs = pos_seqs[900:]


#############################
####### negative data #######
#############################
uniprot = pd.read_csv('uniprot-reviewed_yes.tab', sep = '\t') #(568002, 8)
uniprot = uniprot.drop(['Entry name', 'Organism', 'Status', 'Protein names'], axis = 1).dropna(axis = 0, how = 'any') #(273877, 4)
uniprot.columns = ['id', 'length', 'ec', 'seq'] #[273877 rows x 4 columns]

#for enzymes mapping to multiple ECs, assign 4.3.1.25 if it's one of the ECs
ecs = list(uniprot.ec)
ecs1 = []
for i in ecs:
    if ';' in i:
        splits = i.split(';')
        splits = [i.strip() for i in splits]
        if '4.3.1.25' in splits:
            ecs1.append('4.3.1.25')
        else:
            ecs1.append(splits[0])
    else:
        ecs1.append(i)

uniprot['ec'] = ecs1

####### select seqs with EC4.3.1.X and EC4.3.X.X to evaluate prediction resolution
ec1 = []; ec2 = []; ec3 = []; ec4 = []
for i in list(uniprot.ec):
    splits = i.split('.')
    ec1.append(splits[0])
    ec2.append(splits[1])
    ec3.append(splits[2])
    ec4.append(splits[3])

is_ec4_3 = []
for i in range(len(ec1)):
    if ec1[i] == '4' and ec2[i] == '3':
        is_ec4_3.append('y')
    else:
        is_ec4_3.append('n')

uniprot['ec1'] = ec1
uniprot['ec2'] = ec2
uniprot['ec3'] = ec3
uniprot['ec4'] = ec4
uniprot['is_ec4_3'] = is_ec4_3

#curate negative set
neg_set = uniprot[uniprot['ec'] != '4.3.1.25'] #[273875 rows x 4 columns]
neg_set = neg_set[neg_set['is_ec4_3'] == 'n'] #[270573 rows x 9 columns]

#curate functionally close enzymes for later evaluation
func_close = neg_set[neg_set['is_ec4_3'] == 'y'] #[3302 rows x 9 columns]
func_close[func_close['ec3']=='1'] #[620 rows x 9 columns]
func_close.to_csv('func_close.csv', index=False)

#curate negative set that mimics positive set w.r.t. length distribution
ranges = list(range(450, 2100, 50))
ranges[0] = 483
ranges[-1] = 2173

neg_seqs = [] #
for i in ranges:
    sub_set = neg_set[(neg_set['length'] >= i) & (neg_set['length'] < i+50)]
    n = length_interval_count(pos_seqs, i, i+50)
    neg_seqs += list(sub_set.sample(n).reset_index(drop=True).seq)

random.shuffle(neg_seqs)

train_neg_seqs = neg_seqs[:900]
test_neg_seqs = neg_seqs[900:]

train = pd.DataFrame({'seq':train_neg_seqs + train_pos_seqs, 'label':[0]*900 + [1] *900})
test = pd.DataFrame({'seq':test_neg_seqs + test_pos_seqs, 'label':[0]*100 + [1] *100})

train.to_csv('train.csv', index=False)
test.to_csv('test.csv', index=False)


















