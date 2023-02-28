from transformers import pipeline
import pandas as pd
import numpy as np
from evaluate import load
import pickle, sys, io

headers, seqs = read_fasta('train_sequences.fasta')
headers = ['SEQ'+str(i) for i in list(range(len(headers)))]
mdh_seqs = pd.DataFrame({'seq_id': headers, 'seq':seqs}) #[16706 rows x 2 columns]

cluster_summary = pickle.load(open('cluster_summary', 'rb'))

rep765 = list(mdh_seqs[mdh_seqs['seq_id']=='SEQ9266'].seq)[0]
rep2029 = list(mdh_seqs[mdh_seqs['seq_id']=='SEQ1415'].seq)[0]
rep5987 = list(mdh_seqs[mdh_seqs['seq_id']=='SEQ6386'].seq)[0]
rep7477 = list(mdh_seqs[mdh_seqs['seq_id']=='SEQ14897'].seq)[0]

def gpt_generate(starter_seq, max_length, model):
    '''
    :param starter_seq: template sequence
    :param max_length: expected maximum length of generated sequences (note that actual generation may exceed that limit)
    :param model: path to the finetuned generator
    :return: sequences generated
    '''
    sequences = model(starter_seq, max_length=max_length, do_sample=True, top_k=950, repetition_penalty=1.2,
                         num_return_sequences=1, eos_token_id=0)
    sequences = [i['generated_text'] for i in sequences]
    sequences = [i.replace('\n', '') for i in sequences]
    sequences = [i for i in sequences if i != 'M']
    return sequences

def seq_gen(MODEL_PATH, MAX_LEN, FIRST_NS, TEMPLATE, GROUP):
    '''
    :param MODEL_PATH: path to the finetuned generator
    :param MAX_LEN: expected maximum length of generated sequences (note that actual generation may exceed that limit)
    :param FIRST_NS: first N residues of the template sequence as starter sequence
    :param TEMPLATE: template sequence
    :param GROUP: group name
    :return: a dataframe containing the outputs and parameters
    '''
    model = pipeline('text-generation', model= MODEL_PATH)
    ###
    gen_seqs = []
    first_n = []
    for i in FIRST_NS:
        for j in range(1000): ###can adjust number of generated sequences
            text_trap = io.StringIO()
            sys.stdout = text_trap
            seq = gpt_generate(TEMPLATE[:i], MAX_LEN, model)
            sys.stdout = sys.__stdout__
            gen_seqs += seq
            first_n.append(i)
        print('#############  ', i, '  ###############')
    perplexity = load("perplexity", module_type="metric")
    perps = perplexity.compute(predictions=gen_seqs, model_id=MODEL_PATH)['perplexities']
    gen_res = pd.DataFrame(
        {'group': [GROUP] * len(gen_seqs),
         'first_n': first_n, 'seq': gen_seqs,
         'length': [len(i) for i in gen_seqs],
         'perplexity': perps})
    print('##########  done short ###########')
    ###############################################
    return gen_res

#========== PLEASE SPECIFY PATH TO FINETUNE OUTPUT FIRST =========#
#generate by model finetuned with cluster765
gen765 = seq_gen(PATH/TO/CLUSTER765/FINETUNE/OUT, 400, [0, 10, 50, 100], rep765, 'cluster765')
# gen765.to_csv('gen765.csv', index=False)

#generate by model finetuned with cluster2029
gen2029 = seq_gen(PATH/TO/CLUSTER2029/FINETUNE/OUT, 400, [0, 10, 50, 100], rep2029, 'cluster2029')
# gen2029.to_csv('gen2029.csv', index=False)

#generate by model finetuned with cluster5987
gen5987 = seq_gen(PATH/TO/CLUSTER5987/FINETUNE/OUT, 400, [0, 10, 50, 100], rep5987, 'cluster5987')
# gen5987.to_csv('gen5987.csv', index=False)

#generate by model finetuned with cluster7477
gen7477 = seq_gen(PATH/TO/CLUSTER7477/FINETUNE/OUT, 400, [0, 10, 50, 100], rep7477, 'cluster7477')
# gen7477.to_csv('gen7477.csv', index=False)

#generate by model non-finetuned
gen_noft = seq_gen("nferruz/ProtGPT2", 400, [0, 10, 50, 100], rep7477, 'noft')
# gen_noft.to_csv('gen_noft.csv', index=False)

#combine all results into one
all_gen = pd.concat([gen_noft, gen765, gen2029, gen5987, gen7477], axis = 0)
all_gen.to_csv('all_gen.csv', index=False)




