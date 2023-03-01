from transformers import pipeline
import pandas as pd
import numpy as np
from evaluate import load
import pickle, sys, io, time

def gpt_generate(starter_seq, max_length, model):
    sequences = model(starter_seq, max_length=max_length, do_sample=True, top_k=950, repetition_penalty=1.2,
                         num_return_sequences=1, eos_token_id=0)
    sequences = [i['generated_text'] for i in sequences]
    sequences = [i.replace('\n', '') for i in sequences]
    sequences = [i for i in sequences if i != 'M']
    return sequences

def seq_gen(MODEL_PATH, MAX_LEN, FIRST_NS, TEMPLATE, GROUP): #
    model = pipeline('text-generation', model= MODEL_PATH)
    ###
    gen_seqs = []
    first_n = []
    for i in FIRST_NS:
        for j in range(1000): ###can change
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

t = time.time()
gen = seq_gen('ft_out', 728, [0], '', 'generated_TAL') #728 is median of the curated TAL sequences
time.time() - t # ~4 hours
gen.to_csv('gen.csv', index=False)







