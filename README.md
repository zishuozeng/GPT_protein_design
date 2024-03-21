
## GPT_protein_design

This repository contains data and scripts neccesary to reproduce the results in our manuscript: "Zeng et al. (2023) *Binary Discriminator Facilitates GPT-based Protein Design.*"  

#### Abstract  
>Language models based on generative pre-trained transformers (GPT) architecture, such as GPT2, GPT3, and ChatGPT, are revolutionalizing natural language processing (NLP). Proteins are akin to natural languages in a sense that residues (words) consist of sequences (sentences) carrying specific functions (semantic meanings), and thus advancements in NLP techniques are also being applied to protein modeling. Here we built a novel pipeline for de novo protein design. This pipeline stands on the shoulders of giants (big models): a recently developed generative model (ProtGPT2), plus a deep learning-based discrimnator empowered by embedding scheme derived from another large-scale project (ProtTrans). Finetuned by a set of desired proteins, the generator produces sequences likely with the desired function, which are further prioritized by the discriminator (trained on the desired proteins and negative set). We applied this pipeline to generating novel antimicrobial peptides (AMPs) and malate dehydrogenase (MDHs). Experimental validation showed that 6 of 24 AMP candidates have strong antibacterial effects (with 5 being broad-spectrum). For the MDH task, in silico analyses (functional domain, sequence identity, homology-based functional analysis, and most importantly, discriminator’s prediction) suggested that MDH function is highly likely present in the prioritized candidates. Moreover, the time consumption and data requirement of our pipeline are much less than a state-of-the-art GAN-based method (ProteinGAN). Altogether, we believe this pipeline will significantly faciliate protein de novo design. 

#### Illustrations for the pipeline (A) and the architecture of discriminator (B) are shown below:  
</br>
<p align="center">
<img width="500" alt="fig1" src="https://user-images.githubusercontent.com/125118900/222022597-842eab8e-dc27-4c10-9fa1-086dc8977e85.png">
</p>

#### Instruction
We demonstrated the utility of this pipeline in three peptide/protein generation & screening tasks: AMP (antimicrobial peptides), MDH (malate dehydrogenase), and TAL (phenylalanine ammonia-lyase). To reproduce results for each task, set the corresponding directory as working directory and run the scripts according to each individual instruction (`README.md`). Note that some data or models are too large to be stored on GitHub, but they can be downloaded with the links provided or can be easily reproduced following the script.

#### Prerequisites
Please follow `env_setup.sh` to set up the environment for the pipeline.

#### Application
If you want to follow the pipeline to generate and screen novel proteins, you can follow scripts in "TAL" directory and make modification accordingly for your own protein, as this directory is directly for application without trivial analysis.

#### Citation
If you find our work useful, please cite  
</br>
Zeng, Zishuo, et al. "Binary Discriminator Facilitates GPT-based Protein Design." bioRxiv (2023): 2023-11.


