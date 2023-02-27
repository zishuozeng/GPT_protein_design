
## AMP (antimicrobial peptides) generation & screening

### Prerequisites
Please follow environment setup instructions in upper directory.

### STEP 1: data preparation

**Starting files** 
1) "aa_compo.csv", amino acid composition in natural proteomes, obtained from https://proteopedia.org/wiki/index.php/Amino_acid_composition;
2) "SATPdb_amp.fa", AMP sequences from https://webs.iiitd.edu.in/raghava/satpdb/down.php ; limited to 'antibacterial peptide';
3) "short_peptides_uniprot.fasta", short peptides obtained from UniProt

**Outputs**
1) X_train, sequences of training set
2) y_train, labels of training set (0 is non-AMP, 1 is AMP)
3) X_test, sequences of testing set
4) y_test, labels of testing set (0 is non-AMP, 1 is AMP)

### STEP 2: data embedding and discriminator training

