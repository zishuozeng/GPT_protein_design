#Please specify path to all_gen.fa

conda install -c biocore hmmer
mkdir hmmer
cd hmmer
wget http://ftp.ebi.ac.uk/pub/databases/Pfam/current_release/Pfam-A.seed.gz
wget http://ftp.ebi.ac.uk/pub/databases/Pfam/current_release/Pfam-A.hmm.gz
gunzip *
hmmbuild Pfam-A.hmm Pfam-A.seed
hmmpress Pfam-A.hmm

hmmscan --noali -E 0.001 --domE 0.001 --tblout hmmscan_out Pfam-A.hmm PATH/TO/all_gen.fa > run.log


