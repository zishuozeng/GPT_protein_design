
### We set up our environment with following commands. Note that this setup has been tested on Ubuntu 22.04 system only. 

### Prerequisites: Anaconda, git

### Note that the finetuning process requires a large GPU memory. Among the GPUs we tested (NVDIA A100, A10, V100, T4, P4, P100), only NVDIA A100 (80G) worked.

#create and activate conda environment
conda create -n py38 python=3.8
conda activate py38

#install some basic packages
pip install flax
pip install pandas
pip install datasets
pip install evaluate
pip install scikit-learn

#install pytorch
conda install pytorch torchvision torchaudio pytorch-cuda -c pytorch -c nvidia

#install tensorflow & keras
conda install -c conda-forge keras tensorflow-gpu

#install transformers
pip install SentencePiece
pip install transformers






