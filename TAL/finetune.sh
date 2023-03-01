
#PLEASE SPECIFY PATHS TO INPUT AND OUTPUT
conda activate py38
cd ~/transformers/examples/pytorch/language-modeling/
date
python run_clm.py --model_name_or_path nferruz/ProtGPT2 --train_file INPUT --tokenizer_name nferruz/ProtGPT2  --do_train --output_dir OUTPUT --per_device_train_batch_size 1 --learning_rate 1e-03
date

#***** train metrics *****
#  epoch                    =        3.0
#  train_loss               =     2.3552
#  train_runtime            = 0:05:41.92
#  train_samples            =        230
#  train_samples_per_second =      2.018
# train_steps_per_second   =      2.018


