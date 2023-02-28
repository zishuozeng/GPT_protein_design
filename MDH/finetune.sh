#Please specify paths to TRAIN, VALIDATION, and OUTPUT,
#for cluster765, cluster2029, cluster5987, and cluster7477 separately

cd ~/transformers/examples/pytorch/language-modeling/

#cluster765
python run_clm.py --model_name_or_path nferruz/ProtGPT2 --train_file TRAIN --validation_file VALIDATION --tokenizer_name nferruz/ProtGPT2  --do_train --do_eval --output_dir OUTPUT --per_device_train_batch_size 1 --learning_rate 1e-03

#cluster2029
python run_clm.py --model_name_or_path nferruz/ProtGPT2 --train_file TRAIN --validation_file VALIDATION --tokenizer_name nferruz/ProtGPT2  --do_train --do_eval --output_dir OUTPUT --per_device_train_batch_size 1 --learning_rate 1e-03

#cluster5987
python run_clm.py --model_name_or_path nferruz/ProtGPT2 --train_file TRAIN --validation_file VALIDATION --tokenizer_name nferruz/ProtGPT2  --do_train --do_eval --output_dir OUTPUT --per_device_train_batch_size 1 --learning_rate 1e-03

#cluster7477
python run_clm.py --model_name_or_path nferruz/ProtGPT2 --train_file TRAIN --validation_file VALIDATION --tokenizer_name nferruz/ProtGPT2  --do_train --do_eval --output_dir OUTPUT --per_device_train_batch_size 1 --learning_rate 1e-03



