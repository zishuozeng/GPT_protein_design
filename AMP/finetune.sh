#please have transformers set up
#please specify TRAIN_PATH, VALIDATION_PATH, and OUTPATH

cd transformers/examples/pytorch/language-modeling/

python run_clm.py --model_name_or_path nferruz/ProtGPT2 --train_file TRAIN_PATH --validation_file VALIDATION_PATH --tokenizer_name nferruz/ProtGPT2  --do_train --do_eval --output_dir OUTPATH --learning_rate 1e-03  --num_train_epochs 100 --per_device_train_batch_size 1

