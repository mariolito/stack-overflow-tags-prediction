export PYTHONPATH=.

#need to change model folder name if re-run
MODEL_FOLDER_NAME='codebert_model_results'

python evaluation/run_bert_based_model.py \
    --train_split train \
    --val_split val \
    --num_epochs 3 \
    --model_max_length 128 \
    --batch_size_train 32 \
    --batch_size_val 32 \
    --output_name ${MODEL_FOLDER_NAME}
