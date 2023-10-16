export PYTHONPATH=.

python evaluation/run_data_preprocess.py \
    --top_common_tags 10 \
    --max_num_questions 6000 \
    --min_question_score 3