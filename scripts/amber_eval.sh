#!/bin/bash

CHUNKS=8
output_dir="tests/test_amber"
output_file="${output_dir}/test_llava-13b.jsonl"

# Clear out the output file if it exists.
> "$output_file"

# Install
# pip install -U spacy
# python -m spacy download en_core_web_lg
# pip install nltk
# git clone https://github.com/junyangwang0410/AMBER.git
# mv AMBER playground/data/amber

# Run Inference and generate the results
for IDX in $(seq 0 $((CHUNKS-1))); do
    chunk_output_file=${output_dir}/test_llava-13b-chunk$CHUNKS_$IDX.jsonl
    CUDA_VISIBLE_DEVICES=$IDX python -m llava.eval.model_vqa_amber \
        --model-path liuhaotian/llava-lcs558k-scienceqa-vicuna-13b-v1.3 \
        --question-file playground/data/amber/data/query/query_all.json \
        --image-folder playground/data/amber/images \
        --answers-file ${chunk_output_file} \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --conv-mode llava_v1 &
done

for IDX in $(seq 0 $((CHUNKS-1))); do
    cat "${chunk_output_file}" >> "$output_file"
done


python llava/eval/eval_amber.py \
  --inference_data $output_file \
  --word_association playground/data/amber/data/relation.json \
  --safe_words playground/data/amber/data/safe_words.txt \
  --annotation playground/data/amber/data/annotations.json \
  --metrics playground/data/amber/data/metrics.txt \
  --evaluation_type a \
