# !/bin/bash
PROJECT_ROOT=$(pwd)
POPE_DATA_DIR=playground/data/eval/pope
AMBER_DATA_DIR=playground/data/eval/amber

export NCCL_P2P_DISABLE=1 # not using this cause prepare to hang, for crap gpu

function prepare_data()
{

    # requirements
    pip install gdown

    # clone the repo
    mkdir tmp
    cd tmp
    git clone https://github.com/junyangwang0410/AMBER.git
    git clone https://github.com/AoiDragon/POPE.git
    cd $PROJECT_ROOT

    # prepare amber
    pip install -U spacy
    python -m spacy download en_core_web_lg
    pip install nltk
    
    mkdir -p playground/data/eval/amber
    cp -r tmp/AMBER playground/data/eval/amber
    cd playground/data/eval/pope
    wget http://images.cocodataset.org/zips/val2014.zip
    unzip val2014.zip
    cd $PROJECT_ROOT

    # prepare pope
    mkdir -p playground/data/eval/pope
    cp -r tmp/POPE playground/data/eval/pope
    cd playground/data/eval/amber/data 
    gdown "https://drive.google.com/u/1/uc?id=1MaCHgtupcZUjf007anNl4_MV0o4DjXvl"
    unzip AMBER.zip
    cd ${PROJECT_ROOT}

}

function eval_pope()
{
    python -m llava.eval.model_vqa_loader \
        --model-path liuhaotian/llava-v1.5-13b \
        --question-file playground/data/eval/pope/llava_pope_test.jsonl \
        --image-folder ./playground/data/eval/pope/val2014 \
        --answers-file ./playground/data/eval/pope/answers/llava-v1.5-13b.jsonl \
        --temperature 0 \
        --conv-mode vicuna_v1 

    python llava/eval/eval_pope.py \
        --annotation-dir playground/data/eval/pope/POPE/output/coco \
        --question-file playground/data/eval/pope/llava_pope_test.jsonl \
        --result-file ./playground/data/eval/pope/answers/llava-v1.5-13b.jsonl \

}

function eval_amber()
{
    output_file=playground/data/eval/amber/answers/llava-v1.5-13b.jsonl
    python -m llava.eval.model_vqa_amber \
        --model-path liuhaotian/llava-v1.5-13b \
        --question-file playground/data/eval/amber/AMBER/data/query/query_all.json \
        --image-folder playground/data/eval/amber/image \
        --answers-file ${output_file} \
        --temperature 0 \
        --conv-mode vicuna_v1


    python llava/eval/eval_amber.py \
    --inference_data $output_file \
    --word_association playground/data/eval/amber/AMBER/data/relation.json \
    --safe_words playground/data/eval/amber/AMBER/data/safe_words.txt \
    --annotation playground/data/eval/amber/AMBER/data/annotations.json \
    --metrics playground/data/eval/amber/AMBER/data/metrics.txt \
    --evaluation_type a 
}

# 1. prepare data
# # if already prepare, don't do twice
#  prepare_data


# 2. eval pope
eval_pope

# 3. eval amber
eval_amber