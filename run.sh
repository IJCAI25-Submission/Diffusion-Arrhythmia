#!/bin/bash

method=$1
task=$2

if [ "$method" = "eff" ]; then
    if [ "$task" = "generalization" ]; then
        cd ./Baselines/EffNet/ || exit 1
        python downStream_train.py --mode=test
    elif [ "$task" = "personalization" ]; then
        cd ./Baselines/EffNet/ || exit 1
        python per_train.py --mode=test --split_num=1
    else
        echo "Invalid task"
    fi
elif [ "$method" = "moco" ]; then
    if [ "$task" = "generalization" ]; then
        cd ./Baselines/MoCo/ || exit 1
        python ger_train.py --mode=test
    elif [ "$task" = "personalization" ]; then
        cd ./Baselines/MoCo/ || exit 1
        python per_train.py --mode=test --split_num=1
    else
        echo "Invalid task"
    fi
elif [ "$method" = "diffEcg" ]; then
    if [ "$task" = "generalization" ]; then
        cd ./Diffusion_based/DiffusionModels/ || exit 1
        python ger_train.py --mode=test
    elif [ "$task" = "personalization" ]; then
        cd ./Diffusion_based/DiffusionModels/ || exit 1
        python per_bayes_delta_train.py --mode=test --split_num=1
    else
        echo "Invalid task"
    fi
else
    echo "Invalid method"
fi