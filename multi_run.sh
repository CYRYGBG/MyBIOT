#!/bin/bash

# ================= 配置区域 =================
export CUDA_VISIBLE_DEVICES=1
CODE_PATH="/home/yeqi3/cyr/code/MyBIOT/run_all_model_cross_sub.py"
DATA_BASE_PATH="/usr/data/yeqi3/biot_processed_fixed"
BIOT_PRETRAIN_PATH="/home/yeqi3/cyr/code/BIOT/pretrained-models/EEG-six-datasets-18-channels.ckpt"
SAVE_DIR="/usr/data/yeqi3/biot_results" 

DATASETS=("read" "read_new" "type" "type_new")
# MODELS=("SPaRCNet" "ContraWR" "CNNTransformer" "FFCL" "STTransformer" "BIOT")
MODELS=("BIOT")
SEEDS=(42 0 1 114514 3407)

EPOCHS=50
BATCH_SIZE=32
IN_CHANNELS=18
N_CLASSES=2

# ================= 循环逻辑 =================

for dataset in "${DATASETS[@]}"; do
    for model in "${MODELS[@]}"; do
        
        echo "================================================================"
        echo "Launching Parallel Seeds for Dataset: $dataset | Model: $model"
        echo "================================================================"

        for seed in "${SEEDS[@]}"; do
            CURRENT_DATA_PATH="$DATA_BASE_PATH/$dataset"

            if [ "$model" == "BIOT" ]; then
                PRETRAIN_ARG="--pretrain_model_path $BIOT_PRETRAIN_PATH"
            else
                PRETRAIN_ARG=""
            fi

            # --- 关键修改点 1: 使用 & 放到后台运行 ---
            # 重定向日志到文件，避免 5 个进程的输出在终端里打架混在一起
            LOG_FILE="/home/yeqi3/cyr/code/MyBIOT/log_txt/log_${dataset}_${model}_seed${seed}.txt"
            
            python "$CODE_PATH" \
                --root_path "$CURRENT_DATA_PATH" \
                --dataset "$dataset" \
                --model "$model" \
                --seed "$seed" \
                --in_channels "$IN_CHANNELS" \
                --n_classes "$N_CLASSES" \
                --batch_size "$BATCH_SIZE" \
                --epochs "$EPOCHS" \
                --save_dir "$SAVE_DIR" \
                $PRETRAIN_ARG > "$LOG_FILE" 2>&1 &
            
            echo "Launched seed $seed, logging to $LOG_FILE"
        done

        # --- 关键修改点 2: 使用 wait 等待当前模型的所有种子跑完 ---
        echo "Waiting for all 5 seeds of $model to finish..."
        wait
        echo "Batch for $model finished."

    done
done

echo "All experiments finished!"