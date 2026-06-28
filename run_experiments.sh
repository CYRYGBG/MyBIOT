#!/bin/bash

# ================= 配置区域 =================
# 指定 GPU ID
export CUDA_VISIBLE_DEVICES=0

# 【新增】定义你想保存权重的绝对路径
# 例如保存到大容量硬盘 /mnt/data/my_experiments
SAVE_DIR="/usr/data/yeqi3/biot_results" 

# Python 脚本的绝对路径
CODE_PATH="/home/yeqi3/cyr/code/MyBIOT/run_all_model_cross_sub.py"

# 数据集的根目录基础路径 (脚本会自动在后面拼上 dataset 名字)
# 根据你提供的示例 "/usr/data/yeqi3/biot_processed_fixed/type_new" 推断：
DATA_BASE_PATH="/usr/data/yeqi3/biot_processed_fixed"

# BIOT 预训练权重路径
BIOT_PRETRAIN_PATH="/home/yeqi3/cyr/code/BIOT/pretrained-models/EEG-six-datasets-18-channels.ckpt"

# 定义要运行的变量列表
# 数据集列表
# DATASETS=("read" "read_new" "type" "type_new")
DATASETS=("read_new" "type" "type_new")

# 模型列表
# 如果你只想跑部分模型，可以在这里删除不需要的
MODELS=("SPaRCNet" "ContraWR" "CNNTransformer" "FFCL" "STTransformer")

# 随机种子列表
SEEDS=(42 0 1 114514 3407)

# 其他通用参数
EPOCHS=50
BATCH_SIZE=32
IN_CHANNELS=18
N_CLASSES=2

# ================= 循环逻辑 =================

for dataset in "${DATASETS[@]}"; do
    for model in "${MODELS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            
            echo "----------------------------------------------------------------"
            echo "Starting Experiment:"
            echo "Dataset: $dataset | Model: $model | Seed: $seed"
            echo "----------------------------------------------------------------"

            # 拼接具体的数据集路径
            CURRENT_DATA_PATH="$DATA_BASE_PATH/$dataset"

            # 核心逻辑：判断是否为 BIOT 模型
            if [ "$model" == "BIOT" ]; then
                # 如果是 BIOT，加上预训练参数
                PRETRAIN_ARG="--pretrain_model_path $BIOT_PRETRAIN_PATH"
            else
                # 如果不是，参数为空
                PRETRAIN_ARG=""
            fi

            # 运行命令
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
                $PRETRAIN_ARG

        done
    done
done

echo "All experiments finished!"