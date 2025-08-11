#!/bin/bash
# CCPO (Code Consistency Preference Optimization) Pipeline Script
# Architecture B: 服务器按7B推理思路执行代码验证推理质量

set -e
set -x

export OMP_NUM_THREADS=2

# CCPO默认参数 - 针对代码验证优化
LEARNING_RATE="5.0e-7"
ITER="1"
BETA="0.01"  # 适合代码验证的beta值
LOSS_TYPE="code_verified"  # 使用代码验证损失
OPTIM="rmsprop"
PREF="ccpo_score"
NUM=18
MODEL="mistralai/Mistral-7B-Instruct-v0.2"
DATASET="dylansss/ccpo_math_dataset"  # CCPO数学数据集
BATCH_SIZE=2  # 代码验证需要更小的batch size
ACCUMULATE=8  # 增加梯度累积步数
OUTPUT_DIR=""

# CCPO验证相关参数
VERIFICATION_URL="https://8.134.217.190:17432"
VERIFICATION_USERNAME="newuser"
VERIFICATION_PASSWORD="newPass123"
VERIFICATION_SAMPLE_RATE="0.005"
MAX_CONCURRENT="1"
VERIFICATION_MODEL="claude-sonnet-4-20250514-all"

# 数据处理参数
DATA_FRAC="0"
FRAC_LEN="0"
PAIRS="5"
NUMGPU="8"

# 解析命令行参数
while [[ "$#" -gt 0 ]]; do
    case $1 in
    --learning_rate)
        LEARNING_RATE="$2"
        shift
        ;;
    --beta)
        BETA="$2"
        shift
        ;;
    --optim)
        OPTIM="$2"
        shift
        ;;
    --output_dir)
        OUTPUT_DIR="$2"
        shift
        ;;
    --iter)
        ITER="$2"
        shift
        ;;
    --loss_type)
        LOSS_TYPE="$2"
        shift
        ;;
    --prefix)
        PREF="$2"
        shift
        ;;
    --model)
        MODEL="$2"
        shift
        ;;
    --dataset)
        DATASET="$2"
        shift
        ;;
    --num)
        NUM="$2"
        shift
        ;;
    --batch_size)
        BATCH_SIZE="$2"
        shift
        ;;
    --accumulate)
        ACCUMULATE="$2"
        shift
        ;;
    --verification_url)
        VERIFICATION_URL="$2"
        shift
        ;;
    --verification_sample_rate)
        VERIFICATION_SAMPLE_RATE="$2"
        shift
        ;;
    --verification_model)
        VERIFICATION_MODEL="$2"
        shift
        ;;
    --data_frac)
        DATA_FRAC="$2"
        shift
        ;;
    --frac_len)
        FRAC_LEN="$2"
        shift
        ;;
    --pairs)
        PAIRS="$2"
        shift
        ;;
    --numgpu)
        NUMGPU="$2"
        shift
        ;;
    *)
        echo "Unknown parameter passed: $1"
        exit 1
        ;;
    esac
    shift
done

# 设置输出目录（如果未指定）
if [ -z "$OUTPUT_DIR" ]; then
    PREF="${PREF}_${NUM}"
    LEVEL1="iter${ITER}_${LEARNING_RATE}_beta${BETA}_${OPTIM}"
    LEVEL2="${LOSS_TYPE}_${PREF}"
    OUTPUT_DIR="checkpoints/ccpo/${LEVEL1}/${LEVEL2}"
fi

# 创建日志文件名
log_file="ccpo_iter${ITER}_${LEARNING_RATE}_${BETA}_${OPTIM}_${LOSS_TYPE}_${PREF}"

echo "🚀 启动CCPO训练流程 (Architecture B)"
echo "=================================="
echo "核心创新: 服务器按7B推理思路执行代码验证推理质量"
echo "输出目录: $OUTPUT_DIR"
echo "数据集: $DATASET"
echo "验证服务器: $VERIFICATION_URL"
echo "学习率: $LEARNING_RATE"
echo "Beta值: $BETA"
echo "损失类型: $LOSS_TYPE"
echo "GPU数量: $NUMGPU"
echo "=================================="

# 检查CCPO依赖
echo "🔍 检查CCPO依赖..."
if [ ! -f "execution_verifier.py" ]; then
    echo "❌ execution_verifier.py 不存在"
    exit 1
fi

if [ ! -f "ccpo/trainer_code_verified.py" ]; then
    echo "❌ trainer_code_verified.py 不存在"
    exit 1
fi

if [ ! -f "config_code_verified.yaml" ]; then
    echo "❌ config_code_verified.yaml 不存在"
    exit 1
fi

echo "✅ CCPO依赖检查完成"

# 步骤1: 数据生成 - 7B模型生成推理过程
echo ""
echo "📊 步骤1: 生成推理过程数据"
echo "=========================="

# 检查是否已有生成的数据
if [ ! -d "generated/iter${ITER}" ]; then
    echo "🔄 开始生成推理过程..."
    
    # 运行数据生成
    python scripts/generate.py \
        --model "$MODEL" \
        --dataset "$DATASET" \
        --output_dir "generated/iter${ITER}" \
        --data_frac "$DATA_FRAC" \
        --frac_len "$FRAC_LEN" \
        --pairs "$PAIRS" \
        --numgpu "$NUMGPU" \
        --gpu 0
    
    echo "✅ 推理过程生成完成"
else
    echo "⏭️  跳过数据生成（已存在）"
fi

# 步骤2: CCPO代码验证排名 - Architecture B核心
echo ""
echo "🧠 步骤2: CCPO代码验证排名 (Architecture B)"
echo "======================================="

# 检查是否已有排名结果
ranking_dir="ranking/generated/iter${ITER}"
if [ ! -d "$ranking_dir" ] || [ ! -f "${ranking_dir}/ccpo_0_${DATA_FRAC}.npy" ]; then
    echo "🔄 开始CCPO推理质量验证..."
    
    # 创建排名目录
    mkdir -p "$ranking_dir"
    
    # 运行CCPO代码验证排名
    python scripts/code_verified_rank.py \
        --model "$MODEL" \
        --output_dir "generated/iter${ITER}" \
        --data_frac "$DATA_FRAC" \
        --frac_len "$FRAC_LEN" \
        --pairs "$PAIRS" \
        --gpu 0 \
        --verification_url "$VERIFICATION_URL" \
        --verification_username "$VERIFICATION_USERNAME" \
        --verification_password "$VERIFICATION_PASSWORD" \
        --verification_sample_rate "$VERIFICATION_SAMPLE_RATE" \
        --max_concurrent "$MAX_CONCURRENT" \
        --verification_model "$VERIFICATION_MODEL" \
        --debug_v2
    
    echo "✅ CCPO推理质量验证完成"
else
    echo "⏭️  跳过CCPO验证（已存在排名结果）"
fi

# 步骤3: 处理CCPO验证结果，构建训练数据集
echo ""
echo "📋 步骤3: 处理CCPO验证结果"
echo "========================"

processed_data_dir="processed_data/iter${ITER}"
if [ ! -d "$processed_data_dir" ]; then
    echo "🔄 处理CCPO验证结果..."
    
    # 创建处理数据目录
    mkdir -p "$processed_data_dir"
    
    # 运行CCPO结果处理脚本
    python scripts/process_ccpo_verification_results.py \
        --input_dir "generated_err_0801/iter${ITER}" \
        --ranking_dir "$ranking_dir" \
        --output_dir "$processed_data_dir" \
        --data_frac "$DATA_FRAC" \
        --pairs "$PAIRS"
    
    echo "✅ CCPO验证结果处理完成"
else
    echo "⏭️  跳过结果处理（已存在）"
fi

# 步骤4: 更新配置文件
echo ""
echo "⚙️  步骤4: 更新训练配置"
echo "===================="

# 生成数据集名称
dataset_name=$(echo "$processed_data_dir" | sed 's/\//_/g')
new_config_file="config_ccpo_${dataset_name}.yaml"

# 创建新的配置文件（基于正确的格式）
cat > "$new_config_file" << EOF
# Model arguments
model_name_or_path: $MODEL
torch_dtype: auto

# Data training arguments with CCPO verification
dataset_mixer:
  ${processed_data_dir}/dataset/train_prefs: 1.0

dataset_splits:
- train
preprocessing_num_workers: 4

# Code Verification Settings
enable_code_verification: true
verification_base_url: "$VERIFICATION_URL"
verification_username: "$VERIFICATION_USERNAME"
verification_password: "$VERIFICATION_PASSWORD"
verification_sample_size: 100

# CCPOTrainer arguments with CCPO enhancements
bf16: false
fp16: true
beta: $BETA
do_eval: false
evaluation_strategy: "no"
eval_steps: 500
gradient_accumulation_steps: $ACCUMULATE
gradient_checkpointing: true
gradient_checkpointing_kwargs:
  use_reentrant: false
hub_model_id: mistral-7b-instruct-ccpo-ccpo
learning_rate: $LEARNING_RATE
log_level: info
logging_steps: 10
loss_type: $LOSS_TYPE
lr_scheduler_type: linear
max_length: 1024
max_prompt_length: 512
num_train_epochs: $NUM
optim: $OPTIM
output_dir: $OUTPUT_DIR
per_device_train_batch_size: $BATCH_SIZE
per_device_eval_batch_size: 4
push_to_hub: false
save_strategy: "steps"
save_steps: 500
save_total_limit: 3
seed: 42
warmup_steps: 100
weight_decay: 0.01
remove_unused_columns: false
dataloader_num_workers: 4

# Enhanced logging
report_to: null
logging_first_step: true
logging_nan_inf_filter: true
greater_is_better: false
metric_for_best_model: "train_loss"
load_best_model_at_end: false

# PEFT configuration (disabled for full fine-tuning)
use_peft: false
lora_r: 16
lora_alpha: 32
lora_dropout: 0.1

# CCPO specific settings
generate_during_eval: false
EOF

echo "✅ 配置文件更新完成: $new_config_file"

# 创建对应的DeepSpeed配置文件
deepspeed_config="recipes/accelerate_configs/deepspeed_zero3_${NUMGPU}gpu.yaml"
if [ ! -f "$deepspeed_config" ]; then
    echo "🔧 创建DeepSpeed配置文件: $deepspeed_config"
    cat > "$deepspeed_config" << EOF
compute_environment: LOCAL_MACHINE
debug: false
deepspeed_config:
  deepspeed_multinode_launcher: standard
  offload_optimizer_device: none
  offload_param_device: none
  zero3_init_flag: true
  zero3_save_16bit_model: true
  zero_stage: 3
distributed_type: DEEPSPEED
downcast_bf16: 'no'
machine_rank: 0
main_training_function: main
mixed_precision: fp16
num_machines: 1
num_processes: $NUMGPU
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
EOF
fi

# 步骤5: 启动CCPO训练
echo ""
echo "🚀 步骤5: 启动CCPO训练"
echo "==================="

echo "训练参数:"
echo "  - 模型: $MODEL"
echo "  - 学习率: $LEARNING_RATE"
echo "  - Beta: $BETA"
echo "  - 损失类型: $LOSS_TYPE"
echo "  - 批大小: $BATCH_SIZE"
echo "  - 梯度累积: $ACCUMULATE"
echo "  - 训练轮数: $NUM"
echo "  - 输出目录: $OUTPUT_DIR"
echo "  - GPU数量: $NUMGPU"
echo ""

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 设置PYTHONPATH确保导入正确
export PYTHONPATH="$(pwd):$PYTHONPATH"

# 启动训练 - 使用修改后的run_ccpo.py（兼容原版）
echo "开始CCPO训练..."
ACCELERATE_LOG_LEVEL=info accelerate launch \
    --config_file "$deepspeed_config" \
    --main_process_port 2930 \
    --num_processes "$NUMGPU" \
    ccpo/run_ccpo.py "$new_config_file" \
    --learning_rate="$LEARNING_RATE" \
    --beta="$BETA" \
    --optim="$OPTIM" \
    --output_dir="$OUTPUT_DIR" \
    --run_name="ccpo" \
    --loss_type="$LOSS_TYPE" \
    --per_device_train_batch_size="$BATCH_SIZE" \
    --gradient_accumulation_steps="$ACCUMULATE" \
    --model_name_or_path="$MODEL" \
    --num_train_epochs="$NUM" \
    2>&1 | tee "${log_file}.log"

echo ""
echo "🎉 CCPO训练流程完成!"
echo "==================="
echo "✅ Architecture B核心创新已实现:"
echo "   - 7B模型生成推理过程"
echo "   - 服务器按推理思路生成并执行代码"
echo "   - 验证推理过程的质量"
echo "   - 基于客观验证结果进行强化学习"
echo ""
echo "📊 训练结果:"
echo "   - 模型保存位置: $OUTPUT_DIR"
echo "   - 训练日志: ${log_file}.log"
echo "   - 配置文件: $new_config_file"
echo ""
echo "🔄 下一步: 评估训练后的模型性能"

# 可选: 自动评估
if [ "$AUTO_EVAL" = "true" ]; then
    echo ""
    echo "🔍 自动评估训练后的模型..."
    python scripts/evaluate_ccpo_model.py \
        --model_path "$OUTPUT_DIR" \
        --test_dataset "$DATASET" \
        --verification_url "$VERIFICATION_URL" \
        --output_file "${OUTPUT_DIR}/evaluation_results.json"
    echo "✅ 自动评估完成"
fi