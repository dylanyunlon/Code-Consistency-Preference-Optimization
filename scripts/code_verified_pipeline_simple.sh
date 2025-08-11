#!/bin/bash
# 简化版CCPO Pipeline - 基于原版CCPO结构修改
# 适配现有的目录结构和调用方式

set -e
set -x

export OMP_NUM_THREADS=2

# 参数设置（与原版pipeline.sh保持一致的格式）
LEARNING_RATE="5.0e-7"
ITER="1"
BETA="0.01"  # CCPO适配的beta值
LOSS_TYPE="code_verified"  # 关键：使用CCPO损失类型
OPTIM="rmsprop"
PREF="ccpo_score"
NUM=18
MODEL="mistralai/Mistral-7B-Instruct-v0.2"
DATASET="processed_data/iter1/dataset"  # 处理后的CCPO数据
BATCH_SIZE=2
ACCUMULATE=8

# 解析命令行参数（与原版保持一致）
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
    *)
        echo "Unknown parameter passed: $1"
        exit 1
        ;;
    esac
    shift
done

# 设置输出目录（与原版格式一致）
PREF="${PREF}_${NUM}"
LEVEL1="iter${ITER}_${LEARNING_RATE}_beta${BETA}_${OPTIM}"
LEVEL2="${LOSS_TYPE}_${PREF}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/ccpo/${LEVEL1}/${LEVEL2}}"

log_file="ccpo_iter${ITER}_${LEARNING_RATE}_${BETA}_${OPTIM}_${LOSS_TYPE}_${PREF}"

echo "🚀 CCPO简化训练流程启动"
echo "======================"
echo "基于原版CCPO结构，最小化修改"
echo "输出目录: $OUTPUT_DIR"
echo "数据集: $DATASET"
echo "损失类型: $LOSS_TYPE (关键CCPO标识)"
echo "======================"

# 检查必要文件
echo "🔍 检查必要文件..."

if [ ! -f "ccpo/trainer_code_verified.py" ]; then
    echo "❌ ccpo/trainer_code_verified.py 不存在"
    echo "💡 请确保CCPO trainer文件在正确位置"
    exit 1
fi

if [ ! -d "$DATASET" ]; then
    echo "❌ 数据集目录不存在: $DATASET"
    echo "💡 请先运行CCPO数据处理流程："
    echo "   1. python scripts/generate.py"
    echo "   2. python scripts/code_verified_rank.py"
    echo "   3. python scripts/process_ccpo_verification_results.py"
    exit 1
fi

echo "✅ 文件检查完成"

# 创建CCPO配置文件（基于原版config修改）
dataset_name=$(echo "$DATASET" | sed 's/\//_/g')
new_config_file="recipes/uclaml-ccpo/config_ccpo_${dataset_name}.yaml"

# 复制原版配置
if [ -f "recipes/uclaml-ccpo/config_full.yaml" ]; then
    cp recipes/uclaml-ccpo/config_full.yaml "$new_config_file"
else
    echo "❌ 原版配置文件不存在: recipes/uclaml-ccpo/config_full.yaml"
    exit 1
fi

# 更新数据集路径
python3 scripts/update_dataset.py --dataset "$DATASET" --config "$new_config_file" >"$log_file.update.log" 2>&1

echo "✅ 配置文件更新完成: $new_config_file"

# 检查run_ccpo.py是否支持CCPO
if grep -q "CodeVerifiedCCPOTrainer\|code_verified" ccpo/run_ccpo.py; then
    echo "✅ run_ccpo.py 已支持CCPO"
else
    echo "⚠️  run_ccpo.py 不支持CCPO，将尝试原版训练"
    echo "💡 建议替换为支持CCPO的run_ccpo.py版本"
fi

echo "logging to $log_file.log"

# 启动训练（与原版pipeline.sh调用方式完全一致）
ACCELERATE_LOG_LEVEL=info accelerate launch \
    --config_file recipes/accelerate_configs/deepspeed_zero3.yaml \
    --main_process_port 2930 \
    ccpo/run_ccpo.py "$new_config_file" \
    --learning_rate=$LEARNING_RATE \
    --beta=$BETA \
    --optim="$OPTIM" \
    --output_dir="$OUTPUT_DIR" \
    --run_name="ccpo" \
    --loss_type=$LOSS_TYPE \
    --per_device_train_batch_size=$BATCH_SIZE \
    --gradient_accumulation_steps=$ACCUMULATE \
    --model_name_or_path=$MODEL \
    --num_train_epochs=$NUM \
    2>&1 | tee "${log_file}.log"

echo ""
echo "🎉 CCPO训练流程完成!"
echo "==================="
echo "📊 结果位置:"
echo "   - 模型: $OUTPUT_DIR"
echo "   - 日志: ${log_file}.log"
echo "   - 配置: $new_config_file"
echo ""
echo "🔑 关键CCPO特性:"
echo "   - 损失类型: $LOSS_TYPE"
echo "   - 数据源: 基于代码验证的偏好对"
echo "   - 训练器: 自动检测并使用CCPO trainer"