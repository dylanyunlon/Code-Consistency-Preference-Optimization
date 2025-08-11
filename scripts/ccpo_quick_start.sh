#!/bin/bash
# CCPO快速启动脚本 - 完整流程自动化
# 从数据生成到模型训练的一站式解决方案

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[$(date '+%H:%M:%S')]${NC} $1"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# 默认参数
MODEL="mistralai/Mistral-7B-Instruct-v0.2"
DATASET="dylansss/ccpo_math_dataset"
ITER="1"
PAIRS="5"
DATA_FRAC="0"
FRAC_LEN="0"
VERIFICATION_SAMPLE_RATE="0.005"
BETA="0.01"
LEARNING_RATE="5.0e-7"
BATCH_SIZE="2"
ACCUMULATE="8"
NUM_EPOCHS="18"
SKIP_GENERATION="false"
SKIP_RANKING="false"
SKIP_PROCESSING="false"
AUTO_CLEAN="false"

# 解析命令行参数
while [[ "$#" -gt 0 ]]; do
    case $1 in
    --model) MODEL="$2"; shift ;;
    --dataset) DATASET="$2"; shift ;;
    --iter) ITER="$2"; shift ;;
    --pairs) PAIRS="$2"; shift ;;
    --sample-rate) VERIFICATION_SAMPLE_RATE="$2"; shift ;;
    --beta) BETA="$2"; shift ;;
    --lr) LEARNING_RATE="$2"; shift ;;
    --batch-size) BATCH_SIZE="$2"; shift ;;
    --epochs) NUM_EPOCHS="$2"; shift ;;
    --skip-generation) SKIP_GENERATION="true" ;;
    --skip-ranking) SKIP_RANKING="true" ;;
    --skip-processing) SKIP_PROCESSING="true" ;;
    --auto-clean) AUTO_CLEAN="true" ;;
    --help)
        echo "CCPO快速启动脚本"
        echo ""
        echo "参数："
        echo "  --model MODEL           模型路径 (默认: mistralai/Mistral-7B-Instruct-v0.2)"
        echo "  --dataset DATASET       数据集名称 (默认: dylansss/ccpo_math_dataset)"
        echo "  --iter ITER             迭代轮次 (默认: 1)"
        echo "  --pairs PAIRS           每问题响应数 (默认: 5)"
        echo "  --sample-rate RATE      验证采样率 (默认: 0.005)"
        echo "  --beta BETA             训练beta值 (默认: 0.01)"
        echo "  --lr LR                 学习率 (默认: 5.0e-7)"
        echo "  --batch-size SIZE       批大小 (默认: 2)"
        echo "  --epochs EPOCHS         训练轮数 (默认: 18)"
        echo "  --skip-generation       跳过数据生成"
        echo "  --skip-ranking         跳过CCPO排名"
        echo "  --skip-processing      跳过数据处理"
        echo "  --auto-clean           自动清理中间文件"
        echo "  --help                 显示帮助信息"
        exit 0
        ;;
    *)
        print_error "未知参数: $1"
        exit 1
        ;;
    esac
    shift
done

print_status "🚀 CCPO完整训练流程启动"
echo "=================================="
echo "Architecture B: 服务器按7B推理思路执行代码验证推理质量"
echo ""
echo "🔧 配置参数:"
echo "   模型: $MODEL"
echo "   数据集: $DATASET"
echo "   迭代: $ITER"
echo "   验证采样率: $VERIFICATION_SAMPLE_RATE"
echo "   Beta: $BETA"
echo "   学习率: $LEARNING_RATE"
echo "   批大小: $BATCH_SIZE"
echo "   训练轮数: $NUM_EPOCHS"
echo "=================================="

# 检查依赖
print_status "🔍 检查CCPO依赖..."

required_files=(
    "scripts/generate.py"
    "scripts/code_verified_rank.py"  
    "ccpo/trainer_code_verified.py"
    "execution_verifier.py"
)

missing_files=()
for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        missing_files+=("$file")
    fi
done

if [ ${#missing_files[@]} -ne 0 ]; then
    print_error "缺少必要文件:"
    for file in "${missing_files[@]}"; do
        echo "   - $file"
    done
    exit 1
fi

print_success "依赖检查完成"

# 设置目录
OUTPUT_DIR="generated_err_0801/iter${ITER}"
RANKING_DIR="ranking/$OUTPUT_DIR"
PROCESSED_DIR="processed_data/iter${ITER}"

# 步骤1: 数据生成
if [ "$SKIP_GENERATION" = "false" ]; then
    print_status "📊 步骤1: 生成推理过程数据"
    
    if [ -d "$OUTPUT_DIR" ] && [ -f "$OUTPUT_DIR/metadata_${DATA_FRAC}.json" ]; then
        print_warning "数据已存在，跳过生成步骤"
    else
        print_status "开始7B模型推理过程生成..."
        
        python scripts/generate.py \
            --model "$MODEL" \
            --dataset "$DATASET" \
            --output_dir "$OUTPUT_DIR" \
            --data_frac "$DATA_FRAC" \
            --frac_len "$FRAC_LEN" \
            --pairs "$PAIRS" \
            --numgpu 8 \
            --gpu 0
        
        print_success "推理过程生成完成"
    fi
else
    print_warning "跳过数据生成步骤"
fi

# 步骤2: CCPO代码验证排名
if [ "$SKIP_RANKING" = "false" ]; then
    print_status "🧠 步骤2: CCPO代码验证排名 (Architecture B核心)"
    
    if [ -f "$RANKING_DIR/ccpo_0_${DATA_FRAC}.npy" ]; then
        print_warning "排名结果已存在，跳过CCPO验证步骤"
    else
        print_status "开始服务器代码执行验证..."
        
        mkdir -p "$RANKING_DIR"
        
        python scripts/code_verified_rank.py \
            --model "$MODEL" \
            --output_dir "$OUTPUT_DIR" \
            --data_frac "$DATA_FRAC" \
            --frac_len "$FRAC_LEN" \
            --pairs "$PAIRS" \
            --gpu 0 \
            --verification_sample_rate "$VERIFICATION_SAMPLE_RATE" \
            --max_concurrent 1 \
            --debug_v2
        
        print_success "CCPO代码验证完成"
    fi
else
    print_warning "跳过CCPO验证步骤"
fi

# 步骤3: 数据处理
if [ "$SKIP_PROCESSING" = "false" ]; then
    print_status "📋 步骤3: 处理CCPO验证结果"
    
    if [ -d "$PROCESSED_DIR/dataset" ]; then
        print_warning "处理后数据已存在，跳过处理步骤"
    else
        print_status "构建偏好对数据集..."
        
        # 检查是否存在处理脚本
        if [ ! -f "scripts/process_ccpo_verification_results.py" ]; then
            print_error "缺少数据处理脚本: scripts/process_ccpo_verification_results.py"
            print_status "请创建此脚本或手动处理CCPO验证结果"
            exit 1
        fi
        
        python scripts/process_ccpo_verification_results.py \
            --input_dir "$OUTPUT_DIR" \
            --ranking_dir "$RANKING_DIR" \
            --output_dir "$PROCESSED_DIR" \
            --data_frac "$DATA_FRAC" \
            --pairs "$PAIRS"
        
        print_success "数据处理完成"
    fi
else
    print_warning "跳过数据处理步骤"
fi

# 步骤4: CCPO训练
print_status "🚀 步骤4: 启动CCPO训练"

if [ ! -d "$PROCESSED_DIR/dataset" ]; then
    print_error "训练数据不存在: $PROCESSED_DIR/dataset"
    print_status "请确保前面的步骤都成功完成"
    exit 1
fi

# 设置训练输出目录
TRAIN_OUTPUT_DIR="checkpoints/ccpo/iter${ITER}_${LEARNING_RATE}_beta${BETA}_rmsprop/code_verified_ccpo_score_${NUM_EPOCHS}"
mkdir -p "$TRAIN_OUTPUT_DIR"

print_status "开始CCPO增强训练..."
print_status "训练参数: LR=$LEARNING_RATE, Beta=$BETA, Batch=$BATCH_SIZE, Epochs=$NUM_EPOCHS"

# 检查是否有简化pipeline脚本
if [ -f "scripts/code_verified_pipeline_simple.sh" ]; then
    bash scripts/code_verified_pipeline_simple.sh \
        --dataset "$PROCESSED_DIR/dataset" \
        --output_dir "$TRAIN_OUTPUT_DIR" \
        --learning_rate "$LEARNING_RATE" \
        --beta "$BETA" \
        --batch_size "$BATCH_SIZE" \
        --accumulate "$ACCUMULATE" \
        --num "$NUM_EPOCHS" \
        --model "$MODEL" \
        --loss_type "code_verified"
else
    print_warning "简化pipeline脚本不存在，使用原版调用方式"
    
    # 直接调用训练（需要适当的配置文件）
    dataset_name=$(echo "$PROCESSED_DIR/dataset" | sed 's/\//_/g')
    config_file="recipes/uclaml-ccpo/config_ccpo_${dataset_name}.yaml"
    
    # 创建或更新配置文件
    if [ -f "recipes/uclaml-ccpo/config_full.yaml" ]; then
        cp recipes/uclaml-ccpo/config_full.yaml "$config_file"
        python3 scripts/update_dataset.py --dataset "$PROCESSED_DIR/dataset" --config "$config_file"
    else
        print_error "原版配置文件不存在"
        exit 1
    fi
    
    # 启动训练
    ACCELERATE_LOG_LEVEL=info accelerate launch \
        --config_file recipes/accelerate_configs/deepspeed_zero3.yaml \
        --main_process_port 2930 \
        ccpo/run_ccpo.py "$config_file" \
        --learning_rate="$LEARNING_RATE" \
        --beta="$BETA" \
        --optim="rmsprop" \
        --output_dir="$TRAIN_OUTPUT_DIR" \
        --run_name="ccpo" \
        --loss_type="code_verified" \
        --per_device_train_batch_size="$BATCH_SIZE" \
        --gradient_accumulation_steps="$ACCUMULATE" \
        --model_name_or_path="$MODEL" \
        --num_train_epochs="$NUM_EPOCHS"
fi

print_success "CCPO训练完成"

# 清理中间文件（可选）
if [ "$AUTO_CLEAN" = "true" ]; then
    print_status "🧹 清理中间文件..."
    
    # 保留最终结果，清理中间过程文件
    if [ -d "$TRAIN_OUTPUT_DIR" ] && [ -f "$TRAIN_OUTPUT_DIR/pytorch_model.bin" ]; then
        print_status "清理生成数据和排名文件..."
        rm -rf "$OUTPUT_DIR"
        rm -rf "$RANKING_DIR" 
        print_success "中间文件清理完成"
    else
        print_warning "训练未完成，跳过清理"
    fi
fi

# 输出总结
print_status "🎉 CCPO完整流程执行完成!"
echo "=================================="
echo "📊 结果总结:"
echo "   ✅ Architecture B核心创新已实现"
echo "   ✅ 7B模型推理过程 → 服务器代码执行验证 → 偏好对构建 → 强化学习训练"
echo ""
echo "📁 文件位置:"
echo "   - 训练后模型: $TRAIN_OUTPUT_DIR"
echo "   - 处理后数据: $PROCESSED_DIR"
if [ "$AUTO_CLEAN" = "false" ]; then
echo "   - 原始生成数据: $OUTPUT_DIR"
echo "   - CCPO排名结果: $RANKING_DIR"
fi
echo ""
echo "🔄 下一步建议:"
echo "   1. 评估训练后的模型性能"
echo "   2. 在测试集上验证CCPO效果"
echo "   3. 与原版CCPO模型进行对比"
echo "=================================="

print_success "CCPO快速启动脚本执行完成！"