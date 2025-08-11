#!/bin/bash

# CCPO训练数据生成脚本
# 将code_verified_rank.py的结果转换为训练数据

set -e

echo "🚀 CCPO训练数据生成开始"
echo "========================"

# 配置路径
BASE_DIR="/data/jiacheng/dylan/iclr2026/Code-Consistency-Preference-Optimization"
INPUT_DIR="${BASE_DIR}/generated/iter1"
RANKING_DIR="${BASE_DIR}/ranking/generated/iter1"
OUTPUT_DIR="${BASE_DIR}/processed_data/iter1"

echo "📂 配置路径:"
echo "   输入目录: ${INPUT_DIR}"
echo "   排名目录: ${RANKING_DIR}"
echo "   输出目录: ${OUTPUT_DIR}"

# 检查输入文件是否存在
echo ""
echo "🔍 检查必需文件..."

if [ ! -d "${INPUT_DIR}" ]; then
    echo "❌ 输入目录不存在: ${INPUT_DIR}"
    exit 1
fi

if [ ! -f "${INPUT_DIR}/metadata_0.json" ]; then
    echo "❌ 元数据文件不存在: ${INPUT_DIR}/metadata_0.json"
    exit 1
fi

if [ ! -d "${RANKING_DIR}" ]; then
    echo "❌ 排名目录不存在: ${RANKING_DIR}"
    exit 1
fi

if [ ! -f "${RANKING_DIR}/ccpo_0_0.npy" ]; then
    echo "❌ CCPO排名文件不存在: ${RANKING_DIR}/ccpo_0_0.npy"
    exit 1
fi

echo "✅ 所有必需文件检查通过"

# 检查响应文件
echo ""
echo "🔍 检查响应文件..."
for i in {0..4}; do
    response_file="${INPUT_DIR}/responses_${i}.json"
    if [ ! -f "${response_file}" ]; then
        echo "❌ 响应文件不存在: ${response_file}"
        exit 1
    fi
    echo "✅ 找到响应文件: responses_${i}.json"
done

# 创建输出目录
echo ""
echo "📁 创建输出目录..."
mkdir -p "${OUTPUT_DIR}"
echo "✅ 输出目录已创建: ${OUTPUT_DIR}"

# 运行处理脚本
echo ""
echo "🔄 运行CCPO验证结果处理脚本（对话格式）..."
echo "命令: python scripts/process_ccpo_verification_results.py \\"
echo "  --input_dir ${INPUT_DIR} \\"
echo "  --ranking_dir ${RANKING_DIR} \\"
echo "  --output_dir ${OUTPUT_DIR} \\"
echo "  --data_frac 0 \\"
echo "  --pairs 5 \\"
echo "  --score_threshold 5.0 \\"
echo "  --confidence_threshold 0.1 \\"
echo "  --output_format conversation"

cd "${BASE_DIR}"

python scripts/process_ccpo_verification_results.py \
    --input_dir "${INPUT_DIR}" \
    --ranking_dir "${RANKING_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --data_frac 0 \
    --pairs 5 \
    --score_threshold 5.0 \
    --confidence_threshold 0.1 \
    --output_format conversation

# 检查输出文件
echo ""
echo "🔍 检查生成的训练数据..."

if [ ! -f "${OUTPUT_DIR}/train_prefs.jsonl" ]; then
    echo "❌ 训练数据文件未生成: ${OUTPUT_DIR}/train_prefs.jsonl"
    exit 1
fi

if [ ! -f "${OUTPUT_DIR}/test_prefs.jsonl" ]; then
    echo "❌ 测试数据文件未生成: ${OUTPUT_DIR}/test_prefs.jsonl"
    exit 1
fi

echo "✅ 训练数据生成成功!"

# 显示数据统计
echo ""
echo "📊 数据统计:"
train_lines=$(wc -l < "${OUTPUT_DIR}/train_prefs.jsonl")
test_lines=$(wc -l < "${OUTPUT_DIR}/test_prefs.jsonl")
echo "   训练集: ${train_lines} 个偏好对"
echo "   测试集: ${test_lines} 个偏好对"

# 显示样本数据
echo ""
echo "📝 训练数据样本 (前2行):"
head -n 2 "${OUTPUT_DIR}/train_prefs.jsonl" | python -m json.tool

echo ""
echo "✅ CCPO训练数据生成完成!"
echo ""
echo "🎯 下一步:"
echo "   使用以下配置运行训练:"
echo "   dataset_mixer:"
echo "     \"${OUTPUT_DIR}/train_prefs.jsonl\": 1.0"
echo ""
echo "🚀 启动训练命令:"
echo "   ACCELERATE_LOG_LEVEL=info accelerate launch \\"
echo "     --config_file recipes/accelerate_configs/deepspeed_zero3_3gpu.yaml \\"
echo "     --main_process_port 2930 \\"
echo "     --num_processes 3 \\"
echo "     ccpo/run_ccpo.py config_ccpo_working.yaml"