#!/bin/bash
# CCPO训练脚本 - 基于代码验证的偏好优化
# 修改自run_sppo_mistral.sh，集成数学数据集和代码验证排名

echo "🚀 CCPO (Code-Consistency Preference Optimization) 训练开始"
echo "基于代码执行验证的数学推理优化"
echo "="*70

# 配置参数
iter_num=3
base_model="mistralai/Mistral-7B-Instruct-v0.2"
ccpo_dataset_size=60000
verification_sample_rate=0.003  # 0.3%的样本进行代码验证

# 第一步：构建CCPO数学数据集（只在第一次运行时执行）
if [ ! -d "data/ccpo_math_dataset" ]; then
    echo "📊 步骤1: 构建CCPO数学数据集..."
    python scripts/build_ccpo_math_dataset.py \
        --target_size $ccpo_dataset_size \
        --output_path "data/ccpo_math_dataset" \
        --push_to_hub \
        --hub_dataset_id "UCLA-AGI/ccpo-math-60k" \
        --verify_samples 10
    
    if [ $? -ne 0 ]; then
        echo "❌ 数据集构建失败，退出"
        exit 1
    fi
    echo "✅ 数学数据集构建完成"
else
    echo "✅ 检测到现有数学数据集，跳过构建"
fi

# 迭代训练循环
for i in $(seq 1 $iter_num); do
    echo ""
    echo "🔄 开始CCPO迭代 ${i}/${iter_num}"
    echo "="*50
    
    # 设置模型路径
    if [ "$i" -eq 1 ]; then
        MODEL=$base_model
        echo "📝 使用基础模型: $MODEL"
    else
        MODEL=$OUTPUT_DIR
        echo "📝 使用前一轮模型: $MODEL"
    fi
    
    # 设置输出路径
    OUTPUT_DIR="checkpoints/Mistral-7B-Instruct-CCPO-Iter${i}"
    GENERATED_DIR="generated/ccpo_iter${i}"
    RANKING_DIR="ranking/ccpo_iter${i}"
    
    # 设置数据集（使用CCPO数学数据集）
    if [ "$i" -eq 1 ]; then
        PROMPT="data/ccpo_math_dataset"  # 本地数学数据集
    else
        PROMPT="UCLA-AGI/ccpo-math-iter${i}"  # 后续迭代的数据集
    fi
    
    DATASET_NAME="ccpo_math_mistral-7b-iter${i}_verified"
    
    echo "📂 输出目录: $OUTPUT_DIR"
    echo "📊 数据集: $PROMPT"
    
    # 步骤2: 生成响应
    echo ""
    echo "🎯 步骤2: 生成模型响应..."
    python scripts/generate.py \
        --model $MODEL \
        --output_dir $GENERATED_DIR \
        --prompts $PROMPT \
        --maxlen 1024 \
        --pairs 5 \
        --world_size 1
    
    if [ $? -ne 0 ]; then
        echo "❌ 响应生成失败，退出"
        exit 1
    fi
    echo "✅ 响应生成完成"
    
    # 步骤3: CCPO代码验证排名（替代PairRM）
    echo ""
    echo "🔍 步骤3: CCPO代码验证排名..."
    echo "使用代码执行验证替代传统PairRM排名"
    
    python scripts/code_verified_rank.py \
        --model $MODEL \
        --output_dir $GENERATED_DIR \
        --prompts $PROMPT \
        --pairs 5 \
        --verification_sample_rate $verification_sample_rate \
        --max_concurrent 1 \
        --base_delay 15.0 \
        --max_delay 300.0 \
        --verification_url "https://8.134.217.190:17432" \
        --verification_username "newuser" \
        --verification_password "newPass123" \
        --verification_model "claude-sonnect-4-20250514-all" \
        --checkpoint_interval 1 \
        --max_retries 2 \
        --debug_v2
    
    if [ $? -ne 0 ]; then
        echo "❌ 代码验证排名失败，退出"
        exit 1
    fi
    echo "✅ 代码验证排名完成"
    
    # 步骤4: 处理验证结果并创建偏好数据
    echo ""
    echo "📋 步骤4: 创建CCPO偏好数据..."
    
    python scripts/process_ccpo_verification_results.py \
        --ranking_dir $RANKING_DIR \
        --generated_dir $GENERATED_DIR \
        --output_dataset $DATASET_NAME \
        --verification_threshold 0.6 \
        --confidence_weight 0.8
    
    if [ $? -ne 0 ]; then
        echo "❌ 偏好数据创建失败，退出"
        exit 1
    fi
    echo "✅ CCPO偏好数据创建完成"
    
    # 步骤5: CCPO训练
    echo ""
    echo "🎓 步骤5: CCPO模型训练..."
    echo "使用代码验证增强的SPPO损失函数"
    
    python sppo/run_sppo.py \
        --model_name $MODEL \
        --dataset_name $DATASET_NAME \
        --output_dir $OUTPUT_DIR \
        --learning_rate 5.0e-7 \
        --num_train_epochs 1 \
        --per_device_train_batch_size 2 \
        --gradient_accumulation_steps 8 \
        --warmup_steps 100 \
        --logging_steps 10 \
        --save_steps 500 \
        --eval_steps 500 \
        --beta 0.01 \
        --max_length 1024 \
        --max_prompt_length 512 \
        --loss_type "code_verified" \
        --enable_code_verification \
        --verification_base_url "https://8.134.217.190:17432" \
        --verification_username "newuser" \
        --verification_password "newPass123" \
        --fp16 \
        --dataloader_num_workers 4 \
        --remove_unused_columns false \
        --report_to null
    
    if [ $? -ne 0 ]; then
        echo "❌ CCPO训练失败，退出"
        exit 1
    fi
    echo "✅ CCPO训练完成"
    
    # 步骤6: 模型评估
    echo ""
    echo "📊 步骤6: 模型评估..."
    
    python scripts/evaluate_ccpo_model.py \
        --model_path $OUTPUT_DIR \
        --test_dataset "data/ccpo_math_dataset_split/test" \
        --verification_sample_size 50 \
        --output_report "reports/ccpo_iter${i}_evaluation.json"
    
    if [ $? -eq 0 ]; then
        echo "✅ 模型评估完成"
    else
        echo "⚠️  模型评估失败，但继续下一轮"
    fi
    
    # 清理临时文件
    echo ""
    echo "🧹 清理临时文件..."
    if [ -d "$GENERATED_DIR" ] && [ "$i" -gt 1 ]; then
        # 保留最新两轮的生成结果
        prev_generated="generated/ccpo_iter$((i-2))"
        if [ -d "$prev_generated" ]; then
            rm -rf "$prev_generated"
            echo "删除旧生成文件: $prev_generated"
        fi
    fi
    
    echo "✅ 第 ${i} 轮CCPO训练完成"
    
    # 如果不是最后一轮，为下一轮准备数据集
    if [ "$i" -lt "$iter_num" ]; then
        echo ""
        echo "📋 为下一轮准备数据集..."
        
        # 可选：从当前模型生成新的数学问题
        python scripts/generate_math_problems_from_model.py \
            --model_path $OUTPUT_DIR \
            --output_dataset "UCLA-AGI/ccpo-math-iter$((i+1))" \
            --num_problems 1000 \
            --difficulty_range "2-4"
        
        if [ $? -eq 0 ]; then
            echo "✅ 下一轮数据集准备完成"
        else
            echo "⚠️  数据集生成失败，下一轮将使用原始数据集"
        fi
    fi
    
done

echo ""
echo "🎉 CCPO训练完成！"
echo "="*70
echo "📊 训练总结:"
echo "   - 迭代次数: $iter_num"
echo "   - 基础模型: $base_model"
echo "   - 数学数据集大小: $ccpo_dataset_size"
echo "   - 验证采样率: $verification_sample_rate"
echo "   - 最终模型: $OUTPUT_DIR"
echo ""
echo "🔍 关键创新:"
echo "   ✅ 100%数学问题数据集（替代UltraFeedback）"
echo "   ✅ 代码执行验证排名（替代PairRM主观评分）"
echo "   ✅ 客观一致性损失函数（增强SPPO）"
echo "   ✅ 断点续传和限流处理（提高稳定性）"
echo ""
echo "📁 输出文件:"
echo "   - 最终模型: $OUTPUT_DIR"
echo "   - 数学数据集: data/ccpo_math_dataset"
echo "   - 评估报告: reports/"
echo "   - 验证日志: logs/"
echo ""
echo "🚀 CCPO (Code-Consistency Preference Optimization) 完成！"