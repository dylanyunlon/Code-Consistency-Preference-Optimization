#!/usr/bin/env python3
"""
Code-Verified Data Processor
处理代码验证结果，生成CCPO训练数据
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
from datasets import Dataset, DatasetDict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Process code verification results for CCPO training")
    parser.add_argument("--verification_dir", type=str, required=True,
                       help="代码验证结果目录")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="输出数据集目录")
    parser.add_argument("--min_score_diff", type=float, default=0.1,
                       help="最小分数差异阈值")
    parser.add_argument("--min_confidence", type=float, default=0.3,
                       help="最小置信度阈值")
    parser.add_argument("--max_samples", type=int, default=10000,
                       help="最大样本数量")
    parser.add_argument("--test_ratio", type=float, default=0.1,
                       help="测试集比例")
    
    return parser.parse_args()

def load_verification_results(verification_dir: str) -> List[Dict[str, Any]]:
    """加载所有验证结果文件"""
    verification_dir = Path(verification_dir)
    all_preferences = []
    
    # 查找所有CCPO偏好文件
    for file_path in verification_dir.glob("ccpo_preferences_*.json"):
        logger.info(f"加载偏好文件: {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                preferences = json.load(f)
                all_preferences.extend(preferences)
                logger.info(f"从 {file_path} 加载了 {len(preferences)} 个偏好样本")
        except Exception as e:
            logger.error(f"加载 {file_path} 失败: {e}")
    
    logger.info(f"总共加载了 {len(all_preferences)} 个偏好样本")
    return all_preferences

def filter_high_quality_samples(
    preferences: List[Dict[str, Any]],
    min_score_diff: float = 0.1,
    min_confidence: float = 0.3
) -> List[Dict[str, Any]]:
    """过滤高质量样本"""
    filtered = []
    
    for sample in preferences:
        chosen_score = sample.get('chosen_verification_score', 0.5)
        rejected_score = sample.get('rejected_verification_score', 0.5)
        chosen_prob = sample.get('chosen_probs', 0.5)
        
        # 检查分数差异
        score_diff = chosen_score - rejected_score
        if score_diff < min_score_diff:
            continue
        
        # 检查置信度
        if chosen_prob < min_confidence:
            continue
        
        # 检查基本数据完整性
        if not all(key in sample for key in ['prompt', 'chosen', 'rejected']):
            continue
        
        filtered.append(sample)
    
    logger.info(f"过滤后保留 {len(filtered)}/{len(preferences)} 个高质量样本")
    return filtered

def augment_with_verification_metrics(
    preferences: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """使用验证指标增强偏好数据"""
    augmented = []
    
    for sample in preferences:
        chosen_score = sample.get('chosen_verification_score', 0.5)
        rejected_score = sample.get('rejected_verification_score', 0.5)
        
        # 重新计算偏好概率，更强地基于验证分数
        score_diff = chosen_score - rejected_score
        verification_weight = 0.7  # 验证分数权重
        original_weight = 0.3     # 原始概率权重
        
        # 基于验证分数的偏好概率
        verification_prob = 0.5 + score_diff * 0.4
        verification_prob = max(0.1, min(0.9, verification_prob))
        
        # 原始偏好概率
        original_prob = sample.get('chosen_probs', 0.5)
        
        # 加权组合
        final_prob = (verification_weight * verification_prob + 
                     original_weight * original_prob)
        final_prob = max(0.1, min(0.9, final_prob))
        
        # 更新样本
        augmented_sample = sample.copy()
        augmented_sample.update({
            'chosen_probs': final_prob,
            'chosen_probs_win': final_prob,
            'chosen_probs_lose': 1 - final_prob,
            'verification_enhanced': True,
            'original_chosen_probs': original_prob,
            'verification_chosen_probs': verification_prob,
            'score_difference': score_diff,
            'quality_score': (chosen_score + rejected_score) / 2
        })
        
        augmented.append(augmented_sample)
    
    return augmented

def balance_dataset(
    preferences: List[Dict[str, Any]],
    max_samples: int
) -> List[Dict[str, Any]]:
    """平衡数据集 - 按质量分层采样"""
    if len(preferences) <= max_samples:
        return preferences
    
    # 按质量分数排序
    preferences.sort(key=lambda x: x.get('quality_score', 0), reverse=True)
    
    # 分层采样：70%高质量，30%中等质量
    high_quality_count = int(max_samples * 0.7)
    medium_quality_count = max_samples - high_quality_count
    
    high_quality = preferences[:len(preferences)//2][:high_quality_count]
    medium_quality = preferences[len(preferences)//2:][:medium_quality_count]
    
    balanced = high_quality + medium_quality
    
    # 随机打乱
    import random
    random.shuffle(balanced)
    
    logger.info(f"平衡后的数据集: {len(balanced)} 个样本 "
               f"(高质量: {len(high_quality)}, 中等质量: {len(medium_quality)})")
    
    return balanced

def create_ccpo_dataset(
    preferences: List[Dict[str, Any]],
    test_ratio: float = 0.1
) -> DatasetDict:
    """创建CCPO训练数据集"""
    # 划分训练集和测试集
    test_size = int(len(preferences) * test_ratio)
    test_data = preferences[:test_size]
    train_data = preferences[test_size:]
    
    # 转换为Dataset格式
    def convert_to_dataset_format(data):
        dataset_dict = {
            'prompt': [],
            'chosen': [],
            'rejected': [],
            'chosen_probs': [],
            'chosen_probs_win': [],
            'chosen_probs_lose': [],
            'chosen_verification_score': [],
            'rejected_verification_score': []
        }
        
        for item in data:
            dataset_dict['prompt'].append(item['prompt'])
            dataset_dict['chosen'].append(item['chosen'])
            dataset_dict['rejected'].append(item['rejected'])
            dataset_dict['chosen_probs'].append(item['chosen_probs'])
            dataset_dict['chosen_probs_win'].append(item['chosen_probs_win'])
            dataset_dict['chosen_probs_lose'].append(item['chosen_probs_lose'])
            dataset_dict['chosen_verification_score'].append(item['chosen_verification_score'])
            dataset_dict['rejected_verification_score'].append(item['rejected_verification_score'])
        
        return Dataset.from_dict(dataset_dict)
    
    train_dataset = convert_to_dataset_format(train_data)
    test_dataset = convert_to_dataset_format(test_data)
    
    dataset_dict = DatasetDict({
        'train_prefs': train_dataset,
        'test_prefs': test_dataset
    })
    
    logger.info(f"创建数据集: 训练集 {len(train_data)} 样本, 测试集 {len(test_data)} 样本")
    
    return dataset_dict

def generate_statistics(
    original_preferences: List[Dict[str, Any]],
    final_preferences: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """生成数据处理统计信息"""
    stats = {
        'original_samples': len(original_preferences),
        'final_samples': len(final_preferences),
        'filter_ratio': len(final_preferences) / len(original_preferences) if original_preferences else 0,
        'score_statistics': {},
        'quality_distribution': {}
    }
    
    if final_preferences:
        chosen_scores = [p['chosen_verification_score'] for p in final_preferences]
        rejected_scores = [p['rejected_verification_score'] for p in final_preferences]
        score_diffs = [p['score_difference'] for p in final_preferences]
        quality_scores = [p['quality_score'] for p in final_preferences]
        
        stats['score_statistics'] = {
            'chosen_score_mean': np.mean(chosen_scores),
            'chosen_score_std': np.std(chosen_scores),
            'rejected_score_mean': np.mean(rejected_scores),
            'rejected_score_std': np.std(rejected_scores),
            'score_diff_mean': np.mean(score_diffs),
            'score_diff_std': np.std(score_diffs),
            'quality_score_mean': np.mean(quality_scores),
            'quality_score_std': np.std(quality_scores)
        }
        
        # 质量分布
        high_quality = sum(1 for q in quality_scores if q > 0.7)
        medium_quality = sum(1 for q in quality_scores if 0.4 <= q <= 0.7)
        low_quality = sum(1 for q in quality_scores if q < 0.4)
        
        stats['quality_distribution'] = {
            'high_quality': high_quality,
            'medium_quality': medium_quality,
            'low_quality': low_quality,
            'high_quality_ratio': high_quality / len(final_preferences),
            'medium_quality_ratio': medium_quality / len(final_preferences),
            'low_quality_ratio': low_quality / len(final_preferences)
        }
    
    return stats

def main():
    """主函数"""
    args = parse_arguments()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 加载验证结果
    logger.info("步骤1: 加载验证结果...")
    original_preferences = load_verification_results(args.verification_dir)
    
    if not original_preferences:
        logger.error("没有找到任何验证结果文件")
        return
    
    # 2. 过滤高质量样本
    logger.info("步骤2: 过滤高质量样本...")
    filtered_preferences = filter_high_quality_samples(
        original_preferences,
        min_score_diff=args.min_score_diff,
        min_confidence=args.min_confidence
    )
    
    if not filtered_preferences:
        logger.error("过滤后没有剩余样本，请降低过滤阈值")
        return
    
    # 3. 使用验证指标增强数据
    logger.info("步骤3: 使用验证指标增强数据...")
    augmented_preferences = augment_with_verification_metrics(filtered_preferences)
    
    # 4. 平衡数据集
    logger.info("步骤4: 平衡数据集...")
    balanced_preferences = balance_dataset(augmented_preferences, args.max_samples)
    
    # 5. 创建训练数据集
    logger.info("步骤5: 创建训练数据集...")
    dataset = create_ccpo_dataset(balanced_preferences, args.test_ratio)
    
    # 6. 保存数据集
    logger.info("步骤6: 保存数据集...")
    dataset.save_to_disk(output_dir / "dataset")
    
    # 7. 保存原始偏好数据（供调试使用）
    with open(output_dir / "processed_preferences.json", 'w', encoding='utf-8') as f:
        json.dump(balanced_preferences, f, indent=2, ensure_ascii=False)
    
    # 8. 生成和保存统计信息
    logger.info("步骤7: 生成统计信息...")
    stats = generate_statistics(original_preferences, balanced_preferences)
    
    with open(output_dir / "processing_statistics.json", 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    # 输出总结
    logger.info("=== 数据处理完成 ===")
    logger.info(f"原始样本数: {stats['original_samples']}")
    logger.info(f"最终样本数: {stats['final_samples']}")
    logger.info(f"保留比例: {stats['filter_ratio']:.3f}")
    logger.info(f"训练集大小: {len(dataset['train_prefs'])}")
    logger.info(f"测试集大小: {len(dataset['test_prefs'])}")
    logger.info(f"平均分数差异: {stats['score_statistics']['score_diff_mean']:.3f}")
    logger.info(f"高质量样本比例: {stats['quality_distribution']['high_quality_ratio']:.3f}")
    logger.info(f"数据集已保存到: {output_dir}")

if __name__ == "__main__":
    main()