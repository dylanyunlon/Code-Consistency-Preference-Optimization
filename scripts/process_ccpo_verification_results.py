#!/usr/bin/env python3
"""
CCPO验证结果处理脚本 - 对话格式版本
将code_verified_rank.py的验证结果转换为CCPO训练需要的对话格式数据
"""

import json
import numpy as np
import pandas as pd
import argparse
import os
from typing import Dict, List, Tuple, Any
from datasets import Dataset, DatasetDict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="处理CCPO验证结果")
    parser.add_argument("--input_dir", type=str, required=True, help="输入目录（generated数据）")
    parser.add_argument("--ranking_dir", type=str, required=True, help="排名结果目录")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录")
    parser.add_argument("--data_frac", type=int, default=0, help="数据分片编号")
    parser.add_argument("--pairs", type=int, default=5, help="每个问题的推理过程数")
    parser.add_argument("--score_threshold", type=float, default=5.0, help="CCPO分数差异阈值")
    parser.add_argument("--confidence_threshold", type=float, default=0.1, help="置信度差异阈值")
    parser.add_argument("--output_format", type=str, default="conversation", 
                      choices=["conversation", "string"], help="输出格式：conversation（对话格式）或string（字符串格式）")
    return parser.parse_args()

class CCPOResultProcessor:
    """CCPO验证结果处理器 - 支持对话格式输出"""
    
    def __init__(self, args):
        self.args = args
        self.stats = {
            'total_samples': 0,
            'high_quality_pairs': 0,
            'low_quality_pairs': 0,
            'skipped_samples': 0,
            'avg_score_difference': 0.0
        }
    
    def load_data(self) -> Tuple[List[Dict], List[str], List[List[str]], np.ndarray]:
        """加载生成的数据和CCPO验证结果"""
        print(f"📂 加载数据...")
        
        # 1. 加载元数据
        metadata_file = f"{self.args.input_dir}/metadata_{self.args.data_frac}.json"
        if not os.path.exists(metadata_file):
            raise FileNotFoundError(f"元数据文件不存在: {metadata_file}")
        
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        data_items = metadata['data_items']
        questions = [item['prompt'] for item in data_items]
        ground_truths = [item['answer'] for item in data_items]
        
        print(f"✅ 加载元数据: {len(data_items)} 条")
        
        # 2. 加载生成的推理过程
        all_responses = []
        for i in range(self.args.pairs):
            response_file = f"{self.args.input_dir}/responses_{i}.json"
            if not os.path.exists(response_file):
                raise FileNotFoundError(f"响应文件不存在: {response_file}")
            
            with open(response_file, 'r', encoding='utf-8') as f:
                responses = json.load(f)
                all_responses.append(responses)
        
        # 转置为每个问题的推理过程列表
        reasoning_processes = list(zip(*all_responses))
        print(f"✅ 加载推理过程: {len(reasoning_processes)} 个问题，每个 {self.args.pairs} 个推理过程")
        
        # 3. 加载CCPO验证结果
        ranking_file = f"{self.args.ranking_dir}/ccpo_0_{self.args.data_frac}.npy"
        if not os.path.exists(ranking_file):
            raise FileNotFoundError(f"CCPO排名文件不存在: {ranking_file}")
        
        ccpo_scores = np.load(ranking_file)
        print(f"✅ 加载CCPO分数: {ccpo_scores.shape}")
        
        return data_items, questions, reasoning_processes, ccpo_scores
    
    def create_preference_pairs(
        self, 
        questions: List[str], 
        reasoning_processes: List[List[str]], 
        ccpo_scores: np.ndarray
    ) -> List[Dict[str, Any]]:
        """基于CCPO分数创建偏好对"""
        print(f"🔄 创建偏好对...")
        print(f"   输出格式: {self.args.output_format}")
        
        preference_pairs = []
        
        for idx, (question, processes, scores) in enumerate(zip(questions, reasoning_processes, ccpo_scores)):
            self.stats['total_samples'] += 1
            
            # 找到最高分和最低分的推理过程
            max_idx = np.argmax(scores)
            min_idx = np.argmin(scores)
            
            max_score = scores[max_idx]
            min_score = scores[min_idx]
            score_diff = max_score - min_score
            
            # 检查分数差异是否足够大
            if score_diff < self.args.score_threshold:
                self.stats['skipped_samples'] += 1
                continue
            
            chosen_process = processes[max_idx]
            rejected_process = processes[min_idx]
            
            # 计算选择概率（基于CCPO分数）
            chosen_prob = self._calculate_preference_probability(max_score, min_score)
            
            if self.args.output_format == "conversation":
                # 对话格式：每个都是消息列表
                preference_pair = {
                    'chosen': [
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": chosen_process}
                    ],
                    'rejected': [
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": rejected_process}
                    ],
                    'chosen_probs': chosen_prob,
                    'chosen_probs_win': chosen_prob,
                    'chosen_probs_lose': 1.0 - chosen_prob,
                    'chosen_score': float(max_score),
                    'rejected_score': float(min_score),
                    'score_difference': float(score_diff),
                    'chosen_verification_score': self._normalize_score(max_score),
                    'rejected_verification_score': self._normalize_score(min_score),
                    'original_index': idx,
                    'ccpo_quality': 'high' if score_diff > 10.0 else 'medium'
                }
            else:
                # 字符串格式（原来的格式）
                preference_pair = {
                    'prompt': question,
                    'chosen': chosen_process,
                    'rejected': rejected_process,
                    'chosen_probs': chosen_prob,
                    'chosen_probs_win': chosen_prob,
                    'chosen_probs_lose': 1.0 - chosen_prob,
                    'chosen_score': float(max_score),
                    'rejected_score': float(min_score),
                    'score_difference': float(score_diff),
                    'chosen_verification_score': self._normalize_score(max_score),
                    'rejected_verification_score': self._normalize_score(min_score),
                    'original_index': idx,
                    'ccpo_quality': 'high' if score_diff > 10.0 else 'medium'
                }
            
            preference_pairs.append(preference_pair)
            
            if max_score > 0:
                self.stats['high_quality_pairs'] += 1
            else:
                self.stats['low_quality_pairs'] += 1
            
            self.stats['avg_score_difference'] += score_diff
        
        if preference_pairs:
            self.stats['avg_score_difference'] /= len(preference_pairs)
        
        print(f"✅ 创建偏好对完成: {len(preference_pairs)} 对")
        return preference_pairs
    
    def _calculate_preference_probability(self, chosen_score: float, rejected_score: float) -> float:
        """计算选择概率"""
        # 使用sigmoid函数将分数差异转换为概率
        score_diff = chosen_score - rejected_score
        
        # 确保概率在合理范围内
        if score_diff > 20:
            return 0.95
        elif score_diff > 10:
            return 0.85
        elif score_diff > 5:
            return 0.75
        elif score_diff > 0:
            return 0.65
        else:
            return 0.55  # 即使分数相近，也要有轻微偏好
    
    def _normalize_score(self, score: float) -> float:
        """归一化分数到[0,1]范围"""
        # CCPO分数通常在[-20, 50]范围内
        normalized = (score + 20) / 70  # 映射到[0,1]
        return max(0.0, min(1.0, normalized))
    
    def split_dataset(self, preference_pairs: List[Dict[str, Any]]) -> Tuple[List[Dict], List[Dict]]:
        """分割数据集为训练集和测试集"""
        total_size = len(preference_pairs)
        train_size = int(total_size * 0.9)  # 90%用于训练
        
        # 随机打乱
        np.random.seed(42)
        indices = np.random.permutation(total_size)
        
        train_indices = indices[:train_size]
        test_indices = indices[train_size:]
        
        train_pairs = [preference_pairs[i] for i in train_indices]
        test_pairs = [preference_pairs[i] for i in test_indices]
        
        print(f"📊 数据集分割: 训练集 {len(train_pairs)}, 测试集 {len(test_pairs)}")
        return train_pairs, test_pairs
    
    def save_dataset(self, train_pairs: List[Dict], test_pairs: List[Dict]):
        """保存数据集为多种格式"""
        print(f"💾 保存数据集...")
        
        # 创建输出目录
        os.makedirs(self.args.output_dir, exist_ok=True)
        
        # 1. 保存为JSON Lines格式（便于检查）
        train_jsonl = f"{self.args.output_dir}/train_prefs.jsonl"
        test_jsonl = f"{self.args.output_dir}/test_prefs.jsonl"
        
        with open(train_jsonl, 'w', encoding='utf-8') as f:
            for pair in train_pairs:
                f.write(json.dumps(pair, ensure_ascii=False) + '\n')
        
        with open(test_jsonl, 'w', encoding='utf-8') as f:
            for pair in test_pairs:
                f.write(json.dumps(pair, ensure_ascii=False) + '\n')
        
        print(f"✅ JSONL格式保存完成:")
        print(f"   - 训练集: {train_jsonl}")
        print(f"   - 测试集: {test_jsonl}")
        
        # 2. 保存为Parquet格式（高效加载）
        train_parquet = f"{self.args.output_dir}/train_prefs.parquet"
        test_parquet = f"{self.args.output_dir}/test_prefs.parquet"
        
        import pandas as pd
        pd.DataFrame(train_pairs).to_parquet(train_parquet, index=False)
        pd.DataFrame(test_pairs).to_parquet(test_parquet, index=False)
        
        print(f"✅ Parquet格式保存完成:")
        print(f"   - 训练集: {train_parquet}")
        print(f"   - 测试集: {test_parquet}")
        
        # 3. 创建数据集信息文件
        dataset_info = {
            "description": "CCPO Code-Verified Preference Dataset",
            "format": self.args.output_format,
            "train_size": len(train_pairs),
            "test_size": len(test_pairs),
            "features": {
                "chosen": "conversation list" if self.args.output_format == "conversation" else "string",
                "rejected": "conversation list" if self.args.output_format == "conversation" else "string",
                "chosen_probs": "float",
                "chosen_probs_win": "float",
                "chosen_probs_lose": "float",
                "chosen_score": "float",
                "rejected_score": "float",
                "score_difference": "float"
            },
            "splits": {
                "train": f"train_prefs.jsonl",
                "test": f"test_prefs.jsonl"
            },
            "formats": ["jsonl", "parquet"],
            "usage": {
                "jsonl": "Direct loading with datasets.load_dataset('json', data_files='path')",
                "parquet": "Loading with datasets.load_dataset('parquet', data_files='path')"
            }
        }
        
        info_file = f"{self.args.output_dir}/dataset_info.json"
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(dataset_info, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 数据集信息文件: {info_file}")
        
        # 保存处理统计信息
        stats_file = f"{self.args.output_dir}/processing_stats.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 处理统计文件: {stats_file}")
    
    def print_stats(self):
        """打印处理统计信息"""
        print(f"\n📈 CCPO数据处理统计报告")
        print(f"========================")
        print(f"总样本数: {self.stats['total_samples']}")
        print(f"高质量偏好对: {self.stats['high_quality_pairs']}")
        print(f"低质量偏好对: {self.stats['low_quality_pairs']}")
        print(f"跳过样本数: {self.stats['skipped_samples']}")
        print(f"平均分数差异: {self.stats['avg_score_difference']:.3f}")
        
        if self.stats['total_samples'] > 0:
            quality_rate = self.stats['high_quality_pairs'] / (self.stats['high_quality_pairs'] + self.stats['low_quality_pairs'])
            print(f"高质量偏好对比例: {quality_rate:.2%}")
            skip_rate = self.stats['skipped_samples'] / self.stats['total_samples']
            print(f"跳过样本比例: {skip_rate:.2%}")

def main():
    args = parse_arguments()
    
    print(f"🚀 CCPO验证结果处理器")
    print(f"===================")
    print(f"输入目录: {args.input_dir}")
    print(f"排名目录: {args.ranking_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"输出格式: {args.output_format}")
    print(f"分数阈值: {args.score_threshold}")
    
    # 初始化处理器
    processor = CCPOResultProcessor(args)
    
    try:
        # 加载数据
        data_items, questions, reasoning_processes, ccpo_scores = processor.load_data()
        
        # 创建偏好对
        preference_pairs = processor.create_preference_pairs(questions, reasoning_processes, ccpo_scores)
        
        if not preference_pairs:
            print("❌ 没有创建任何偏好对，请检查分数阈值设置")
            return 1
        
        # 分割数据集
        train_pairs, test_pairs = processor.split_dataset(preference_pairs)
        
        # 保存数据集
        processor.save_dataset(train_pairs, test_pairs)
        
        # 打印统计信息
        processor.print_stats()
        
        print(f"\n✅ CCPO数据处理完成!")
        print(f"🎯 Architecture B数据准备就绪，可以开始训练")
        
        if args.output_format == "conversation":
            print(f"\n📋 对话格式使用方法:")
            print(f"   JSONL格式: dataset_mixer: {{'{args.output_dir}/train_prefs.jsonl': 1.0}}")
            print(f"   数据格式: chosen/rejected 字段包含对话消息列表")
        else:
            print(f"\n📋 字符串格式使用方法:")
            print(f"   JSONL格式: dataset_mixer: {{'{args.output_dir}/train_prefs.jsonl': 1.0}}")
            print(f"   数据格式: prompt/chosen/rejected 字段包含文本字符串")
        
        return 0
        
    except Exception as e:
        logger.error(f"处理过程中出现错误: {e}")
        return 1

if __name__ == "__main__":
    exit(main())