#!/usr/bin/env python3
"""
CCPO数学数据集构建器 - 简化版
直接使用MetaMath的response和OlympiadBench的final_answer
"""

import json
import argparse
import logging
import random
from typing import Dict, List, Any
from pathlib import Path
from datasets import load_dataset, Dataset
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="构建CCPO数学数据集 - 简化版")
    parser.add_argument("--output_path", type=str, default="/data/jiacheng/dylan/iclr2026/ccpo_math_dataset")
    parser.add_argument("--target_size", type=int, default=60000)
    parser.add_argument("--test_split_ratio", type=float, default=0.1)
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_dataset_id", type=str, default="UCLA-AGI/ccpo-math-60k")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--verify_samples", type=int, default=10)
    parser.add_argument("--include_olympiad", action="store_true", 
                       help="包含OlympiadBench等竞赛题")
    return parser.parse_args()


class SimpleCCPODatasetBuilder:
    """简化的CCPO数学数据集构建器 - 直接使用现成的answer字段"""
    
    def __init__(self, args):
        self.args = args
        
        # 统计信息
        self.processing_stats = {
            'total_processed': 0,
            'metamath_processed': 0,
            'olympiad_processed': 0,
            'final_count': 0
        }
    
    def build_dataset(self, target_size: int = 60000) -> Dataset:
        """构建CCPO数学数据集"""
        logger.info(f"🚀 开始构建CCPO数学数据集（简化版）")
        logger.info(f"   目标大小: {target_size:,}")
        logger.info(f"   包含竞赛题: {self.args.include_olympiad}")
        
        all_problems = []
        
        # 定义数据源配置
        data_sources = self._get_data_source_config(target_size)
        
        for source_config in data_sources:
            logger.info(f"\n📊 处理数据源: {source_config['name']}")
            logger.info(f"   目标数量: {source_config['target']:,}")
            
            try:
                problems = source_config['processor'](source_config['target'])
                all_problems.extend(problems)
                logger.info(f"   ✅ 获得问题: {len(problems):,}")
                
            except Exception as e:
                logger.error(f"   ❌ 处理失败: {e}")
                if self.args.debug:
                    import traceback
                    traceback.print_exc()
        
        # 随机打乱并截取目标大小
        random.shuffle(all_problems)
        final_problems = all_problems[:target_size]
        self.processing_stats['final_count'] = len(final_problems)
        
        # 转换为Dataset
        dataset = self._convert_to_dataset(final_problems)
        
        # 输出统计报告
        self._print_final_stats()
        
        return dataset
    
    def _get_data_source_config(self, target_size: int) -> List[Dict[str, Any]]:
        """获取数据源配置"""
        configs = [
            {
                'name': 'metamath',
                'target': int(target_size * 0.7),  # 70%
                'processor': self._process_metamath,
            }
        ]
        
        if self.args.include_olympiad:
            configs.insert(0, {
                'name': 'olympiad_bench',
                'target': int(target_size * 0.3),  # 30%
                'processor': self._process_olympiad_bench,
            })
            # 调整MetaMath比例
            configs[1]['target'] = int(target_size * 0.7)
        
        return configs
    
    def _process_metamath(self, target_count: int) -> List[Dict[str, Any]]:
        """处理MetaMath数据集 - 直接使用response字段"""
        logger.info(f"   🔬 加载MetaMath数据集...")
        
        try:
            dataset = load_dataset("meta-math/MetaMathQA-40K", split="train")
            logger.info(f"   ✅ MetaMath加载成功: {len(dataset)} 条记录")
            
            if self.args.debug:
                dataset = dataset.select(range(min(2000, len(dataset))))
                logger.info(f"   🔧 调试模式，限制为: {len(dataset)} 条记录")
            
            problems = []
            
            for item in dataset:
                if len(problems) >= target_count:
                    break
                
                self.processing_stats['total_processed'] += 1
                
                question = item['query']
                response = item['response']  # 直接使用response作为answer
                
                # 基本有效性检查
                if not question or not response or len(question) < 10:
                    continue
                
                problems.append({
                    'question': question,
                    'answer': response,  # 直接使用完整的response
                    'source': 'metamath'
                })
                
                self.processing_stats['metamath_processed'] += 1
                
                # 进度报告
                if len(problems) % 1000 == 0:
                    logger.info(f"     MetaMath处理进度: {len(problems)}")
            
            logger.info(f"   ✅ MetaMath处理完成: {len(problems)} 个问题")
            return problems
            
        except Exception as e:
            logger.error(f"   ❌ MetaMath处理失败: {e}")
            if self.args.debug:
                import traceback
                traceback.print_exc()
            return []
    
    def _process_olympiad_bench(self, target_count: int) -> List[Dict[str, Any]]:
        """处理OlympiadBench数据集 - 直接使用final_answer字段"""
        logger.info(f"   🏆 处理OlympiadBench竞赛题...")
        
        try:
            dataset = load_dataset("lmms-lab/OlympiadBench", split="test_en")
            if self.args.debug:
                dataset = dataset.select(range(min(500, len(dataset))))
            
            logger.info(f"   ✅ OlympiadBench加载成功: {len(dataset)} 条记录")
            
            problems = []
            
            for item in dataset:
                if len(problems) >= target_count:
                    break
                
                self.processing_stats['total_processed'] += 1
                
                # 使用question字段作为问题
                question = item.get('question', '')
                
                # 直接使用final_answer字段
                final_answers = item.get('final_answer', [])
                if not final_answers or len(final_answers) == 0:
                    continue
                
                # 将final_answer列表转换为字符串
                if isinstance(final_answers, list):
                    answer = str(final_answers[0]) if final_answers else ""
                else:
                    answer = str(final_answers)
                
                # 基本有效性检查
                if not question or not answer or len(question) < 10:
                    continue
                
                problems.append({
                    'question': question,
                    'answer': answer,  # 直接使用final_answer
                    'source': 'olympiad_bench'
                })
                
                self.processing_stats['olympiad_processed'] += 1
                
                # 进度报告
                if len(problems) % 100 == 0:
                    logger.info(f"     OlympiadBench处理进度: {len(problems)}")
            
            logger.info(f"   ✅ OlympiadBench处理完成: {len(problems)} 个问题")
            return problems
            
        except Exception as e:
            logger.error(f"   ❌ OlympiadBench处理失败: {e}")
            if self.args.debug:
                import traceback
                traceback.print_exc()
            return []
    
    def _convert_to_dataset(self, problems: List[Dict[str, Any]]) -> Dataset:
        """转换为Dataset格式"""
        if not problems:
            raise ValueError("没有问题可转换为数据集")
        
        dataset_dict = {
            'prompt': [p['question'] for p in problems],
            'answer': [p['answer'] for p in problems],  # 统一的answer字段
            'source': [p['source'] for p in problems],
        }
        
        return Dataset.from_dict(dataset_dict)
    
    def _save_as_huggingface_format(self, dataset: Dataset, output_path: str):
        """保存为HuggingFace兼容格式"""
        logger.info(f"💾 保存为HuggingFace格式: {output_path}")
        
        # 创建输出目录
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存为Arrow格式的parquet文件
        dataset_file = output_dir / "dataset.parquet"
        dataset.to_parquet(dataset_file)
        
        # 创建dataset_info.json
        dataset_info = {
            "citation": "",
            "description": "CCPO Math Dataset - MetaMath + OlympiadBench",
            "features": {
                "prompt": {"dtype": "string", "_type": "Value"},
                "answer": {"dtype": "string", "_type": "Value"},
                "source": {"dtype": "string", "_type": "Value"},
            },
            "homepage": "",
            "license": "",
            "size_in_bytes": dataset_file.stat().st_size,
            "splits": {
                "train": {
                    "name": "train",
                    "num_bytes": dataset_file.stat().st_size,
                    "num_examples": len(dataset),
                    "shard_lengths": [len(dataset)],
                    "dataset_name": "ccpo_math_dataset"
                }
            },
            "version": {"version_str": "1.0.0", "major": 1, "minor": 0, "patch": 0}
        }
        
        with open(output_dir / "dataset_info.json", 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        # 创建state.json
        state_info = {
            "_data_files": [{"filename": "dataset.parquet"}],
            "_fingerprint": "ccpo_math_dataset_v1",
            "_format_columns": None,
            "_format_kwargs": {},
            "_format_type": None,
            "_output_all_columns": False,
            "_split": "train"
        }
        
        with open(output_dir / "state.json", 'w') as f:
            json.dump(state_info, f, indent=2)
        
        logger.info(f"✅ HuggingFace格式保存完成")
        logger.info(f"   数据文件: {dataset_file}")
        logger.info(f"   元数据: dataset_info.json, state.json")
        
        return output_dir
    
    def _print_final_stats(self):
        """打印最终统计"""
        stats = self.processing_stats
        
        print(f"\n" + "="*80)
        print(f"📊 CCPO数学数据集构建完成 (简化版)")
        print(f"="*80)
        print(f"🎯 最终数据集大小: {stats['final_count']:,}")
        print(f"📈 处理统计:")
        print(f"   总处理样本: {stats['total_processed']:,}")
        print(f"   MetaMath处理: {stats['metamath_processed']:,}")
        print(f"   OlympiadBench处理: {stats['olympiad_processed']:,}")
        
        print(f"\n✅ 数据源特色:")
        print(f"   ✓ MetaMath: 直接使用response字段作为answer")
        print(f"   ✓ OlympiadBench: 直接使用final_answer字段作为answer")
        print(f"   ✓ 适合CCPO训练的统一answer格式")
        print(f"   ✓ 无需复杂的答案提取逻辑")
        print(f"="*80)


def main():
    """主函数"""
    import time
    
    args = parse_arguments()
    
    print(f"🚀 CCPO数学数据集构建器 (简化版)")
    print(f"="*60)
    print(f"目标大小: {args.target_size:,}")
    print(f"包含竞赛题: {args.include_olympiad}")
    print(f"输出路径: {args.output_path}")
    
    # 构建数据集
    builder = SimpleCCPODatasetBuilder(args)
    start_time = time.time()
    
    try:
        dataset = builder.build_dataset(target_size=args.target_size)
        
        build_time = time.time() - start_time
        print(f"\n⏱️  总构建时间: {build_time:.1f}秒")
        
        # 保存为HuggingFace兼容格式
        logger.info(f"💾 保存数据集...")
        saved_path = builder._save_as_huggingface_format(dataset, args.output_path)
        
        # 创建分割
        if args.test_split_ratio > 0:
            logger.info(f"🔄 创建训练/测试分割...")
            split_dataset = dataset.train_test_split(test_size=args.test_split_ratio, seed=42)
            
            # 分别保存训练和测试集
            train_path = f"{args.output_path}_train"
            test_path = f"{args.output_path}_test"
            
            builder._save_as_huggingface_format(split_dataset['train'], train_path)
            builder._save_as_huggingface_format(split_dataset['test'], test_path)
            
            logger.info(f"✅ 分割保存完成:")
            logger.info(f"   训练集: {train_path} ({len(split_dataset['train'])} 样本)")
            logger.info(f"   测试集: {test_path} ({len(split_dataset['test'])} 样本)")
        
        # 推送到Hub（可选）
        if args.push_to_hub:
            logger.info(f"🔄 推送到HuggingFace Hub...")
            try:
                dataset.push_to_hub(args.hub_dataset_id)
                logger.info(f"✅ 成功推送: {args.hub_dataset_id}")
            except Exception as e:
                logger.error(f"❌ 推送失败: {e}")
        
        # 验证可以用load_dataset加载
        logger.info(f"🔍 验证数据集可加载性...")
        try:
            from datasets import load_dataset
            test_dataset = load_dataset(str(saved_path), split="train")
            logger.info(f"✅ 验证成功: 可以使用 load_dataset('{saved_path}', split='train') 加载")
            logger.info(f"   加载的数据集大小: {len(test_dataset)}")
            logger.info(f"   列名: {test_dataset.column_names}")
        except Exception as e:
            logger.error(f"❌ 验证失败: {e}")
            logger.info(f"回退方案: 使用 load_from_disk('{saved_path}') 加载")
        
        # 验证样本
        if args.verify_samples > 0:
            print(f"\n🔍 质量验证样本:")
            samples = dataset.select(range(min(args.verify_samples, len(dataset))))
            
            for i, sample in enumerate(samples):
                print(f"\n样本 {i+1}:")
                print(f"  问题: {sample['prompt'][:120]}...")
                print(f"  答案: {sample['answer'][:120]}...")  # 显示answer字段前120个字符
                print(f"  来源: {sample['source']}")
        
        print(f"\n🎉 CCPO数学数据集构建成功! (简化版)")
        print(f"数据集位置: {saved_path}")
        print(f"总问题数: {len(dataset):,}")
        print(f"\n📋 使用方法:")
        print(f"from datasets import load_dataset")
        print(f"dataset = load_dataset('{saved_path}', split='train')")
        print(f"\n数据格式:")
        print(f"- prompt: 问题文本")
        print(f"- answer: MetaMath的response或OlympiadBench的final_answer")
        print(f"- source: 数据来源 (metamath/olympiad_bench)")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ 构建失败: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())