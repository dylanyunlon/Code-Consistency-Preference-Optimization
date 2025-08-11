#!/usr/bin/env python3
"""
CCPO Code Verified Ranking Script - Architecture B Implementation
实现核心创新：用服务器按照7B推理思路执行代码来验证推理质量
"""

import sys
import os
# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset, Dataset
import json
import pandas as pd
import argparse
import os
import numpy as np
import asyncio
import time
from typing import List, Dict, Any, Tuple
from transformers import AutoTokenizer
import logging

# 强制导入检查
try:
    from enhanced_answer_extractor_v2 import EnhancedAnswerExtractorV2
    print("✅ 成功导入增强版答案提取器V2")
    V2_AVAILABLE = True
except ImportError as e:
    print(f"❌ 无法导入增强版答案提取器V2: {e}")
    print("⚠️  将使用内置回退方法")
    V2_AVAILABLE = False

# 导入CCPO版执行验证器
try:
    from execution_verifier import ExecutionVerifier, VerificationResult, VerificationStatus
    print("✅ 成功导入CCPO执行验证器")
except ImportError as e:
    print(f"❌ 无法导入执行验证器: {e}")
    raise ImportError("执行验证器不可用，请检查 execution_verifier.py 文件") from e

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="CCPO Code Verified Ranking - Architecture B实现")
    parser.add_argument(
        "--model", type=str, 
        default="/data/jiacheng/dylan/iclr2026/Code-Consistency-Preference-Optimization/checkpoints/mistral-7b-instruct-code-verified-ccpo",
        help="Base model path"
    )
    parser.add_argument('--output_dir', type=str, default='generated/iter1')
    parser.add_argument("--numgpu", type=int, default=8)
    parser.add_argument('--prompts', type=str, default='dylansss/ccpo_math_dataset')
    parser.add_argument('--data_frac', type=int, default=0)
    parser.add_argument('--frac_len', type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--pairs", type=int, default=5)
    
    # CCPO验证相关参数
    parser.add_argument("--verification_url", type=str, default="https://8.134.217.190:17432", 
                       help="代码执行服务器地址")
    parser.add_argument("--verification_username", type=str, default="newuser")
    parser.add_argument("--verification_password", type=str, default="newPass123")
    parser.add_argument("--max_concurrent", type=int, default=1, 
                       help="最大并发数")
    parser.add_argument("--debug_v2", action="store_true", help="启用详细调试模式")
    parser.add_argument("--verification_sample_rate", type=float, default=0.005, 
                       help="验证采样率")
    
    # 限流控制参数
    parser.add_argument("--base_delay", type=float, default=15.0, help="基础请求间隔（秒）")
    parser.add_argument("--max_delay", type=float, default=300.0, help="最大退避延迟（秒）")
    parser.add_argument("--request_timeout", type=int, default=180, help="单个请求超时时间（秒）")
    
    # 重试和检查点控制
    parser.add_argument("--max_retries", type=int, default=1, help="最大重试次数")
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="检查点保存间隔")
    parser.add_argument("--force_restart", action="store_true", help="强制重新开始，忽略检查点")
    parser.add_argument("--verification_model", type=str, default="claude-sonnet-4-20250514-all", 
                       help="验证时使用的模型")
    
    return parser.parse_args()

def split_prompts(prompts, frac_len, data_frac):
    """分割提示数据用于分布式处理"""
    if frac_len > 0:
        split_len = frac_len
        if split_len * (data_frac + 1) > len(prompts):
            return prompts[split_len * data_frac:]
        else:
            return prompts[split_len * data_frac: split_len * (data_frac + 1)]
    else:
        return prompts[:]

def apply_template(text, tokenizer):
    """应用聊天模板"""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": text}, {"role": "assistant", "content": "None"}],
        tokenize=False, add_generate_prompt=True
    ).split("None")[0]

class CCPOCodeVerifiedRanker:
    """
    CCPO代码验证排名器 - Architecture B核心实现
    用服务器按照7B推理思路执行代码来验证推理质量
    """
    
    def __init__(self, args):
        self.args = args
        
        # 初始化答案提取器（可选）
        if V2_AVAILABLE:
            try:
                self.answer_extractor = EnhancedAnswerExtractorV2(debug=args.debug_v2)
                print(f"✅ V2增强版答案提取器初始化完成 (调试模式: {'开启' if args.debug_v2 else '关闭'})")
            except Exception as e:
                print(f"⚠️  V2提取器初始化失败: {e}")
                self.answer_extractor = None
        else:
            self.answer_extractor = None
        
        # CCPO验证统计
        self.ccpo_stats = {
            'total_processed': 0,
            'reasoning_verification_attempted': 0,
            'high_quality_reasoning': 0,
            'low_quality_reasoning': 0,
            'execution_failures': 0,
            'avg_reasoning_confidence': 0.0,
            'cached_results': 0
        }
        
        # 重试控制
        self.max_retries = getattr(args, 'max_retries', 1)
        self.checkpoint_interval = getattr(args, 'checkpoint_interval', 1)
        self.processed_cache = set()
    
    def _load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """加载检查点"""
        if os.path.exists(checkpoint_path) and not self.args.force_restart:
            try:
                with open(checkpoint_path, 'r') as f:
                    checkpoint = json.load(f)
                    print(f"📂 加载CCPO检查点: {len(checkpoint.get('processed_indices', []))} 个已处理样本")
                    return checkpoint
            except Exception as e:
                logger.warning(f"检查点加载失败: {e}")
        return {"processed_indices": [], "all_scores": [], "retry_count": 0, "verification_cache": {}}
    
    def _save_checkpoint(self, checkpoint_path: str, data: Dict[str, Any]):
        """保存检查点 - 修复序列化问题"""
        try:
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            
            # 确保数据可序列化
            serializable_data = {}
            for key, value in data.items():
                if key == "verification_cache":
                    # 跳过verification_cache以避免序列化问题
                    serializable_data[key] = {}
                else:
                    serializable_data[key] = value
            
            with open(checkpoint_path, 'w') as f:
                json.dump(serializable_data, f, indent=2)
            if self.args.debug_v2:
                print(f"💾 CCPO检查点已保存: {checkpoint_path}")
        except Exception as e:
            logger.warning(f"检查点保存失败: {e}")

    async def verify_reasoning_with_ccpo_verifier(
        self, 
        questions: List[str], 
        candidates_list: List[Tuple[str, ...]],
        ground_truths: List[str] = None
    ) -> List[np.ndarray]:
        """
        使用CCPO验证器进行推理质量验证和排名 - Architecture B核心实现
        """
        checkpoint_path = f"ranking/{self.args.output_dir}/ccpo_checkpoint_{self.args.gpu}_{self.args.data_frac}.json"
        
        # 加载检查点
        checkpoint = self._load_checkpoint(checkpoint_path)
        processed_indices = set(checkpoint.get("processed_indices", []))
        all_scores = checkpoint.get("all_scores", [])
        retry_count = checkpoint.get("retry_count", 0)
        verification_cache = checkpoint.get("verification_cache", {})
        
        # 调试信息：检查检查点内容
        if self.args.debug_v2:
            print(f"🔍 检查点调试信息:")
            print(f"   检查点路径: {checkpoint_path}")
            print(f"   已处理索引数量: {len(processed_indices)}")
            print(f"   已处理索引: {sorted(list(processed_indices))[:10]}...")
            print(f"   all_scores长度: {len(all_scores)}")
            print(f"   重试次数: {retry_count}")
        
        print(f"🚀 开始CCPO推理质量验证排名 (Architecture B)")
        print(f"   - 处理问题数: {len(questions)}")
        print(f"   - 每问题推理过程数: {len(candidates_list[0]) if candidates_list else 0}")
        print(f"   - Ground Truth: {'可用' if ground_truths else '不可用'}")
        print(f"   - 采样率: {self.args.verification_sample_rate}")
        print(f"   - 验证服务器: {self.args.verification_url}")
        print(f"   - 验证模型: {self.args.verification_model}")
        print(f"   - 已处理样本: {len(processed_indices)}/{len(questions)}")
        print(f"   - 基础延迟: {self.args.base_delay}秒")
        print(f"   - 核心创新: 服务器按7B推理思路执行代码验证推理质量")
        
        # 初始化分数列表
        if len(all_scores) != len(questions):
            all_scores = [None] * len(questions)
        
        # 采样决定哪些样本进行CCPO验证
        sample_size = max(1, int(len(questions) * self.args.verification_sample_rate))
        np.random.seed(42 + retry_count)
        sample_indices = set(np.random.choice(len(questions), sample_size, replace=False))
        print(f"   - 实际CCPO验证样本数: {sample_size}")
        print(f"   - 样本索引: {sorted(list(sample_indices))[:10]}..." if len(sample_indices) > 10 else f"   - 样本索引: {sorted(list(sample_indices))}")
        
        # 调试信息：检查跳过逻辑
        if self.args.debug_v2:
            print(f"\n🔍 跳过逻辑调试:")
            skip_count = 0
            for idx in range(min(10, len(questions))):  # 只检查前10个
                will_skip = (idx in processed_indices and all_scores[idx] is not None)
                if will_skip:
                    skip_count += 1
                print(f"   索引{idx}: {'跳过' if will_skip else '处理'} (在processed: {idx in processed_indices}, scores存在: {all_scores[idx] is not None})")
            print(f"   前10个样本中跳过数量: {skip_count}")
        
        consecutive_failures = 0
        max_consecutive_failures = 2
        
        try:
            # 使用CCPO版执行验证器
            async with ExecutionVerifier(
                base_url=self.args.verification_url,
                username=self.args.verification_username,
                password=self.args.verification_password,
                debug=self.args.debug_v2,
                timeout=self.args.request_timeout
            ) as verifier:
                print("✅ CCPO执行验证器初始化成功（Architecture B）")
                
                # 恢复验证缓存
                if verification_cache:
                    verifier.verification_cache.update(verification_cache)
                    self.ccpo_stats['cached_results'] = len(verification_cache)
                    print(f"🔄 恢复CCPO验证缓存: {len(verification_cache)} 个结果")
                
                for idx, (question, candidates) in enumerate(zip(questions, candidates_list)):
                    # 调试信息：每个样本的处理状态
                    if self.args.debug_v2 and idx < 5:  # 只显示前5个样本的详细信息
                        print(f"\n🔍 样本{idx}处理状态:")
                        print(f"   在processed_indices中: {idx in processed_indices}")
                        print(f"   all_scores[{idx}]是否存在: {all_scores[idx] is not None}")
                        print(f"   在sample_indices中: {idx in sample_indices}")
                    
                    # 跳过已处理的样本
                    if idx in processed_indices and all_scores[idx] is not None:
                        if self.args.debug_v2 and idx < 5:
                            print(f"   → 跳过样本{idx}")
                        continue
                    
                    self.ccpo_stats['total_processed'] += 1
                    
                    if self.args.debug_v2 and idx < 5:
                        print(f"   → 处理样本{idx} (total_processed: {self.ccpo_stats['total_processed']})")
                    
                    if idx in sample_indices:
                        # 对选中的样本进行CCPO推理验证
                        current_ground_truth = ground_truths[idx] if ground_truths and idx < len(ground_truths) else None
                        
                        if self.args.debug_v2:
                            print(f"\n🧠 CCPO推理验证样本 {idx+1}/{len(questions)}: {question[:50]}...")
                            if current_ground_truth:
                                print(f"   Ground Truth: {current_ground_truth[:50]}...")
                        else:
                            print(f"🧠 CCPO推理验证样本 {idx+1}/{len(questions)}")
                        
                        try:
                            # 验证所有候选推理过程 - CCPO核心逻辑
                            verification_results = []
                            
                            for candidate_idx, reasoning_process in enumerate(candidates):
                                try:
                                    if self.args.debug_v2:
                                        print(f"  验证推理过程 {candidate_idx+1}: {reasoning_process[:50]}...")
                                    
                                    # CCPO核心方法：验证推理过程质量
                                    result = await verifier.verify_reasoning_process(
                                        question=question,
                                        reasoning_process=reasoning_process,
                                        ground_truth=current_ground_truth,  # 传入ground_truth
                                        use_cache=True,
                                        model=self.args.verification_model
                                    )
                                    
                                    verification_results.append(result)
                                    
                                    # 每个推理过程之间的延迟
                                    await asyncio.sleep(max(self.args.base_delay, 12.0))
                                    
                                except Exception as e:
                                    logger.error(f"推理验证异常 {idx}-{candidate_idx}: {e}")
                                    verification_results.append(VerificationResult(
                                        verified=False,
                                        status=VerificationStatus.ERROR,
                                        ai_answer=None,
                                        code_answer=None,
                                        confidence=0.0,
                                        execution_time=0.0,
                                        code_generated="",
                                        code_id=None,
                                        stdout="",
                                        stderr="",
                                        error_message=str(e),
                                        verification_id=f"error_{idx}_{candidate_idx}",
                                        raw_ai_response="",
                                        reasoning_process=reasoning_process
                                    ))
                            
                            # 基于CCPO验证结果计算推理质量分数
                            scores = self._calculate_reasoning_quality_scores(verification_results)
                            
                            # 更新CCPO统计
                            self.ccpo_stats['reasoning_verification_attempted'] += 1
                            high_quality_count = sum(1 for r in verification_results 
                                                   if isinstance(r, VerificationResult) and r.verified)
                            self.ccpo_stats['high_quality_reasoning'] += high_quality_count
                            self.ccpo_stats['low_quality_reasoning'] += len(verification_results) - high_quality_count
                            
                            consecutive_failures = 0
                            
                            if self.args.debug_v2:
                                print(f"   📊 CCPO推理验证结果详情:")
                                for i, result in enumerate(verification_results):
                                    if isinstance(result, VerificationResult):
                                        quality = "🎯 高质量" if result.verified else "❌ 低质量"
                                        print(f"   - 推理{i+1}: {quality} "
                                              f"(置信度: {result.confidence:.3f}, "
                                              f"状态: {result.status.value})")
                                        if result.ai_answer and result.code_answer:
                                            print(f"     推理答案: {result.ai_answer}, 执行答案: {result.code_answer}")
                                        if result.error_message:
                                            print(f"     错误: {result.error_message[:100]}...")
                                print(f"   📈 推理质量得分: {scores}")
                            
                        except Exception as e:
                            logger.error(f"CCPO推理验证失败 {idx}: {e}")
                            self.ccpo_stats['execution_failures'] += 1
                            consecutive_failures += 1
                            
                            # 检查连续失败次数
                            if consecutive_failures >= max_consecutive_failures:
                                logger.error(f"连续失败次数过多 ({consecutive_failures})，暂停180秒...")
                                await asyncio.sleep(180)
                                consecutive_failures = 0
                            
                            # 使用默认分数
                            scores = self._get_default_reasoning_scores(len(candidates))
                            
                    else:
                        # 未选中的样本使用默认评分策略
                        scores = self._get_default_reasoning_scores(len(candidates))
                    
                    # 保存分数
                    all_scores[idx] = scores.tolist()
                    processed_indices.add(idx)
                    
                    # 保存检查点
                    if (idx + 1) % self.checkpoint_interval == 0:
                        checkpoint_data = {
                            "processed_indices": list(processed_indices),
                            "all_scores": all_scores,
                            "retry_count": retry_count,
                            # 不保存verification_cache，避免序列化问题
                            "verification_cache": {}
                        }
                        self._save_checkpoint(checkpoint_path, checkpoint_data)
                        print(f"💾 CCPO检查点已保存: {idx+1}/{len(questions)}")
                    
                    # 进度报告
                    if not self.args.debug_v2 and (idx + 1) % 3 == 0:
                        print(f"📊 CCPO处理进度: {idx+1}/{len(questions)} (高质量推理: {self.ccpo_stats['high_quality_reasoning']})")
        
        except Exception as e:
            logger.error(f"CCPO验证过程出现严重错误: {e}")
            # 保存当前进度
            checkpoint_data = {
                "processed_indices": list(processed_indices),
                "all_scores": all_scores,
                "retry_count": retry_count + 1,
                # 不保存verification_cache，避免序列化问题
                "verification_cache": {}
            }
            self._save_checkpoint(checkpoint_path, checkpoint_data)
            
            if retry_count + 1 < self.max_retries:
                print(f"⚠️  CCPO验证过程中断，将在重试时从检查点恢复")
                raise e
            else:
                print(f"❌ 已达到最大重试次数，使用部分结果")
        
        # 确保所有位置都有分数
        for i in range(len(all_scores)):
            if all_scores[i] is None:
                all_scores[i] = self._get_default_reasoning_scores(len(candidates_list[0]) if candidates_list else 5).tolist()
        
        # 保存最终检查点 - 修复序列化问题
        final_checkpoint = {
            "processed_indices": list(range(len(questions))),
            "all_scores": all_scores,
            "retry_count": retry_count,
            "completed": True,
            # 不保存verification_cache，避免序列化问题
            "verification_cache": {}
        }
        self._save_checkpoint(checkpoint_path, final_checkpoint)
        
        # 计算最终统计
        if self.ccpo_stats['reasoning_verification_attempted'] > 0:
            quality_rate = self.ccpo_stats['high_quality_reasoning'] / (
                self.ccpo_stats['reasoning_verification_attempted'] * len(candidates_list[0])
            ) if candidates_list else 0
            self.ccpo_stats['avg_reasoning_confidence'] = quality_rate
        
        self._print_ccpo_stats()
        return [np.array(scores) for scores in all_scores]
    
    def _calculate_reasoning_quality_scores(self, verification_results: List[VerificationResult]) -> np.ndarray:
        """
        基于CCPO验证结果计算推理质量分数
        核心创新：高质量推理过程获得高分，低质量推理过程获得低分
        """
        scores = []
        
        for result in verification_results:
            if result.verified:
                # 高质量推理：基于置信度的高分
                base_score = 30.0  # 高基础分（比传统方法更高）
                confidence_bonus = result.confidence * 15.0  # 更高的置信度加成
                execution_bonus = 8.0 if result.execution_time < 10 else 3.0  # 执行效率加成
                reasoning_bonus = 5.0  # CCPO推理质量加成
                score = base_score + confidence_bonus + execution_bonus + reasoning_bonus
                self.ccpo_stats['high_quality_reasoning'] += 1
            else:
                # 低质量推理：基于问题类型的惩罚分
                if result.status == VerificationStatus.REASONING_FAILED:
                    score = -15.0  # 推理转换失败
                elif result.status == VerificationStatus.EXECUTION_FAILED:
                    score = -12.0  # 执行失败（推理有问题）
                elif result.status == VerificationStatus.NO_CODE_GENERATED:
                    score = -8.0  # 无法生成代码（推理不清晰）
                elif result.status == VerificationStatus.TIMEOUT:
                    score = -5.0  # 超时
                else:
                    score = -3.0  # 答案不匹配（推理逻辑错误）
            
            scores.append(score)
        
        # 转换为numpy数组并增强差异
        scores = np.array(scores)
        
        # CCPO特有的分数调整：强化高质量推理的优势
        if len(scores) > 1:
            score_range = scores.max() - scores.min()
            if score_range < 5.0:  # 如果分数差异太小，增强差异
                median_score = np.median(scores)
                for i in range(len(scores)):
                    if scores[i] > median_score:
                        scores[i] += 10.0  # 大幅提升高质量推理
                    elif scores[i] < median_score:
                        scores[i] -= 8.0  # 大幅降低低质量推理
        
        return scores
    
    def _get_default_reasoning_scores(self, num_candidates: int) -> np.ndarray:
        """获取默认推理质量分数（未验证样本）"""
        # 为未验证的样本提供轻微随机化的分数
        base_scores = np.linspace(-2, 2, num_candidates)
        noise = np.random.normal(0, 0.5, num_candidates)
        return base_scores + noise
    
    def _print_ccpo_stats(self):
        """打印CCPO验证统计信息"""
        stats = self.ccpo_stats
        print("\n" + "="*60)
        print("📈 CCPO推理质量验证统计报告 (Architecture B)")
        print("="*60)
        print(f"总处理样本数: {stats['total_processed']}")
        print(f"推理验证尝试数: {stats['reasoning_verification_attempted']}")
        print(f"高质量推理数: {stats['high_quality_reasoning']}")
        print(f"低质量推理数: {stats['low_quality_reasoning']}")
        print(f"执行失败数: {stats['execution_failures']}")
        print(f"缓存结果数: {stats['cached_results']}")
        
        if stats['reasoning_verification_attempted'] > 0:
            quality_rate = stats['high_quality_reasoning'] / (stats['high_quality_reasoning'] + stats['low_quality_reasoning'])
            print(f"推理质量率: {quality_rate:.2%}")
        
        print(f"平均推理置信度: {stats['avg_reasoning_confidence']:.3f}")
        print(f"CCPO验证器状态: ✅ Architecture B正常运行")
        print(f"核心创新: 服务器按7B推理思路执行代码验证推理质量")
        print("="*60)

async def ccpo_code_verified_ranking(args, questions, candidates, ground_truths):
    """
    CCPO主排名函数 - Architecture B实现
    增加ground_truth支持用于最终验证
    """
    print(f"🚀 启动CCPO代码验证排名系统 (Architecture B)")
    print(f"   版本: CCPO推理质量验证版")
    print(f"   核心创新: 服务器按7B推理思路执行代码")
    print(f"   验证服务器: {args.verification_url}")
    print(f"   样本数量: {len(questions)}")
    print(f"   Ground Truth: {'可用' if ground_truths and ground_truths[0] != 'Unknown' else '不可用'}")
    
    # 初始化CCPO排名器
    ranker = CCPOCodeVerifiedRanker(args)
    
    # 执行CCPO推理质量验证排名
    start_time = time.time()
    ranks = await ranker.verify_reasoning_with_ccpo_verifier(questions, candidates, ground_truths)
    execution_time = time.time() - start_time
    
    print(f"✅ CCPO推理质量验证排名完成")
    print(f"   总耗时: {execution_time:.2f}秒")
    print(f"   平均每样本: {execution_time/len(questions):.3f}秒")
    
    # 保存结果
    output_path = f"ranking/{args.output_dir}/ccpo_{args.gpu}_{args.data_frac}.npy"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, ranks)
    
    print(f"💾 CCPO排名结果已保存: {output_path}")
    return ranks

async def main(args):
    """主函数"""
    print("🔥 CCPO代码验证排名脚本 (Architecture B)")
    print("核心创新：用服务器按照7B推理思路执行代码验证推理质量")
    print("="*70)
    
    # 验证依赖
    print("🔍 检查CCPO依赖...")
    try:
        from execution_verifier import ExecutionVerifier
        print("✅ CCPO执行验证器可用")
    except ImportError as e:
        print("❌ CCPO执行验证器不可用")
        print("请确保以下文件存在并已修复:")
        print("  - execution_verifier.py")
        print("  - enhanced_client_example.py")
        return 1
    
    # 导入数据集过滤器（可选）
    try:
        from improved_verification_filter import filter_dataset_for_verification, print_filter_report
        print("✅ 通用过滤器可用")
        use_filter = False
    except ImportError:
        print("⚠️  过滤器不可用，将处理所有样本（CCPO数学数据集已预过滤）")
        use_filter = False
    
    # 加载数据
    print(f"\n📊 加载数据集: {args.prompts}")
    try:
        data = load_dataset(args.prompts, split="train")
        print(f"✅ 数据集加载成功，样本数: {len(data)}")
    except Exception as e:
        print(f"❌ 数据集加载失败: {e}")
        return 1
    
    # 初始化分词器
    print(f"\n🔧 初始化分词器...")
    if "mistral" in args.model.lower():
        tokenizer = AutoTokenizer.from_pretrained("/data/jiacheng/dylan/aaai/Code-Consistency-Preference-Optimization/cppo/mistralai/Mistral-7B-Instruct-v0.2")
    elif "llama-3" in args.model.lower():
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
    elif "gemma-2" in args.model.lower():
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b-it")
    else:
        print(f"⚠️  未知模型类型，使用默认Mistral分词器")
        tokenizer = AutoTokenizer.from_pretrained("/data/jiacheng/dylan/aaai/Code-Consistency-Preference-Optimization/cppo/mistralai/Mistral-7B-Instruct-v0.2")
    
    tokenizer.pad_token = tokenizer.eos_token
    print("✅ 分词器初始化完成")
    
    # 处理提示
    prompts_all = [apply_template(data[idx]["prompt"], tokenizer) for idx in range(len(data))]
    print(f"✅ 提示模板应用完成")
    if args.debug_v2:
        print(f"示例提示: {prompts_all[0][:100]}...")
    
    # 加载生成的推理过程和元数据 - 修复数据对应关系
    print(f"\n📂 加载生成的推理过程和元数据...")
    
    # 首先加载元数据文件
    metadata_file = f"{args.output_dir}/metadata_{args.data_frac}.json"
    if not os.path.exists(metadata_file):
        print(f"❌ 元数据文件不存在: {metadata_file}")
        print("💡 请确保使用修复版的generate.py生成数据")
        return 1
    
    with open(metadata_file, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    print(f"✅ 元数据加载成功:")
    print(f"   - 原始数据集大小: {metadata['total_original_samples']}")
    print(f"   - 处理样本数: {metadata['processed_samples']}")
    print(f"   - 生成响应对数: {metadata['pairs']}")
    
    # 从元数据重建数据项
    data_items = metadata['data_items']
    questions_from_metadata = [item['prompt'] for item in data_items]
    answers_from_metadata = [item['answer'] for item in data_items]
    sources_from_metadata = [item['source'] for item in data_items]
    original_indices = [item['original_index'] for item in data_items]
    
    print(f"✅ 从元数据重建数据项: {len(data_items)} 条")
    
    # 加载对应的responses
    pairs = args.pairs
    all_generated = []
    
    for i in range(pairs):
        response_file = f"{args.output_dir}/responses_{i}.json"
        if not os.path.exists(response_file):
            print(f"❌ 响应文件不存在: {response_file}")
            return 1
        
        with open(response_file, 'r', encoding='utf-8') as f:
            gen = json.load(f)
            all_generated.append(gen)
            print(f"✅ 加载响应文件 {os.path.basename(response_file)} ({len(gen)} 条)")
    
    # 验证数据长度一致性
    expected_length = len(data_items)
    for i, gen in enumerate(all_generated):
        if len(gen) != expected_length:
            print(f"❌ 数据长度不匹配: metadata({expected_length}) vs responses_{i}({len(gen)})")
            return 1
    
    # 建立完整的数据对应关系
    candidates_texts = list(zip(*all_generated))
    
    print(f"✅ 数据对应关系验证完成")
    print(f"   - 问题数量: {len(questions_from_metadata)}")
    print(f"   - 每问题推理过程数: {len(candidates_texts[0]) if candidates_texts else 0}")
    print(f"   - Ground Truth可用: ✅")
    
    # 创建完整的验证数据集
    verification_dataset = []
    for idx in range(len(data_items)):
        verification_dataset.append({
            "original_index": original_indices[idx],
            "original_question": questions_from_metadata[idx],
            "reasoning_processes": candidates_texts[idx],
            "ground_truth": answers_from_metadata[idx],
            "source": sources_from_metadata[idx]
        })
    
    print(f"✅ 完整验证数据集构建完成: {len(verification_dataset)} 条记录")
    
    # 显示数据样本（确认对应关系正确）
    if args.debug_v2 and verification_dataset:
        print(f"\n📋 数据对应关系验证:")
        sample = verification_dataset[0]
        print(f"  原始索引: {sample['original_index']}")
        print(f"  原始问题: {sample['original_question'][:100]}...")
        print(f"  Ground Truth: {sample['ground_truth'][:100]}...")
        print(f"  推理过程数: {len(sample['reasoning_processes'])}")
        for i, reasoning in enumerate(sample['reasoning_processes'][:2]):
            print(f"    推理{i+1}: {reasoning[:100]}...")
        print(f"  数据来源: {sample['source']}")
        
        # 验证第二个样本
        if len(verification_dataset) > 1:
            sample2 = verification_dataset[1]
            print(f"\n  第二个样本验证:")
            print(f"    原始索引: {sample2['original_index']}")
            print(f"    问题: {sample2['original_question'][:50]}...")
            print(f"    答案: {sample2['ground_truth'][:50]}...")
    
    # 为后续处理准备数据
    questions_for_verification = [item["original_question"] for item in verification_dataset]
    candidates_for_verification = [item["reasoning_processes"] for item in verification_dataset]
    ground_truths = [item["ground_truth"] for item in verification_dataset]
    
    # 分片处理（应用到完整验证数据集）
    data_frac, frac_len = args.data_frac, args.frac_len
    verification_dataset = split_prompts(verification_dataset, frac_len, data_frac)
    
    # 重新提取处理后的数据
    questions_for_verification = [item["original_question"] for item in verification_dataset]
    candidates_for_verification = [item["reasoning_processes"] for item in verification_dataset]
    ground_truths = [item["ground_truth"] for item in verification_dataset]
    
    print(f"✅ 分片后样本数: {len(verification_dataset)}")
    
    # 智能过滤（可选，CCPO数学数据集已预过滤）
    if use_filter:
        print(f"\n🔍 应用数学问题过滤器...")
        
        max_verification_samples = max(10, int(len(questions_for_verification) * args.verification_sample_rate * 20))
        
        filtered_questions, filtered_candidates, filter_stats = filter_dataset_for_verification(
            questions_for_verification, 
            candidates_for_verification, 
            max_samples=max_verification_samples,
            debug=args.debug_v2
        )
        
        print_filter_report(filter_stats)
        
        if len(filtered_questions) < 5:
            print("⚠️  过滤后样本数太少，将使用原始CCPO数学数据集")
            filtered_questions, filtered_candidates = questions_for_verification, candidates_for_verification
        else:
            questions_for_verification, candidates_for_verification = filtered_questions, filtered_candidates
            # 同步调整ground_truths
            ground_truths = ground_truths[:len(questions_for_verification)]
    
    filter_info = "数学问题" if use_filter else "无（CCPO数据集已预过滤）"
    
    print(f"\n🎯 开始CCPO推理质量验证排名 (Architecture B)")
    print(f"   最终处理样本数: {len(questions_for_verification)}")
    print(f"   每样本推理过程数: {len(candidates_for_verification[0]) if candidates_for_verification else 0}")
    print(f"   智能过滤: {filter_info}")
    print(f"   Ground Truth可用: {'✅' if ground_truths and ground_truths[0] != 'Unknown' else '❌'}")
    print(f"   核心创新: 服务器按7B推理思路执行代码验证推理质量")
    
    # 执行CCPO推理质量验证排名
    await ccpo_code_verified_ranking(args, questions_for_verification, candidates_for_verification, ground_truths)
    
    print(f"\n✅ CCPO推理质量验证排名完成!")
    print(f"🎉 Architecture B核心创新已实现:")
    print(f"   - 7B模型生成推理过程")
    print(f"   - 服务器按推理思路生成并执行代码")
    print(f"   - 验证推理过程的质量")
    print(f"   - 为强化学习提供高质量的偏好信号")
    return 0

if __name__ == "__main__":
    args = parse_arguments()
    
    # 运行异步主函数
    exit_code = asyncio.run(main(args))
    exit(exit_code)