#!/usr/bin/env python3
"""
Self Evolution Controller - 自我进化循环控制器
管理模型的持续学习和自我改进过程
"""

import os
import json
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import torch
from datasets import Dataset, concatenate_datasets
import yaml

from execution_verifier import ExecutionVerifier, VerificationResult, batch_verify_responses

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EvolutionMetrics:
    """进化指标"""
    iteration: int
    timestamp: str
    execution_success_rate: float
    answer_accuracy_rate: float
    consistency_score: float
    improvement_rate: float
    total_verified_samples: int
    high_confidence_samples: int
    model_version: str

@dataclass
class EvolutionConfig:
    """进化配置"""
    # 数据收集配置
    min_samples_per_iteration: int = 100
    max_samples_per_iteration: int = 1000
    verification_batch_size: int = 10
    
    # 质量阈值
    min_execution_success_rate: float = 0.6
    min_accuracy_improvement: float = 0.02
    confidence_threshold: float = 0.8
    
    # 训练配置
    training_epochs_per_iteration: int = 1
    learning_rate_decay: float = 0.95
    warmup_steps: int = 100
    
    # 模型管理
    max_model_versions: int = 10
    checkpoint_interval: int = 5
    
    # 安全设置
    max_continuous_failures: int = 3
    rollback_threshold: float = 0.1  # 性能下降超过10%时回滚

class DataCollector:
    """数据收集器"""
    
    def __init__(self, config: EvolutionConfig):
        self.config = config
        self.collected_samples = []
        
    def add_sample(self, question: str, response: str, verification: VerificationResult):
        """添加样本"""
        sample = {
            'question': question,
            'response': response,
            'verification': asdict(verification),
            'timestamp': datetime.now().isoformat(),
            'quality_score': self._compute_quality_score(verification)
        }
        self.collected_samples.append(sample)
    
    def _compute_quality_score(self, verification: VerificationResult) -> float:
        """计算样本质量分数"""
        base_score = verification.confidence
        
        # 执行成功奖励
        if verification.status.value == "success":
            base_score += 0.1
        
        # 高置信度奖励
        if verification.confidence > 0.9:
            base_score += 0.1
        
        # 执行时间惩罚（太慢的代码）
        if verification.execution_time > 10:
            base_score -= 0.1
        
        return min(1.0, max(0.0, base_score))
    
    def get_high_quality_samples(self, min_quality: float = 0.8) -> List[Dict]:
        """获取高质量样本"""
        return [
            sample for sample in self.collected_samples
            if sample['quality_score'] >= min_quality
        ]
    
    def clear_samples(self):
        """清空样本"""
        self.collected_samples = []

class PreferenceGenerator:
    """偏好生成器 - 基于代码验证结果生成训练偏好对"""
    
    def __init__(self):
        pass
    
    def generate_preference_pairs(
        self, 
        verified_samples: List[Dict]
    ) -> List[Dict]:
        """
        从验证样本生成偏好对
        
        Args:
            verified_samples: 已验证的样本列表
        
        Returns:
            偏好对列表，格式符合SPPO训练要求
        """
        preference_pairs = []
        
        # 按问题分组
        question_groups = {}
        for sample in verified_samples:
            question = sample['question']
            if question not in question_groups:
                question_groups[question] = []
            question_groups[question].append(sample)
        
        # 为每个问题组生成偏好对
        for question, samples in question_groups.items():
            if len(samples) < 2:
                continue
            
            # 按质量分数排序
            samples.sort(key=lambda x: x['quality_score'], reverse=True)
            
            # 生成多个偏好对
            for i in range(len(samples)):
                for j in range(i + 1, min(i + 3, len(samples))):  # 限制对比数量
                    chosen_sample = samples[i]
                    rejected_sample = samples[j]
                    
                    # 计算偏好概率
                    chosen_verification = chosen_sample['verification']
                    rejected_verification = rejected_sample['verification']
                    
                    preference_probs = self._compute_preference_probabilities(
                        chosen_verification, rejected_verification
                    )
                    
                    pair = {
                        'prompt': question,
                        'chosen': chosen_sample['response'],
                        'rejected': rejected_sample['response'],
                        'chosen_probs': preference_probs['chosen_prob'],
                        'chosen_probs_win': preference_probs['chosen_prob_win'],
                        'chosen_probs_lose': preference_probs['chosen_prob_lose'],
                        'chosen_verification_score': chosen_sample['quality_score'],
                        'rejected_verification_score': rejected_sample['quality_score']
                    }
                    
                    preference_pairs.append(pair)
        
        logger.info(f"Generated {len(preference_pairs)} preference pairs from {len(verified_samples)} samples")
        return preference_pairs
    
    def _compute_preference_probabilities(
        self,
        chosen_verification: Dict,
        rejected_verification: Dict
    ) -> Dict[str, float]:
        """计算偏好概率"""
        chosen_verified = chosen_verification.get('verified', False)
        rejected_verified = rejected_verification.get('verified', False)
        chosen_confidence = chosen_verification.get('confidence', 0.0)
        rejected_confidence = rejected_verification.get('confidence', 0.0)
        
        if chosen_verified and not rejected_verified:
            # chosen正确，rejected错误：强偏好chosen
            chosen_prob = 0.9 + 0.1 * chosen_confidence
            chosen_prob_win = chosen_prob
            chosen_prob_lose = 1 - chosen_prob
        elif not chosen_verified and rejected_verified:
            # chosen错误，rejected正确：强偏好rejected
            chosen_prob = 0.1 - 0.1 * rejected_confidence
            chosen_prob_win = chosen_prob
            chosen_prob_lose = 1 - chosen_prob
        elif chosen_verified and rejected_verified:
            # 都正确：基于置信度决定偏好
            confidence_diff = chosen_confidence - rejected_confidence
            chosen_prob = 0.5 + 0.3 * confidence_diff
            chosen_prob = max(0.1, min(0.9, chosen_prob))
            chosen_prob_win = chosen_prob
            chosen_prob_lose = 1 - chosen_prob
        else:
            # 都错误：基于置信度决定偏好（偏好置信度更高的）
            if chosen_confidence > rejected_confidence:
                chosen_prob = 0.6
                chosen_prob_win = 0.6
                chosen_prob_lose = 0.4
            elif chosen_confidence < rejected_confidence:
                chosen_prob = 0.4
                chosen_prob_win = 0.4
                chosen_prob_lose = 0.6
            else:
                chosen_prob = 0.5
                chosen_prob_win = 0.5
                chosen_prob_lose = 0.5
        
        return {
            'chosen_prob': chosen_prob,
            'chosen_prob_win': chosen_prob_win,
            'chosen_prob_lose': chosen_prob_lose
        }

class ModelManager:
    """模型版本管理器"""
    
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir = self.base_dir / "models"
        self.models_dir.mkdir(exist_ok=True)
        self.metrics_file = self.base_dir / "evolution_metrics.json"
        
    def save_model(self, model, tokenizer, version: str, metrics: EvolutionMetrics):
        """保存模型版本"""
        version_dir = self.models_dir / f"version_{version}"
        version_dir.mkdir(exist_ok=True)
        
        # 保存模型和tokenizer
        model.save_pretrained(version_dir / "model")
        tokenizer.save_pretrained(version_dir / "tokenizer")
        
        # 保存指标
        metrics_path = version_dir / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(asdict(metrics), f, indent=2)
        
        # 更新全局指标记录
        self._update_global_metrics(metrics)
        
        logger.info(f"Saved model version {version} to {version_dir}")
    
    def load_model(self, version: str):
        """加载指定版本的模型"""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        version_dir = self.models_dir / f"version_{version}"
        if not version_dir.exists():
            raise ValueError(f"Model version {version} not found")
        
        model = AutoModelForCausalLM.from_pretrained(version_dir / "model")
        tokenizer = AutoTokenizer.from_pretrained(version_dir / "tokenizer")
        
        return model, tokenizer
    
    def get_best_model_version(self) -> Optional[str]:
        """获取最佳模型版本"""
        if not self.metrics_file.exists():
            return None
        
        with open(self.metrics_file, 'r') as f:
            all_metrics = json.load(f)
        
        if not all_metrics:
            return None
        
        # 按综合分数排序
        best_metrics = max(
            all_metrics,
            key=lambda x: x['answer_accuracy_rate'] * 0.4 + 
                         x['execution_success_rate'] * 0.3 + 
                         x['consistency_score'] * 0.3
        )
        
        return best_metrics['model_version']
    
    def _update_global_metrics(self, metrics: EvolutionMetrics):
        """更新全局指标记录"""
        all_metrics = []
        if self.metrics_file.exists():
            with open(self.metrics_file, 'r') as f:
                all_metrics = json.load(f)
        
        all_metrics.append(asdict(metrics))
        
        # 保持最近的记录
        if len(all_metrics) > 100:
            all_metrics = all_metrics[-100:]
        
        with open(self.metrics_file, 'w') as f:
            json.dump(all_metrics, f, indent=2)

class SelfEvolutionController:
    """自我进化控制器 - 主要组件"""
    
    def __init__(
        self,
        config: EvolutionConfig,
        model_base_dir: str = "./evolution_models",
        verification_config: Dict = None
    ):
        self.config = config
        self.model_manager = ModelManager(model_base_dir)
        self.data_collector = DataCollector(config)
        self.preference_generator = PreferenceGenerator()
        
        # 初始化验证器配置
        verification_config = verification_config or {}
        self.verifier_config = {
            'base_url': verification_config.get('base_url', 'https://httpsnet.top:17432'),
            'username': verification_config.get('username', 'newuser'),
            'password': verification_config.get('password', 'newPass123'),
        }
        
        # 状态跟踪
        self.current_iteration = 0
        self.continuous_failures = 0
        self.best_metrics = None
        
    async def evolve_model(
        self,
        model,
        tokenizer,
        initial_questions: List[str],
        max_iterations: int = 10
    ) -> Tuple[Any, Any, List[EvolutionMetrics]]:
        """
        执行模型自我进化
        
        Args:
            model: 初始模型
            tokenizer: 分词器
            initial_questions: 初始问题集
            max_iterations: 最大迭代次数
        
        Returns:
            (evolved_model, tokenizer, metrics_history)
        """
        logger.info(f"Starting self-evolution process for {max_iterations} iterations")
        
        metrics_history = []
        current_model = model
        current_questions = initial_questions.copy()
        
        async with ExecutionVerifier(**self.verifier_config) as verifier:
            for iteration in range(max_iterations):
                self.current_iteration = iteration
                logger.info(f"Starting evolution iteration {iteration + 1}/{max_iterations}")
                
                try:
                    # 生成新的回答
                    responses = await self._generate_responses(
                        current_model, tokenizer, current_questions
                    )
                    
                    # 验证回答质量
                    verification_results = await batch_verify_responses(
                        current_questions, responses, verifier,
                        max_concurrent=self.config.verification_batch_size
                    )
                    
                    # 收集数据
                    for q, r, v in zip(current_questions, responses, verification_results):
                        self.data_collector.add_sample(q, r, v)
                    
                    # 计算当前指标
                    current_metrics = self._compute_metrics(verification_results, iteration)
                    metrics_history.append(current_metrics)
                    
                    # 检查是否需要训练
                    if self._should_train(current_metrics):
                        logger.info("Training model with verified samples...")
                        
                        # 生成偏好对
                        high_quality_samples = self.data_collector.get_high_quality_samples()
                        preference_pairs = self.preference_generator.generate_preference_pairs(
                            high_quality_samples
                        )
                        
                        if len(preference_pairs) >= self.config.min_samples_per_iteration:
                            # 训练模型
                            new_model = await self._train_model(
                                current_model, tokenizer, preference_pairs
                            )
                            
                            # 评估新模型
                            new_metrics = await self._evaluate_model(
                                new_model, tokenizer, current_questions[:50], verifier
                            )
                            
                            # 决定是否采用新模型
                            if self._should_adopt_model(current_metrics, new_metrics):
                                logger.info("Adopting improved model")
                                current_model = new_model
                                self.continuous_failures = 0
                                
                                # 保存模型
                                version = f"{iteration + 1:03d}"
                                self.model_manager.save_model(
                                    current_model, tokenizer, version, new_metrics
                                )
                                
                                current_metrics = new_metrics
                            else:
                                logger.info("New model did not improve, keeping current model")
                                self.continuous_failures += 1
                        else:
                            logger.warning(f"Not enough high-quality samples: {len(preference_pairs)}")
                            self.continuous_failures += 1
                    
                    # 更新最佳指标
                    if self.best_metrics is None or self._is_better_metrics(current_metrics, self.best_metrics):
                        self.best_metrics = current_metrics
                    
                    # 扩展问题集（自我生成新问题）
                    if iteration % 3 == 0:  # 每3轮扩展一次
                        new_questions = await self._generate_new_questions(
                            current_model, tokenizer, len(current_questions) // 10
                        )
                        current_questions.extend(new_questions)
                        logger.info(f"Extended question set to {len(current_questions)} questions")
                    
                    # 安全检查
                    if self.continuous_failures >= self.config.max_continuous_failures:
                        logger.warning("Too many continuous failures, stopping evolution")
                        break
                    
                    # 清理样本（保持内存使用合理）
                    self.data_collector.clear_samples()
                    
                    logger.info(f"Iteration {iteration + 1} completed. "
                              f"Accuracy: {current_metrics.answer_accuracy_rate:.3f}, "
                              f"Success Rate: {current_metrics.execution_success_rate:.3f}")
                
                except Exception as e:
                    logger.error(f"Error in iteration {iteration + 1}: {e}")
                    self.continuous_failures += 1
                    if self.continuous_failures >= self.config.max_continuous_failures:
                        break
        
        logger.info("Self-evolution process completed")
        return current_model, tokenizer, metrics_history
    
    async def _generate_responses(
        self, 
        model, 
        tokenizer, 
        questions: List[str]
    ) -> List[str]:
        """生成回答"""
        # 这里应该调用模型生成回答
        # 为了简化，我们假设有一个生成函数
        responses = []
        
        for question in questions:
            # 构建输入
            messages = [{"role": "user", "content": question}]
            input_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            # 编码
            inputs = tokenizer(input_text, return_tensors="pt")
            
            # 生成
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # 解码
            response = tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:], 
                skip_special_tokens=True
            )
            responses.append(response.strip())
        
        return responses
    
    def _compute_metrics(
        self, 
        verification_results: List[VerificationResult], 
        iteration: int
    ) -> EvolutionMetrics:
        """计算进化指标"""
        total_samples = len(verification_results)
        successful_executions = sum(1 for r in verification_results if r.status.value == "success")
        verified_answers = sum(1 for r in verification_results if r.verified)
        high_confidence = sum(1 for r in verification_results if r.confidence > 0.8)
        
        execution_success_rate = successful_executions / total_samples if total_samples > 0 else 0
        answer_accuracy_rate = verified_answers / total_samples if total_samples > 0 else 0
        consistency_score = np.mean([r.confidence for r in verification_results])
        
        # 计算改进率
        improvement_rate = 0.0
        if self.best_metrics:
            improvement_rate = answer_accuracy_rate - self.best_metrics.answer_accuracy_rate
        
        return EvolutionMetrics(
            iteration=iteration,
            timestamp=datetime.now().isoformat(),
            execution_success_rate=execution_success_rate,
            answer_accuracy_rate=answer_accuracy_rate,
            consistency_score=consistency_score,
            improvement_rate=improvement_rate,
            total_verified_samples=total_samples,
            high_confidence_samples=high_confidence,
            model_version=f"iter_{iteration:03d}"
        )
    
    def _should_train(self, metrics: EvolutionMetrics) -> bool:
        """判断是否应该进行训练"""
        # 如果执行成功率太低，不进行训练
        if metrics.execution_success_rate < self.config.min_execution_success_rate:
            return False
        
        # 如果有足够的高质量样本，进行训练
        high_quality_samples = len(self.data_collector.get_high_quality_samples())
        if high_quality_samples >= self.config.min_samples_per_iteration:
            return True
        
        return False
    
    async def _train_model(self, model, tokenizer, preference_pairs: List[Dict]):
        """训练模型（这里需要集成SPPO训练逻辑）"""
        # 这里应该调用您现有的SPPO训练代码
        # 为了演示，我们返回原模型（实际实现中应该训练并返回新模型）
        logger.info(f"Training with {len(preference_pairs)} preference pairs")
        
        # TODO: 集成实际的SPPO训练逻辑
        # 1. 创建Dataset
        # 2. 调用SPPOTrainer
        # 3. 返回训练后的模型
        
        return model  # 临时返回原模型
    
    async def _evaluate_model(
        self, 
        model, 
        tokenizer, 
        test_questions: List[str], 
        verifier: ExecutionVerifier
    ) -> EvolutionMetrics:
        """评估模型性能"""
        responses = await self._generate_responses(model, tokenizer, test_questions)
        verification_results = await batch_verify_responses(
            test_questions, responses, verifier
        )
        
        return self._compute_metrics(verification_results, -1)  # -1表示评估
    
    def _should_adopt_model(
        self, 
        old_metrics: EvolutionMetrics, 
        new_metrics: EvolutionMetrics
    ) -> bool:
        """判断是否应该采用新模型"""
        # 综合分数比较
        old_score = (old_metrics.answer_accuracy_rate * 0.4 + 
                    old_metrics.execution_success_rate * 0.3 + 
                    old_metrics.consistency_score * 0.3)
        
        new_score = (new_metrics.answer_accuracy_rate * 0.4 + 
                    new_metrics.execution_success_rate * 0.3 + 
                    new_metrics.consistency_score * 0.3)
        
        # 必须有显著改进
        improvement = new_score - old_score
        return improvement > self.config.min_accuracy_improvement
    
    def _is_better_metrics(
        self, 
        metrics1: EvolutionMetrics, 
        metrics2: EvolutionMetrics
    ) -> bool:
        """比较两个指标哪个更好"""
        score1 = (metrics1.answer_accuracy_rate * 0.4 + 
                 metrics1.execution_success_rate * 0.3 + 
                 metrics1.consistency_score * 0.3)
        
        score2 = (metrics2.answer_accuracy_rate * 0.4 + 
                 metrics2.execution_success_rate * 0.3 + 
                 metrics2.consistency_score * 0.3)
        
        return score1 > score2
    
    async def _generate_new_questions(
        self, 
        model, 
        tokenizer, 
        num_questions: int
    ) -> List[str]:
        """生成新的问题来扩展训练集"""
        new_questions = []
        
        # 问题生成提示模板
        question_prompts = [
            "请生成一个需要数学计算的问题，要求答案是一个具体的数字：",
            "请生成一个物理问题，需要通过公式计算得出数值答案：",
            "请生成一个逻辑推理问题，答案应该是一个明确的数字：",
            "请生成一个几何问题，需要计算面积、体积或长度：",
            "请生成一个概率或统计问题，答案是一个数值："
        ]
        
        for i in range(num_questions):
            try:
                prompt = question_prompts[i % len(question_prompts)]
                messages = [{"role": "user", "content": prompt}]
                input_text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                
                inputs = tokenizer(input_text, return_tensors="pt")
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=256,
                        temperature=0.8,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id
                    )
                
                question = tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[1]:], 
                    skip_special_tokens=True
                ).strip()
                
                # 简单的问题质量检查
                if len(question) > 20 and "?" in question:
                    new_questions.append(question)
                
            except Exception as e:
                logger.warning(f"Failed to generate question {i}: {e}")
        
        return new_questions

# 使用示例和测试函数
async def test_evolution():
    """测试自我进化系统"""
    # 配置
    config = EvolutionConfig(
        min_samples_per_iteration=20,
        max_samples_per_iteration=100,
        verification_batch_size=5,
        min_execution_success_rate=0.5,
        min_accuracy_improvement=0.01
    )
    
    # 初始问题集
    initial_questions = [
        "strawberry中有几个字母r？",
        "计算 15 + 27 × 3 的结果",
        "一个正方形的边长是5米，它的面积是多少平方米？",
        "从1到100的所有偶数的和是多少？",
        "一个圆的半径是3米，它的周长是多少米？（π取3.14）"
    ]
    
    # 验证器配置（使用你的服务器）
    verification_config = {
        'base_url': 'https://8.134.217.190:17432',
        'username': 'newuser',
        'password': 'newPass123'
    }
    
    # 创建进化控制器
    controller = SelfEvolutionController(
        config=config,
        model_base_dir="./test_evolution",
        verification_config=verification_config
    )
    
    # 加载初始模型（这里需要替换为实际的模型加载）
    # model, tokenizer = load_your_model()
    
    # 模拟进化过程（因为没有实际模型，这里只是演示结构）
    logger.info("Starting evolution test...")
    
    # 测试单个组件
    async with ExecutionVerifier(**verification_config) as verifier:
        # 测试问题验证
        test_responses = [
            "strawberry中有3个字母r",
            "15 + 27 × 3 = 15 + 81 = 96",
            "正方形面积 = 边长² = 5² = 25平方米",
            "偶数和 = 2 + 4 + ... + 100 = 2550",
            "圆周长 = 2πr = 2 × 3.14 × 3 = 18.84米"
        ]
        
        results = await batch_verify_responses(
            initial_questions, test_responses, verifier
        )
        
        # 显示验证结果
        for i, result in enumerate(results):
            print(f"\n问题 {i+1}: {initial_questions[i]}")
            print(f"回答: {test_responses[i]}")
            print(f"验证通过: {result.verified}")
            print(f"置信度: {result.confidence:.3f}")
            print(f"AI答案: {result.ai_answer}")
            print(f"代码答案: {result.code_answer}")
        
        # 测试指标计算
        metrics = controller._compute_metrics(results, 0)
        print(f"\n整体指标:")
        print(f"执行成功率: {metrics.execution_success_rate:.3f}")
        print(f"答案准确率: {metrics.answer_accuracy_rate:.3f}")
        print(f"一致性分数: {metrics.consistency_score:.3f}")

if __name__ == "__main__":
    asyncio.run(test_evolution())