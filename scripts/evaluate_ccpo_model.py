#!/usr/bin/env python3
"""
CCPO模型评估脚本
评估基于代码验证训练的模型在数学推理任务上的表现
"""

import asyncio
import json
import argparse
import logging
import time
from typing import Dict, List, Any, Tuple
from pathlib import Path
import numpy as np
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 导入验证器
from execution_verifier import ExecutionVerifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="评估CCPO训练的模型")
    parser.add_argument("--model_path", type=str, required=True,
                       help="CCPO训练的模型路径")
    parser.add_argument("--test_dataset", type=str, required=True,
                       help="测试数据集路径")
    parser.add_argument("--verification_sample_size", type=int, default=50,
                       help="进行代码验证的样本数量")
    parser.add_argument("--output_report", type=str, required=True,
                       help="评估报告输出路径")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                       help="生成的最大token数")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="生成温度")
    parser.add_argument("--top_p", type=float, default=0.9,
                       help="核采样参数")
    parser.add_argument("--verification_url", type=str, 
                       default="https://8.134.217.190:17432",
                       help="代码执行验证服务器")
    parser.add_argument("--verification_username", type=str, default="newuser")
    parser.add_argument("--verification_password", type=str, default="newPass123")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


class CCPOModelEvaluator:
    """CCPO模型评估器"""
    
    def __init__(self, args):
        self.args = args
        self.model = None
        self.tokenizer = None
        self.verification_stats = {
            'total_evaluated': 0,
            'verification_attempted': 0,
            'verification_successful': 0,
            'high_accuracy_responses': 0,
            'average_confidence': 0.0,
            'perfect_matches': 0,
            'generation_time': 0.0,
            'verification_time': 0.0
        }
    
    def load_model(self):
        """加载CCPO训练的模型"""
        logger.info(f"🔄 加载CCPO模型: {self.args.model_path}")
        
        try:
            # 加载tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_path)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # 加载模型
            self.model = AutoModelForCausalLM.from_pretrained(
                self.args.model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            
            logger.info(f"✅ 模型加载完成")
            logger.info(f"   参数量: {self.model.num_parameters():,}")
            logger.info(f"   设备: {next(self.model.parameters()).device}")
            
        except Exception as e:
            logger.error(f"❌ 模型加载失败: {e}")
            raise
    
    def load_test_dataset(self):
        """加载测试数据集"""
        logger.info(f"📊 加载测试数据集: {self.args.test_dataset}")
        
        try:
            if Path(self.args.test_dataset).exists():
                # 本地数据集
                dataset = load_from_disk(self.args.test_dataset)
            else:
                # HuggingFace数据集
                dataset = load_dataset(self.args.test_dataset, split="test")
            
            logger.info(f"✅ 测试数据集加载完成: {len(dataset)} 个样本")
            return dataset
            
        except Exception as e:
            logger.error(f"❌ 测试数据集加载失败: {e}")
            raise
    
    def generate_response(self, prompt: str) -> str:
        """生成模型响应"""
        # 应用聊天模板
        formatted_prompt = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}, {"role": "assistant", "content": ""}],
            tokenize=False,
            add_generation_prompt=True
        ).rstrip()
        
        # 编码输入
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        ).to(self.model.device)
        
        # 生成响应
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.args.max_new_tokens,
                temperature=self.args.temperature,
                top_p=self.args.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # 解码响应
        generated_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        return generated_text.strip()
    
    async def evaluate_model(self) -> Dict[str, Any]:
        """评估模型性能"""
        logger.info(f"🎯 开始CCPO模型评估")
        
        # 加载模型和数据
        self.load_model()
        test_dataset = self.load_test_dataset()
        
        # 选择评估样本
        eval_samples = min(self.args.verification_sample_size, len(test_dataset))
        eval_indices = np.random.choice(len(test_dataset), eval_samples, replace=False)
        
        logger.info(f"📋 评估样本数: {eval_samples}")
        
        # 生成响应
        logger.info(f"🎯 开始生成响应...")
        generation_start = time.time()
        
        generated_responses = []
        questions = []
        expected_answers = []
        
        for i, idx in enumerate(eval_indices):
            sample = test_dataset[int(idx)]
            question = sample['prompt']
            expected_answer = sample['expected_answer']
            
            try:
                response = self.generate_response(question)
                generated_responses.append(response)
                questions.append(question)
                expected_answers.append(expected_answer)
                
                if self.args.debug and i < 3:
                    logger.info(f"   样本 {i+1}: {question[:50]}...")
                    logger.info(f"   响应: {response[:100]}...")
                
            except Exception as e:
                logger.error(f"   ❌ 生成失败 样本 {i+1}: {e}")
                continue
        
        self.verification_stats['generation_time'] = time.time() - generation_start
        self.verification_stats['total_evaluated'] = len(generated_responses)
        
        logger.info(f"✅ 响应生成完成: {len(generated_responses)} 个")
        
        # 代码验证评估
        logger.info(f"🔍 开始代码验证评估...")
        verification_start = time.time()
        
        verification_results = await self._verify_responses(questions, generated_responses)
        
        self.verification_stats['verification_time'] = time.time() - verification_start
        
        # 计算评估指标
        evaluation_metrics = self._calculate_metrics(
            questions, 
            generated_responses, 
            expected_answers, 
            verification_results
        )
        
        # 生成评估报告
        report = self._generate_evaluation_report(evaluation_metrics)
        
        # 保存报告
        self._save_report(report)
        
        return report
    
    async def _verify_responses(self, questions: List[str], responses: List[str]) -> List[Dict[str, Any]]:
        """使用代码验证评估响应质量"""
        verification_results = []
        
        try:
            async with ExecutionVerifier(
                base_url=self.args.verification_url,
                username=self.args.verification_username,
                password=self.args.verification_password,
                debug=self.args.debug
            ) as verifier:
                
                for i, (question, response) in enumerate(zip(questions, responses)):
                    self.verification_stats['verification_attempted'] += 1
                    
                    try:
                        # 进行代码验证
                        result = await verifier.verify_response(question, response)
                        
                        verification_info = {
                            'question': question,
                            'response': response,
                            'verified': result.verified,
                            'confidence': result.confidence,
                            'ai_answer': result.ai_answer,
                            'code_answer': result.code_answer,
                            'status': result.status.value,
                            'execution_time': result.execution_time,
                            'error_message': result.error_message
                        }
                        
                        verification_results.append(verification_info)
                        
                        if result.verified:
                            self.verification_stats['verification_successful'] += 1
                            if result.confidence > 0.9:
                                self.verification_stats['perfect_matches'] += 1
                        
                        if self.args.debug:
                            logger.info(f"   验证 {i+1}: {'✅' if result.verified else '❌'} "
                                      f"(置信度: {result.confidence:.3f})")
                        
                        # 适当延迟以避免限流
                        await asyncio.sleep(2)
                        
                    except Exception as e:
                        logger.error(f"   ❌ 验证失败 {i+1}: {e}")
                        verification_results.append({
                            'question': question,
                            'response': response,
                            'verified': False,
                            'confidence': 0.0,
                            'ai_answer': None,
                            'code_answer': None,
                            'status': 'error',
                            'execution_time': 0.0,
                            'error_message': str(e)
                        })
        
        except Exception as e:
            logger.error(f"❌ 验证器初始化失败: {e}")
        
        return verification_results
    
    def _calculate_metrics(
        self, 
        questions: List[str], 
        responses: List[str], 
        expected_answers: List[str], 
        verification_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """计算评估指标"""
        metrics = {}
        
        # 基础统计
        total_samples = len(responses)
        verified_samples = sum(1 for r in verification_results if r['verified'])
        
        metrics['total_samples'] = total_samples
        metrics['verified_samples'] = verified_samples
        metrics['verification_rate'] = verified_samples / total_samples if total_samples > 0 else 0
        
        # 置信度统计
        confidences = [r['confidence'] for r in verification_results]
        if confidences:
            metrics['average_confidence'] = np.mean(confidences)
            metrics['confidence_std'] = np.std(confidences)
            metrics['high_confidence_rate'] = sum(1 for c in confidences if c > 0.8) / len(confidences)
        
        # 答案匹配分析
        exact_matches = 0
        ai_answers = []
        code_answers = []
        
        for i, result in enumerate(verification_results):
            ai_answer = result.get('ai_answer')
            code_answer = result.get('code_answer')
            expected = expected_answers[i] if i < len(expected_answers) else None
            
            if ai_answer:
                ai_answers.append(ai_answer)
            if code_answer:
                code_answers.append(code_answer)
            
            # 检查与期望答案的匹配
            if expected and ai_answer:
                try:
                    if abs(float(ai_answer) - float(expected)) < 1e-6:
                        exact_matches += 1
                except (ValueError, TypeError):
                    if str(ai_answer).strip() == str(expected).strip():
                        exact_matches += 1
        
        metrics['exact_match_rate'] = exact_matches / total_samples if total_samples > 0 else 0
        metrics['ai_answer_extraction_rate'] = len(ai_answers) / total_samples if total_samples > 0 else 0
        metrics['code_answer_generation_rate'] = len(code_answers) / total_samples if total_samples > 0 else 0
        
        # 性能统计
        metrics['average_generation_time'] = self.verification_stats['generation_time'] / total_samples if total_samples > 0 else 0
        metrics['average_verification_time'] = self.verification_stats['verification_time'] / len(verification_results) if verification_results else 0
        
        # 错误分析
        status_counts = {}
        for result in verification_results:
            status = result.get('status', 'unknown')
            status_counts[status] = status_counts.get(status, 0) + 1
        metrics['status_distribution'] = status_counts
        
        return metrics
    
    def _generate_evaluation_report(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """生成评估报告"""
        report = {
            'model_path': self.args.model_path,
            'test_dataset': self.args.test_dataset,
            'evaluation_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'evaluation_settings': {
                'verification_sample_size': self.args.verification_sample_size,
                'max_new_tokens': self.args.max_new_tokens,
                'temperature': self.args.temperature,
                'top_p': self.args.top_p
            },
            'metrics': metrics,
            'summary': self._generate_summary(metrics)
        }
        
        return report
    
    def _generate_summary(self, metrics: Dict[str, Any]) -> Dict[str, str]:
        """生成评估摘要"""
        verification_rate = metrics.get('verification_rate', 0)
        exact_match_rate = metrics.get('exact_match_rate', 0)
        average_confidence = metrics.get('average_confidence', 0)
        
        # 评估等级
        if verification_rate > 0.8 and exact_match_rate > 0.7:
            grade = "A (优秀)"
        elif verification_rate > 0.6 and exact_match_rate > 0.5:
            grade = "B (良好)"
        elif verification_rate > 0.4 and exact_match_rate > 0.3:
            grade = "C (一般)"
        else:
            grade = "D (需要改进)"
        
        summary = {
            'overall_grade': grade,
            'verification_performance': f"{verification_rate:.1%}",
            'accuracy_performance': f"{exact_match_rate:.1%}",
            'confidence_level': f"{average_confidence:.3f}",
            'recommendation': self._get_recommendation(verification_rate, exact_match_rate, average_confidence)
        }
        
        return summary
    
    def _get_recommendation(self, verification_rate: float, exact_match_rate: float, confidence: float) -> str:
        """生成改进建议"""
        if verification_rate < 0.3:
            return "建议增加代码验证相关的训练数据，提高模型生成可验证回答的能力"
        elif exact_match_rate < 0.4:
            return "建议优化答案提取和格式化，提高答案准确性"
        elif confidence < 0.6:
            return "建议调整训练参数，提高模型回答的置信度"
        else:
            return "模型表现良好，可考虑在更复杂的数学问题上进行测试"
    
    def _save_report(self, report: Dict[str, Any]):
        """保存评估报告"""
        output_path = Path(self.args.output_report)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 评估报告已保存: {output_path}")
        
        # 打印摘要
        self._print_summary(report)
    
    def _print_summary(self, report: Dict[str, Any]):
        """打印评估摘要"""
        metrics = report['metrics']
        summary = report['summary']
        
        print(f"\n" + "="*70)
        print(f"📊 CCPO模型评估报告")
        print(f"="*70)
        print(f"模型路径: {report['model_path']}")
        print(f"评估时间: {report['evaluation_timestamp']}")
        print(f"样本数量: {metrics['total_samples']}")
        
        print(f"\n🎯 核心指标:")
        print(f"   代码验证成功率: {summary['verification_performance']}")
        print(f"   答案准确率: {summary['accuracy_performance']}")
        print(f"   平均置信度: {summary['confidence_level']}")
        print(f"   综合评级: {summary['overall_grade']}")
        
        print(f"\n📈 详细统计:")
        print(f"   验证样本数: {metrics['verified_samples']}/{metrics['total_samples']}")
        print(f"   高置信度比例: {metrics.get('high_confidence_rate', 0):.1%}")
        print(f"   AI答案提取率: {metrics.get('ai_answer_extraction_rate', 0):.1%}")
        print(f"   代码答案生成率: {metrics.get('code_answer_generation_rate', 0):.1%}")
        
        print(f"\n⏱️  性能统计:")
        print(f"   平均生成时间: {metrics.get('average_generation_time', 0):.2f}秒/样本")
        print(f"   平均验证时间: {metrics.get('average_verification_time', 0):.2f}秒/样本")
        
        print(f"\n💡 改进建议:")
        print(f"   {summary['recommendation']}")
        
        print(f"="*70)


async def main():
    """主函数"""
    args = parse_arguments()
    
    logger.info(f"🚀 CCPO模型评估器启动")
    logger.info(f"模型: {args.model_path}")
    logger.info(f"测试集: {args.test_dataset}")
    logger.info(f"验证样本数: {args.verification_sample_size}")
    
    evaluator = CCPOModelEvaluator(args)
    
    try:
        report = await evaluator.evaluate_model()
        logger.info(f"✅ 评估完成!")
        return 0
        
    except Exception as e:
        logger.error(f"❌ 评估失败: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(asyncio.run(main()))