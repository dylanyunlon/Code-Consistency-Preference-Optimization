#!/usr/bin/env python3
"""
Enhanced SPPO Runner - 增强版SPPO训练启动脚本
整合代码验证和自我进化功能
"""

import os
import sys
import asyncio
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset, load_dataset

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from execution_verifier import ExecutionVerifier, batch_verify_responses
from self_evolution_controller import (
    SelfEvolutionController, 
    EvolutionConfig, 
    EvolutionMetrics
)

# 导入原有的SPPO组件
from sppo.alignment import (
    DataArguments,
    SPPOConfig,
    H4ArgumentParser,
    ModelArguments,
    get_datasets,
    get_tokenizer,
)
from sppo.trainer import SPPOTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedSPPORunner:
    """增强版SPPO训练运行器"""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self._load_config()
        self.evolution_controller = None
        
    def _load_config(self) -> Dict:
        """加载配置文件"""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            if self.config_path.endswith('.yaml') or self.config_path.endswith('.yml'):
                return yaml.safe_load(f)
            else:
                return json.load(f)
    
    def setup_evolution_controller(self):
        """设置进化控制器"""
        evolution_config = EvolutionConfig(
            min_samples_per_iteration=self.config.get('evolution', {}).get('min_samples_per_iteration', 100),
            max_samples_per_iteration=self.config.get('evolution', {}).get('max_samples_per_iteration', 1000),
            verification_batch_size=self.config.get('evolution', {}).get('verification_batch_size', 10),
            min_execution_success_rate=self.config.get('evolution', {}).get('min_execution_success_rate', 0.6),
            min_accuracy_improvement=self.config.get('evolution', {}).get('min_accuracy_improvement', 0.02),
            confidence_threshold=self.config.get('evolution', {}).get('confidence_threshold', 0.8),
            training_epochs_per_iteration=self.config.get('evolution', {}).get('training_epochs_per_iteration', 1),
            max_model_versions=self.config.get('evolution', {}).get('max_model_versions', 10),
            max_continuous_failures=self.config.get('evolution', {}).get('max_continuous_failures', 3)
        )
        
        verification_config = {
            'base_url': self.config.get('verification', {}).get('base_url', 'https://8.134.217.190:17432'),
            'username': self.config.get('verification', {}).get('username', 'newuser'),
            'password': self.config.get('verification', {}).get('password', 'newPass123'),
        }
        
        self.evolution_controller = SelfEvolutionController(
            config=evolution_config,
            model_base_dir=self.config.get('evolution', {}).get('model_base_dir', './evolution_models'),
            verification_config=verification_config
        )
    
    async def run_code_verified_training(self):
        """运行代码验证的SPPO训练"""
        logger.info("Starting Code-Verified SPPO Training")
        
        # 1. 解析参数（模拟命令行参数）
        model_args, data_args, training_args = self._setup_training_args()
        
        # 2. 加载tokenizer
        tokenizer = get_tokenizer(model_args, data_args)
        
        # 3. 加载和处理数据集（集成代码验证）
        logger.info("Loading and processing datasets with code verification...")
        raw_datasets = get_datasets(
            data_args,
            splits=["train_prefs"],
            enable_code_verification=self.config.get('verification', {}).get('enable', True),
            verification_base_url=self.config.get('verification', {}).get('base_url', 'https://8.134.217.190:17432'),
            verification_username=self.config.get('verification', {}).get('username', 'newuser'),
            verification_password=self.config.get('verification', {}).get('password', 'newPass123'),
            verification_sample_size=self.config.get('verification', {}).get('sample_size', 100)
        )
        
        # 4. 加载模型
        logger.info("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            torch_dtype=torch.float16 if self.config.get('model', {}).get('fp16', False) else torch.float32,
            device_map="auto"
        )
        
        # 5. 初始化SPPO训练器
        logger.info("Initializing SPPO trainer with code verification...")
        trainer = SPPOTrainer(
            model=model,
            args=training_args,
            train_dataset=raw_datasets["train_prefs"],
            tokenizer=tokenizer,
            beta=training_args.beta,
            max_length=training_args.max_length,
            max_prompt_length=training_args.max_prompt_length,
            loss_type="code_verified",  # 使用代码验证的损失函数
            enable_code_verification=True,
            verification_base_url=self.config.get('verification', {}).get('base_url', 'https://8.134.217.190:17432'),
            verification_username=self.config.get('verification', {}).get('username', 'newuser'),
            verification_password=self.config.get('verification', {}).get('password', 'newPass123'),
            precompute_ref_log_probs=True  # 节省内存
        )
        
        # 6. 开始训练
        logger.info("Starting training...")
        train_result = trainer.train()
        
        # 7. 保存模型
        output_dir = training_args.output_dir
        logger.info(f"Saving model to {output_dir}")
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        # 8. 保存训练指标
        metrics = train_result.metrics
        with open(os.path.join(output_dir, "training_metrics.json"), 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info("Code-Verified SPPO Training completed!")
        return model, tokenizer, metrics
    
    async def run_self_evolution(self, max_iterations: int = 10):
        """运行自我进化训练"""
        if not self.evolution_controller:
            self.setup_evolution_controller()
        
        logger.info(f"Starting Self-Evolution Training for {max_iterations} iterations")
        
        # 1. 加载初始模型
        model_args, data_args, training_args = self._setup_training_args()
        tokenizer = get_tokenizer(model_args, data_args)
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            torch_dtype=torch.float16 if self.config.get('model', {}).get('fp16', False) else torch.float32,
            device_map="auto"
        )
        
        # 2. 准备初始问题集
        initial_questions = self._load_initial_questions()
        
        # 3. 运行自我进化
        evolved_model, evolved_tokenizer, metrics_history = await self.evolution_controller.evolve_model(
            model=model,
            tokenizer=tokenizer,
            initial_questions=initial_questions,
            max_iterations=max_iterations
        )
        
        # 4. 保存最终模型
        final_output_dir = self.config.get('evolution', {}).get('final_output_dir', './final_evolved_model')
        Path(final_output_dir).mkdir(parents=True, exist_ok=True)
        
        evolved_model.save_pretrained(final_output_dir)
        evolved_tokenizer.save_pretrained(final_output_dir)
        
        # 5. 保存进化历史
        evolution_history = [metrics.__dict__ for metrics in metrics_history]
        with open(os.path.join(final_output_dir, "evolution_history.json"), 'w') as f:
            json.dump(evolution_history, f, indent=2)
        
        logger.info("Self-Evolution Training completed!")
        return evolved_model, evolved_tokenizer, metrics_history
    
    async def run_evaluation(self, model_path: str, test_questions: List[str] = None):
        """运行模型评估"""
        logger.info(f"Evaluating model at {model_path}")
        
        # 加载模型
        model = AutoModelForCausalLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # 准备测试问题
        if test_questions is None:
            test_questions = self._load_test_questions()
        
        # 生成回答
        responses = []
        for question in test_questions:
            messages = [{"role": "user", "content": question}]
            input_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(input_text, return_tensors="pt")
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            response = tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:], 
                skip_special_tokens=True
            )
            responses.append(response.strip())
        
        # 验证回答
        verification_config = {
            'base_url': self.config.get('verification', {}).get('base_url', 'https://8.134.217.190:17432'),
            'username': self.config.get('verification', {}).get('username', 'newuser'),
            'password': self.config.get('verification', {}).get('password', 'newPass123'),
        }
        
        async with ExecutionVerifier(**verification_config) as verifier:
            verification_results = await batch_verify_responses(
                test_questions, responses, verifier
            )
        
        # 计算评估指标
        total_questions = len(test_questions)
        successful_executions = sum(1 for r in verification_results if r.status.value == "success")
        verified_answers = sum(1 for r in verification_results if r.verified)
        
        evaluation_metrics = {
            'total_questions': total_questions,
            'execution_success_rate': successful_executions / total_questions,
            'answer_accuracy_rate': verified_answers / total_questions,
            'average_confidence': sum(r.confidence for r in verification_results) / total_questions,
            'detailed_results': [
                {
                    'question': q,
                    'response': r,
                    'verified': vr.verified,
                    'confidence': vr.confidence,
                    'ai_answer': vr.ai_answer,
                    'code_answer': vr.code_answer
                }
                for q, r, vr in zip(test_questions, responses, verification_results)
            ]
        }
        
        # 保存评估结果
        eval_output_path = f"{model_path}/evaluation_results.json"
        with open(eval_output_path, 'w') as f:
            json.dump(evaluation_metrics, f, indent=2)
        
        logger.info(f"Evaluation completed. Results saved to {eval_output_path}")
        logger.info(f"Execution Success Rate: {evaluation_metrics['execution_success_rate']:.3f}")
        logger.info(f"Answer Accuracy Rate: {evaluation_metrics['answer_accuracy_rate']:.3f}")
        
        return evaluation_metrics
    
    def _setup_training_args(self):
        """设置训练参数"""
        # 从配置创建参数对象
        model_config = self.config.get('model', {})
        data_config = self.config.get('data', {})
        training_config = self.config.get('training', {})
        
        # 模型参数
        model_args = ModelArguments(
            model_name_or_path=model_config.get('name_or_path', 'mistralai/Mistral-7B-Instruct-v0.2'),
            model_revision=model_config.get('revision', 'main'),
            torch_dtype=model_config.get('torch_dtype', 'auto'),
            use_flash_attention_2=model_config.get('use_flash_attention_2', False),
            trust_remote_code=model_config.get('trust_remote_code', False)
        )
        
        # 数据参数
        data_args = DataArguments(
            dataset_mixer=data_config.get('dataset_mixer', {"HuggingFaceH4/ultrafeedback_binarized": 1.0}),
            dataset_splits=data_config.get('dataset_splits', ["train_prefs", "test_prefs"]),
            preprocessing_num_workers=data_config.get('preprocessing_num_workers', 4)
        )
        
        # 训练参数
        training_args = SPPOConfig(
            output_dir=training_config.get('output_dir', './checkpoints/code-verified-sppo'),
            num_train_epochs=training_config.get('num_train_epochs', 1),
            per_device_train_batch_size=training_config.get('per_device_train_batch_size', 2),
            per_device_eval_batch_size=training_config.get('per_device_eval_batch_size', 4),
            gradient_accumulation_steps=training_config.get('gradient_accumulation_steps', 8),
            learning_rate=training_config.get('learning_rate', 5e-7),
            max_length=training_config.get('max_length', 1024),
            max_prompt_length=training_config.get('max_prompt_length', 512),
            beta=training_config.get('beta', 0.01),
            loss_type=training_config.get('loss_type', 'code_verified'),
            gradient_checkpointing=training_config.get('gradient_checkpointing', True),
            logging_steps=training_config.get('logging_steps', 10),
            save_steps=training_config.get('save_steps', 500),
            eval_steps=training_config.get('eval_steps', 500),
            warmup_steps=training_config.get('warmup_steps', 100),
            remove_unused_columns=False,
            fp16=training_config.get('fp16', True),
            dataloader_num_workers=training_config.get('dataloader_num_workers', 4),
            report_to=training_config.get('report_to', None)
        )
        
        return model_args, data_args, training_args
    
    def _load_initial_questions(self) -> List[str]:
        """加载初始问题集"""
        questions_file = self.config.get('evolution', {}).get('initial_questions_file')
        if questions_file and os.path.exists(questions_file):
            with open(questions_file, 'r', encoding='utf-8') as f:
                if questions_file.endswith('.json'):
                    return json.load(f)
                else:
                    return [line.strip() for line in f if line.strip()]
        
        # 默认问题集
        return [
            "strawberry中有几个字母r？",
            "计算 23 + 45 × 2 的结果",
            "一个长方形的长是8米，宽是5米，它的面积是多少平方米？",
            "从1到50的所有奇数的和是多少？",
            "一个圆的直径是14米，它的面积是多少平方米？（π取3.14）",
            "如果一个数的3倍加上7等于22，这个数是多少？",
            "一辆车以60公里/小时的速度行驶3小时，它行驶了多少公里？",
            "一个班级有30个学生，其中40%是女生，女生有多少人？",
            "计算 5! (5的阶乘) 的值",
            "解方程 2x + 5 = 17，x等于多少？"
        ]
    
    def _load_test_questions(self) -> List[str]:
        """加载测试问题集"""
        test_file = self.config.get('evaluation', {}).get('test_questions_file')
        if test_file and os.path.exists(test_file):
            with open(test_file, 'r', encoding='utf-8') as f:
                if test_file.endswith('.json'):
                    return json.load(f)
                else:
                    return [line.strip() for line in f if line.strip()]
        
        # 使用初始问题作为测试集
        return self._load_initial_questions()[:5]  # 取前5个作为测试

def create_default_config():
    """创建默认配置文件"""
    default_config = {
        "model": {
            "name_or_path": "mistralai/Mistral-7B-Instruct-v0.2",
            "revision": "main",
            "torch_dtype": "auto",
            "use_flash_attention_2": False,
            "trust_remote_code": False,
            "fp16": True
        },
        "data": {
            "dataset_mixer": {
                "HuggingFaceH4/ultrafeedback_binarized": 1.0
            },
            "dataset_splits": ["train_prefs", "test_prefs"],
            "preprocessing_num_workers": 4
        },
        "training": {
            "output_dir": "./checkpoints/code-verified-sppo",
            "num_train_epochs": 1,
            "per_device_train_batch_size": 2,
            "per_device_eval_batch_size": 4,
            "gradient_accumulation_steps": 8,
            "learning_rate": 5e-7,
            "max_length": 1024,
            "max_prompt_length": 512,
            "beta": 0.01,
            "loss_type": "code_verified",
            "gradient_checkpointing": True,
            "logging_steps": 10,
            "save_steps": 500,
            "eval_steps": 500,
            "warmup_steps": 100,
            "fp16": True,
            "dataloader_num_workers": 4
        },
        "verification": {
            "enable": True,
            "base_url": "https://8.134.217.190:17432",
            "username": "newuser",
            "password": "newPass123",
            "sample_size": 100
        },
        "evolution": {
            "min_samples_per_iteration": 50,
            "max_samples_per_iteration": 500,
            "verification_batch_size": 5,
            "min_execution_success_rate": 0.6,
            "min_accuracy_improvement": 0.01,
            "confidence_threshold": 0.8,
            "training_epochs_per_iteration": 1,
            "max_model_versions": 10,
            "max_continuous_failures": 3,
            "model_base_dir": "./evolution_models",
            "final_output_dir": "./final_evolved_model"
        },
        "evaluation": {
            "test_questions_file": null
        }
    }
    
    return default_config

async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Enhanced SPPO Training with Code Verification")
    parser.add_argument("--mode", choices=["train", "evolve", "eval"], required=True,
                      help="运行模式: train=基础训练, evolve=自我进化, eval=评估")
    parser.add_argument("--config", type=str, default="config.yaml",
                      help="配置文件路径")
    parser.add_argument("--model_path", type=str,
                      help="评估模式下的模型路径")
    parser.add_argument("--max_iterations", type=int, default=10,
                      help="自我进化的最大迭代次数")
    parser.add_argument("--create_config", action="store_true",
                      help="创建默认配置文件")
    
    args = parser.parse_args()
    
    # 创建默认配置
    if args.create_config:
        config = create_default_config()
        with open("default_config.yaml", 'w') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        print("默认配置文件已创建: default_config.yaml")
        return
    
    # 检查配置文件
    if not os.path.exists(args.config):
        print(f"配置文件不存在: {args.config}")
        print("使用 --create_config 创建默认配置文件")
        return
    
    # 创建运行器
    runner = EnhancedSPPORunner(args.config)
    
    try:
        if args.mode == "train":
            # 基础SPPO训练
            model, tokenizer, metrics = await runner.run_code_verified_training()
            print("训练完成!")
            print(f"训练指标: {metrics}")
            
        elif args.mode == "evolve":
            # 自我进化训练
            model, tokenizer, history = await runner.run_self_evolution(args.max_iterations)
            print("自我进化完成!")
            print("进化历史:")
            for i, metrics in enumerate(history):
                print(f"迭代 {i+1}: 准确率={metrics.answer_accuracy_rate:.3f}, "
                     f"成功率={metrics.execution_success_rate:.3f}")
            
        elif args.mode == "eval":
            # 模型评估
            if not args.model_path:
                print("评估模式需要指定 --model_path")
                return
            
            metrics = await runner.run_evaluation(args.model_path)
            print("评估完成!")
            print(f"执行成功率: {metrics['execution_success_rate']:.3f}")
            print(f"答案准确率: {metrics['answer_accuracy_rate']:.3f}")
            print(f"平均置信度: {metrics['average_confidence']:.3f}")
    
    except Exception as e:
        logger.error(f"运行出错: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())