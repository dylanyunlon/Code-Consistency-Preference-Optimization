#!/usr/bin/env python3
"""
Quick Start for Code-Verified CCPO Training
代码验证增强CCPO训练的快速启动脚本
"""

import asyncio
import argparse
import logging
import os
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from enhanced_ccpo_runner import EnhancedCCPORunner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Quick Start for Code-Verified CCPO Training")
    
    # 运行模式
    parser.add_argument("--mode", choices=["demo", "quick_train", "full_train", "eval"], 
                       default="demo", help="运行模式")
    
    # 模型相关
    parser.add_argument("--model", type=str, 
                       default="/data/jiacheng/dylan/aaai/Code-Consistency-Preference-Optimization/cppo/mistralai/Mistral-7B-Instruct-v0.2",
                       help="基础模型路径")
    parser.add_argument("--model_path", type=str,
                       help="已训练模型路径（用于评估）")
    
    # 训练参数
    parser.add_argument("--max_iterations", type=int, default=3,
                       help="最大训练迭代次数")
    parser.add_argument("--batch_size", type=int, default=2,
                       help="批大小")
    parser.add_argument("--learning_rate", type=float, default=5e-7,
                       help="学习率")
    parser.add_argument("--beta", type=float, default=0.01,
                       help="CCPO beta参数")
    
    # 验证服务器配置
    parser.add_argument("--verification_url", type=str,
                       default="https://8.134.217.190:17432",
                       help="代码验证服务器URL")
    parser.add_argument("--verification_username", type=str,
                       default="newuser",
                       help="验证服务器用户名")
    parser.add_argument("--verification_password", type=str,
                       default="newPass123",
                       help="验证服务器密码")
    
    # 输出配置
    parser.add_argument("--output_dir", type=str,
                       default="./checkpoints/code-verified-ccpo-test",
                       help="输出目录")
    parser.add_argument("--config_file", type=str,
                       default="config.yaml",
                       help="配置文件路径")
    
    return parser.parse_args()

async def demo_mode():
    """演示模式 - 测试代码验证功能"""
    print("🧪 代码验证演示模式")
    print("=" * 50)
    
    # 测试验证器连接
    try:
        from execution_verifier import ExecutionVerifier
        
        async with ExecutionVerifier(
            base_url="https://8.134.217.190:17432",
            username="newuser",
            password="newPass123",
            debug=True
        ) as verifier:
            print("✅ 验证器连接成功")
            
            # 测试单个验证
            test_question = "strawberry中有几个字母r？"
            test_response = "strawberry中有3个字母r"
            
            print(f"测试问题: {test_question}")
            print(f"测试回答: {test_response}")
            print("正在验证...")
            
            result = await verifier.verify_response(test_question, test_response)
            
            print(f"\n验证结果:")
            print(f"  验证通过: {'✅' if result.verified else '❌'}")
            print(f"  置信度: {result.confidence:.3f}")
            print(f"  AI答案: {result.ai_answer}")
            print(f"  代码答案: {result.code_answer}")
            print(f"  执行状态: {result.status.value}")
            print(f"  执行时间: {result.execution_time:.2f}s")
            
            if result.stdout:
                print(f"  执行输出预览: {result.stdout[:200]}...")
    
    except Exception as e:
        print(f"❌ 演示失败: {e}")
        return False
    
    print("\n✅ 演示完成！验证系统工作正常。")
    return True

async def quick_train_mode(args):
    """快速训练模式 - 小规模测试训练"""
    print("🚀 快速训练模式")
    print("=" * 50)
    
    # 创建临时配置
    temp_config = {
        "model": {
            "name_or_path": args.model,
            "fp16": True
        },
        "data": {
            "dataset_mixer": {"HuggingFaceH4/ultrafeedback_binarized": 1.0}
        },
        "training": {
            "output_dir": args.output_dir,
            "num_train_epochs": 1,
            "per_device_train_batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "beta": args.beta,
            "max_length": 1024,
            "max_prompt_length": 512,
            "loss_type": "code_verified"
        },
        "verification": {
            "enable": True,
            "base_url": args.verification_url,
            "username": args.verification_username,
            "password": args.verification_password,
            "sample_size": 50  # 小样本快速测试
        }
    }
    
    # 保存临时配置
    import yaml
    temp_config_path = "temp_quick_config.yaml"
    with open(temp_config_path, 'w') as f:
        yaml.dump(temp_config, f, default_flow_style=False)
    
    try:
        # 运行快速训练
        runner = EnhancedCCPORunner(temp_config_path)
        model, tokenizer, metrics = await runner.run_code_verified_training()
        
        print("✅ 快速训练完成！")
        print(f"训练指标: {metrics}")
        
        return True
        
    except Exception as e:
        print(f"❌ 快速训练失败: {e}")
        return False
    finally:
        # 清理临时文件
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)

async def full_train_mode(args):
    """完整训练模式 - 使用配置文件进行完整训练"""
    print("🎯 完整训练模式")
    print("=" * 50)
    
    if not os.path.exists(args.config_file):
        print(f"❌ 配置文件不存在: {args.config_file}")
        print("使用 --mode demo 生成默认配置文件")
        return False
    
    try:
        runner = EnhancedCCPORunner(args.config_file)
        
        if args.max_iterations > 1:
            # 运行自我进化训练
            print(f"开始自我进化训练，最大迭代次数: {args.max_iterations}")
            model, tokenizer, history = await runner.run_self_evolution(args.max_iterations)
            
            print("✅ 自我进化训练完成！")
            print("进化历史:")
            for i, metrics in enumerate(history):
                print(f"  迭代 {i+1}: 准确率={metrics.answer_accuracy_rate:.3f}, "
                     f"成功率={metrics.execution_success_rate:.3f}")
        else:
            # 运行单次训练
            print("开始单次代码验证训练")
            model, tokenizer, metrics = await runner.run_code_verified_training()
            
            print("✅ 代码验证训练完成！")
            print(f"训练指标: {metrics}")
        
        return True
        
    except Exception as e:
        print(f"❌ 完整训练失败: {e}")
        return False

async def eval_mode(args):
    """评估模式 - 评估已训练的模型"""
    print("📊 评估模式")
    print("=" * 50)
    
    if not args.model_path:
        print("❌ 评估模式需要指定 --model_path")
        return False
    
    if not os.path.exists(args.model_path):
        print(f"❌ 模型路径不存在: {args.model_path}")
        return False
    
    try:
        runner = EnhancedCCPORunner(args.config_file)
        metrics = await runner.run_evaluation(args.model_path)
        
        print("✅ 评估完成！")
        print(f"执行成功率: {metrics['execution_success_rate']:.3f}")
        print(f"答案准确率: {metrics['answer_accuracy_rate']:.3f}")
        print(f"平均置信度: {metrics['average_confidence']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ 评估失败: {e}")
        return False

def create_default_config(args):
    """创建默认配置文件"""
    from enhanced_ccpo_runner import create_default_config
    
    config = create_default_config()
    
    # 根据命令行参数更新配置
    config["model"]["name_or_path"] = args.model
    config["training"]["output_dir"] = args.output_dir
    config["training"]["per_device_train_batch_size"] = args.batch_size
    config["training"]["learning_rate"] = args.learning_rate
    config["training"]["beta"] = args.beta
    config["verification"]["base_url"] = args.verification_url
    config["verification"]["username"] = args.verification_username
    config["verification"]["password"] = args.verification_password
    
    return config

def print_usage_tips():
    """打印使用提示"""
    print("\n💡 使用提示:")
    print("1. 首次使用，先运行演示模式测试环境:")
    print("   python quick_start_code_verified.py --mode demo")
    print()
    print("2. 快速测试训练（小样本）:")
    print("   python quick_start_code_verified.py --mode quick_train")
    print()
    print("3. 完整训练（需要配置文件）:")
    print("   python quick_start_code_verified.py --mode full_train --config_file config.yaml")
    print()
    print("4. 模型评估:")
    print("   python quick_start_code_verified.py --mode eval --model_path ./checkpoints/model")
    print()
    print("5. 自我进化训练:")
    print("   python quick_start_code_verified.py --mode full_train --max_iterations 5")

async def main():
    """主函数"""
    args = parse_arguments()
    
    print("🤖 代码验证增强CCPO训练 - 快速启动")
    print("=" * 60)
    print(f"运行模式: {args.mode}")
    print(f"基础模型: {args.model}")
    print(f"输出目录: {args.output_dir}")
    print(f"验证服务器: {args.verification_url}")
    print("=" * 60)
    
    success = False
    
    try:
        if args.mode == "demo":
            success = await demo_mode()
            
        elif args.mode == "quick_train":
            success = await quick_train_mode(args)
            
        elif args.mode == "full_train":
            success = await full_train_mode(args)
            
        elif args.mode == "eval":
            success = await eval_mode(args)
        
        else:
            print(f"❌ 未知的运行模式: {args.mode}")
    
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断操作")
    except Exception as e:
        print(f"❌ 运行出错: {e}")
        import traceback
        traceback.print_exc()
    
    if success:
        print("\n🎉 操作成功完成！")
        
        # 如果是演示模式，提供后续建议
        if args.mode == "demo":
            print("\n📝 后续步骤建议:")
            print("1. 运行快速训练测试: --mode quick_train")
            print("2. 创建完整配置文件并运行完整训练")
            print("3. 使用自我进化模式进行多轮优化")
        
        elif args.mode == "quick_train":
            print(f"\n📁 训练结果保存在: {args.output_dir}")
            print("💡 可以使用 --mode eval 评估模型性能")
        
        elif args.mode == "full_train":
            print(f"\n📁 训练结果保存在: {args.output_dir}")
            print("📊 建议运行评估检查模型性能")
        
        elif args.mode == "eval":
            print("\n📈 评估结果已保存到模型目录")
    else:
        print("\n❌ 操作失败")
        print_usage_tips()

if __name__ == "__main__":
    asyncio.run(main())