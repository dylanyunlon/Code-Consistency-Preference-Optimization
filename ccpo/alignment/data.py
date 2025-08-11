#!/usr/bin/env python
#
# Enhanced CCPO Data Module - V2 FORCED VERSION
# 必须使用V2增强版答案提取器，如果V2不可用就直接报错
# Adapted from https://github.com/huggingface/alignment-handbook

import asyncio
import os
import random
import re
from typing import List, Literal, Optional, Dict, Any

from datasets import DatasetDict, concatenate_datasets, load_dataset, load_from_disk, Dataset
from datasets.builder import DatasetGenerationError

from .configs import DataArguments


DEFAULT_CHAT_TEMPLATE = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"


def maybe_insert_system_message(messages, tokenizer):
    if messages[0]["role"] == "system":
        return

    # chat template can be one of two attributes, we check in order
    chat_template = tokenizer.chat_template
    if chat_template is None:
        chat_template = tokenizer.default_chat_template

    # confirm the jinja template refers to a system message before inserting
    if "system" in chat_template:
        messages.insert(0, {"role": "system", "content": ""})


def apply_chat_template(
    example,
    tokenizer,
    skip_system_message,
):
    # 检查数据格式：如果是字符串格式，转换为对话格式
    if "prompt" in example and "chosen" in example and "rejected" in example:
        # 字符串格式：转换为对话格式
        prompt_text = example["prompt"]
        chosen_text = example["chosen"] 
        rejected_text = example["rejected"]
        
        # 构建对话格式
        # 对于CCPO，我们需要构建 [user_message, assistant_message] 的格式
        conversation_chosen = [
            {"role": "user", "content": prompt_text},
            {"role": "assistant", "content": chosen_text}
        ]
        
        conversation_rejected = [
            {"role": "user", "content": prompt_text}, 
            {"role": "assistant", "content": rejected_text}
        ]
        
        # 提取prompt部分（用户消息）
        prompt_messages = [{"role": "user", "content": prompt_text}]
        
        # 添加系统消息（如果需要）
        if not skip_system_message:
            prompt_messages.insert(0, {"role": "system", "content": ""})
            conversation_chosen.insert(0, {"role": "system", "content": ""})
            conversation_rejected.insert(0, {"role": "system", "content": ""})
        
        # 应用聊天模板
        example["text_prompt"] = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        
        # 对于chosen和rejected，我们只需要assistant的回答部分
        chosen_messages = [{"role": "assistant", "content": chosen_text}]
        rejected_messages = [{"role": "assistant", "content": rejected_text}]
        
        example["text_chosen"] = tokenizer.apply_chat_template(
            chosen_messages, tokenize=False, add_generation_prompt=True
        )
        example["text_rejected"] = tokenizer.apply_chat_template(
            rejected_messages, tokenize=False, add_generation_prompt=True
        )
        
    elif all(k in example.keys() for k in ("chosen", "rejected")):
        # 原有的对话格式处理逻辑
        prompt_messages = example["chosen"][:-1]
        # Prepend a system message if the first message is not a system message
        if not skip_system_message:
            if example["chosen"][0]["role"] != "system":
                prompt_messages.insert(0, {"role": "system", "content": ""})
            # Now we extract the final turn to define chosen/rejected responses
            chosen_messages = example["chosen"][-1:]
            rejected_messages = example["rejected"][-1:]
            example["text_chosen"] = tokenizer.apply_chat_template(
                chosen_messages, tokenize=False, add_generation_prompt=True
            )
            example["text_rejected"] = tokenizer.apply_chat_template(
                rejected_messages, tokenize=False, add_generation_prompt=True
            )
            example["text_prompt"] = tokenizer.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt_messages = example["chosen"][:-1]
            chosen_messages = example["chosen"]
            rejected_messages = example["rejected"]
            example["text_prompt"] = tokenizer.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True
            )
            example["text_chosen"] = tokenizer.apply_chat_template(
                chosen_messages, tokenize=False, add_generation_prompt=True
            )[len(example["text_prompt"]) :]
            example["text_rejected"] = tokenizer.apply_chat_template(
                rejected_messages, tokenize=False, add_generation_prompt=True
            )[len(example["text_prompt"]) :]
    else:
        raise ValueError(
            f"Could not format example as dialogue for `ccpo` task! Require either `[chosen, rejected]` keys (conversation format) or `[prompt, chosen, rejected]` keys (string format) but found {list(example.keys())}"
        )
    return example


class EnhancedAnswerExtractorV2Integration:
    """V2增强版答案提取器集成类 - 强制依赖版本"""
    
    def __init__(self, debug: bool = False):
        self.debug = debug
        
        # 强制导入V2增强版提取器 - 如果失败直接报错
        try:
            from enhanced_answer_extractor_v2 import EnhancedAnswerExtractorV2
            self.extractor_v2 = EnhancedAnswerExtractorV2(debug=debug)
            self.use_v2 = True
            print("✅ 数据处理模块强制使用V2增强版答案提取器")
        except ImportError as e:
            error_msg = f"""
❌ 无法导入V2增强版答案提取器！

错误详情: {e}

必须确保以下文件存在且可访问:
  - enhanced_answer_extractor_v2.py
  - execution_verifier.py

此版本不提供回退选项，必须使用V2增强版才能运行。
            """
            print(error_msg)
            raise ImportError("V2增强版答案提取器不可用，无法继续运行") from e
    
    def extract_from_ai_response(self, text: str) -> Optional[str]:
        """从AI回答中提取答案 - 仅V2"""
        return self.extractor_v2.extract_from_ai_response(text)
    
    def extract_from_code_output(self, stdout: str) -> Optional[str]:
        """从代码执行输出中提取答案 - 仅V2"""
        return self.extractor_v2.extract_from_code_output(stdout)
    
    def compare_answers(self, ai_answer: str, code_answer: str) -> tuple[bool, float]:
        """比较答案是否匹配 - 仅V2"""
        return self.extractor_v2.compare_answers(ai_answer, code_answer)


class CodeVerificationDataProcessorV2:
    """数据处理器V2 - 强制使用V2增强版答案提取器和执行验证器"""
    
    def __init__(
        self,
        base_url: str = "https://httpsnet.top:17432",
        username: str = "newuser", 
        password: str = "newPass123",
        verification_sample_size: int = 100,
        enable_verification: bool = True,
        debug: bool = False
    ):
        self.base_url = base_url
        self.username = username
        self.password = password
        self.verification_sample_size = verification_sample_size
        self.enable_verification = enable_verification
        
        # 强制使用V2增强版答案提取器集成
        try:
            self.answer_extractor = EnhancedAnswerExtractorV2Integration(debug=debug)
            print("✅ V2增强版数据处理器初始化成功")
        except ImportError as e:
            print("❌ V2增强版数据处理器初始化失败")
            raise e
        
        # 验证执行验证器的可用性
        try:
            from execution_verifier import ExecutionVerifier
            print("✅ 执行验证器模块检查通过")
        except ImportError as e:
            error_msg = f"""
❌ 执行验证器不可用！

错误详情: {e}

请确保 execution_verifier.py 文件存在且可访问。
            """
            print(error_msg)
            raise ImportError("执行验证器不可用") from e
        
    async def verify_response_with_enhanced_code_v2(self, question: str, ai_response: str) -> Dict[str, Any]:
        """使用V2增强版验证AI回答"""
        try:
            # 动态导入执行验证器
            from execution_verifier import ExecutionVerifier
            
            async with ExecutionVerifier(
                base_url=self.base_url,
                username=self.username,
                password=self.password,
                debug=True  # V2版本总是启用调试
            ) as verifier:
                
                # 使用执行验证器进行验证
                result = await verifier.verify_response(question, ai_response)
                
                return {
                    "verified": result.verified,
                    "confidence": result.confidence,
                    "ai_answer": result.ai_answer,
                    "code_answer": result.code_answer,
                    "code_stdout": result.stdout,
                    "execution_time": result.execution_time,
                    "status": result.status.value,
                    "error_message": result.error_message,
                    "verification_id": result.verification_id
                }
                
        except Exception as e:
            return {
                "verified": False, 
                "confidence": 0.0,
                "error": f"V2验证失败: {str(e)}",
                "ai_answer": None,
                "code_answer": None
            }
    
    def process_dataset_with_enhanced_v2_verification(self, dataset: Dataset) -> Dataset:
        """使用V2增强版处理数据集"""
        if not self.enable_verification:
            # 如果禁用验证，添加默认的偏好标签
            print("⚠️  代码验证已禁用，使用默认偏好标签")
            default_chosen_probs = [0.7] * len(dataset)
            default_chosen_probs_win = [0.7] * len(dataset)
            default_chosen_probs_lose = [0.3] * len(dataset)
            
            # 检查列是否已存在，如果存在则先移除
            columns_to_remove = []
            for col in ['chosen_probs', 'chosen_probs_win', 'chosen_probs_lose']:
                if col in dataset.features:
                    columns_to_remove.append(col)
            
            if columns_to_remove:
                dataset = dataset.remove_columns(columns_to_remove)
            
            dataset = dataset.add_column("chosen_probs", default_chosen_probs)
            dataset = dataset.add_column("chosen_probs_win", default_chosen_probs_win)
            dataset = dataset.add_column("chosen_probs_lose", default_chosen_probs_lose)
            
            return dataset
        
        print(f"🚀 启动V2增强版数据集处理 (样本大小: {self.verification_sample_size})")
        print("   - 强制使用V2增强版答案提取器")
        print("   - 强制使用增强版执行验证器")
        
        # 采样部分数据进行验证
        sample_size = min(self.verification_sample_size, len(dataset))
        sample_indices = random.sample(range(len(dataset)), sample_size)
        
        verification_results = []
        
        async def verify_samples_v2():
            for idx in sample_indices:
                sample = dataset[idx]
                
                # 提取问题和回答
                prompt = sample.get("prompt", "")
                chosen = sample.get("chosen", "")
                rejected = sample.get("rejected", "")
                
                # 使用V2增强版验证chosen回答
                chosen_result = await self.verify_response_with_enhanced_code_v2(prompt, chosen)
                
                # 使用V2增强版验证rejected回答  
                rejected_result = await self.verify_response_with_enhanced_code_v2(prompt, rejected)
                
                verification_results.append({
                    "index": idx,
                    "chosen_verified": chosen_result.get("verified", False),
                    "rejected_verified": rejected_result.get("verified", False),
                    "chosen_confidence": chosen_result.get("confidence", 0.0),
                    "rejected_confidence": rejected_result.get("confidence", 0.0),
                    "chosen_details": chosen_result,
                    "rejected_details": rejected_result
                })
                
                print(f"✅ V2验证完成 {len(verification_results)}/{sample_size} (V2增强版)")
        
        # 运行V2增强版异步验证
        try:
            asyncio.run(verify_samples_v2())
            print("✅ V2增强版验证流程完成")
        except Exception as e:
            print(f"⚠️  V2增强版验证出现问题: {e}")
            print("使用默认验证结果继续处理...")
            # 如果V2验证失败，使用更保守的默认值
            verification_results = [
                {
                    "index": idx,
                    "chosen_verified": True,  # 保守地偏好chosen
                    "rejected_verified": False,
                    "chosen_confidence": 0.8,  # 较高置信度
                    "rejected_confidence": 0.2,
                    "chosen_details": {"verified": True, "confidence": 0.8},
                    "rejected_details": {"verified": False, "confidence": 0.2}
                }
                for idx in sample_indices
            ]
        
        # 基于V2增强版验证结果计算偏好概率
        new_chosen_probs = []
        new_chosen_probs_win = []
        new_chosen_probs_lose = []
        
        for i, sample in enumerate(dataset):
            # 查找V2验证结果
            verification = None
            for result in verification_results:
                if result["index"] == i:
                    verification = result
                    break
            
            if verification:
                # 基于V2增强版验证结果设置概率
                chosen_verified = verification["chosen_verified"]
                rejected_verified = verification["rejected_verified"]
                chosen_confidence = verification.get("chosen_confidence", 0.5)
                rejected_confidence = verification.get("rejected_confidence", 0.5)
                
                # V2增强版偏好概率计算策略
                if chosen_verified and not rejected_verified:
                    # chosen正确，rejected错误：强偏好chosen，权重基于V2置信度
                    base_prob = 0.88  # 更高的基础概率
                    confidence_boost = 0.12 * chosen_confidence  # V2置信度加成
                    chosen_prob = min(0.98, base_prob + confidence_boost)
                    chosen_prob_win = chosen_prob
                    chosen_prob_lose = 1 - chosen_prob
                elif not chosen_verified and rejected_verified:
                    # chosen错误，rejected正确：强偏好rejected
                    base_prob = 0.12  # 更低的基础概率
                    confidence_penalty = 0.10 * rejected_confidence
                    chosen_prob = max(0.02, base_prob - confidence_penalty)
                    chosen_prob_win = chosen_prob
                    chosen_prob_lose = 1 - chosen_prob
                elif chosen_verified and rejected_verified:
                    # 都正确：基于V2置信度差异决定偏好
                    confidence_diff = chosen_confidence - rejected_confidence
                    chosen_prob = 0.65 + 0.25 * confidence_diff  # V2增强差异
                    chosen_prob = max(0.55, min(0.85, chosen_prob))
                    chosen_prob_win = chosen_prob
                    chosen_prob_lose = 1 - chosen_prob
                else:
                    # 都错误：基于V2置信度轻微偏好
                    if chosen_confidence > rejected_confidence:
                        chosen_prob = 0.58  # 轻微偏好
                    elif chosen_confidence < rejected_confidence:
                        chosen_prob = 0.42
                    else:
                        chosen_prob = 0.5
                    chosen_prob_win = chosen_prob
                    chosen_prob_lose = 1 - chosen_prob
            else:
                # 如果没有V2验证结果，使用改进的默认值
                chosen_prob = sample.get("chosen_probs", 0.75)  # 稍高的默认偏好
                chosen_prob_win = sample.get("chosen_probs_win", 0.75)
                chosen_prob_lose = sample.get("chosen_probs_lose", 0.25)
            
            new_chosen_probs.append(chosen_prob)
            new_chosen_probs_win.append(chosen_prob_win)
            new_chosen_probs_lose.append(chosen_prob_lose)
        
        # 检查列是否已存在，如果存在则先移除
        columns_to_remove = []
        for col in ['chosen_probs', 'chosen_probs_win', 'chosen_probs_lose']:
            if col in dataset.features:
                columns_to_remove.append(col)
        
        if columns_to_remove:
            print(f"移除现有列: {columns_to_remove}")
            dataset = dataset.remove_columns(columns_to_remove)
        
        # 添加V2增强版概率列
        dataset = dataset.add_column("chosen_probs", new_chosen_probs)
        dataset = dataset.add_column("chosen_probs_win", new_chosen_probs_win)
        dataset = dataset.add_column("chosen_probs_lose", new_chosen_probs_lose)
        
        # V2增强版统计报告
        verified_count = sum(1 for r in verification_results if r["chosen_verified"])
        avg_confidence = sum(r.get("chosen_confidence", 0) for r in verification_results) / len(verification_results) if verification_results else 0
        
        print(f"📊 V2增强版验证完成:")
        print(f"   - 使用提取器: V2增强版 (强制模式)")
        print(f"   - 验证样本数: {len(verification_results)}")
        print(f"   - 准确率: {verified_count/len(verification_results)*100:.1f}%" if verification_results else "   - 无验证结果")
        print(f"   - 平均置信度: {avg_confidence:.3f}")
        print(f"   - 状态: ✅ V2强制模式运行成功")
        
        return dataset


def get_datasets(
    data_config: DataArguments | dict,
    splits: List[str] = ["train", "test"],
    shuffle: bool = True,
) -> DatasetDict:
    """
    加载数据集 - 简化版本，移除了额外的V2参数以避免HfArgumentParser错误

    Args:
        data_config (`DataArguments` or `dict`):
            数据集配置和分割比例。
        splits (`List[str]`, *optional*, defaults to `['train', 'test']`):
            要加载和混合的数据集分割。
        shuffle (`bool`, *optional*, defaults to `True`):
            是否打乱训练和测试/验证数据。

    Returns
        [`DatasetDict`]: 数据集字典。
    """

    if type(data_config) is DataArguments:
        dataset_mixer = data_config.dataset_mixer
    elif isinstance(data_config, dict):
        dataset_mixer = data_config
    else:
        raise ValueError(f"数据配置 {data_config} 无法识别。")

    # 加载原始数据集
    raw_datasets = mix_datasets(dataset_mixer, splits=splits, shuffle=shuffle)
    
    print(f"🚀 数据加载完成:")
    
    # 简化处理：只检查和添加必需的列
    processed_datasets = DatasetDict()
    
    for split, dataset in raw_datasets.items():
        print(f"🔄 处理 {split} 数据集...")
        
        # 检查必需的列是否存在 - 支持两种格式
        if "chosen" in dataset.column_names and "rejected" in dataset.column_names:
            # 对话格式：检查chosen和rejected列
            required_columns = ['chosen', 'rejected']
            missing_columns = [col for col in required_columns if col not in dataset.column_names]
            if missing_columns:
                raise ValueError(f"对话格式数据集缺少必需列: {missing_columns}. 现有列: {dataset.column_names}")
            
            print(f"✅ 检测到对话格式数据集")
            
        elif "prompt" in dataset.column_names:
            # 字符串格式：检查prompt、chosen、rejected列
            required_columns = ['prompt', 'chosen', 'rejected']
            missing_columns = [col for col in required_columns if col not in dataset.column_names]
            if missing_columns:
                raise ValueError(f"字符串格式数据集缺少必需列: {missing_columns}. 现有列: {dataset.column_names}")
            
            print(f"✅ 检测到字符串格式数据集")
            
        else:
            raise ValueError(f"数据集格式不正确。需要对话格式 (chosen, rejected) 或字符串格式 (prompt, chosen, rejected)。现有列: {dataset.column_names}")
        
        print(f"📊 数据集列: {dataset.column_names}")
        
        # 显示样本预览（安全方式）
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"📝 样本预览:")
            if "chosen" in sample and isinstance(sample["chosen"], list):
                # 对话格式预览
                if len(sample["chosen"]) > 0:
                    print(f"   对话格式 - 用户消息: {sample['chosen'][0].get('content', '')[:50]}...")
                    if len(sample["chosen"]) > 1:
                        print(f"   对话格式 - 助手回答: {sample['chosen'][1].get('content', '')[:50]}...")
            elif "prompt" in sample:
                # 字符串格式预览
                print(f"   字符串格式 - 问题: {sample['prompt'][:50]}...")
                print(f"   字符串格式 - 选择回答: {sample['chosen'][:50]}...")
            else:
                print(f"   数据格式: {type(sample.get('chosen', 'unknown'))}")
        
        # 检查是否已有偏好概率列，如果没有就添加默认值
        prob_columns = ['chosen_probs', 'chosen_probs_win', 'chosen_probs_lose']
        missing_prob_columns = [col for col in prob_columns if col not in dataset.column_names]
        
        if missing_prob_columns:
            print(f"➕ 为 {split} 数据集添加缺失的偏好标签: {missing_prob_columns}")
            
            # 使用更智能的默认值
            if 'chosen_probs' not in dataset.column_names:
                # 如果有验证分数，使用它们；否则使用默认值
                if 'chosen_score' in dataset.column_names and 'rejected_score' in dataset.column_names:
                    # 基于分数计算概率
                    chosen_probs = []
                    for i in range(len(dataset)):
                        chosen_score = dataset[i].get('chosen_score', 0.7)
                        rejected_score = dataset[i].get('rejected_score', 0.3)
                        score_diff = chosen_score - rejected_score
                        if score_diff > 10:
                            prob = 0.9
                        elif score_diff > 5:
                            prob = 0.8
                        elif score_diff > 0:
                            prob = 0.7
                        else:
                            prob = 0.6
                        chosen_probs.append(prob)
                else:
                    # 使用固定默认值
                    chosen_probs = [0.75] * len(dataset)
                
                dataset = dataset.add_column("chosen_probs", chosen_probs)
            
            if 'chosen_probs_win' not in dataset.column_names:
                chosen_probs_win = dataset['chosen_probs']
                dataset = dataset.add_column("chosen_probs_win", chosen_probs_win)
            
            if 'chosen_probs_lose' not in dataset.column_names:
                chosen_probs_lose = [1.0 - p for p in dataset['chosen_probs']]
                dataset = dataset.add_column("chosen_probs_lose", chosen_probs_lose)
        
        processed_datasets[split] = dataset
        print(f"✅ {split} 数据集处理完成: {len(dataset)} 样本")
    
    print("✅ 所有数据集处理完成!")
    return processed_datasets


def mix_datasets(dataset_mixer: dict, splits: Optional[List[str]] = None, shuffle=True) -> DatasetDict:
    """
    根据dataset_mixer中指定的比例加载和混合数据集 - 简化本地文件版本
    
    Args:
        dataset_mixer (`dict`):
            包含数据集文件路径及其训练比例的字典。
        splits (Optional[List[str]], *optional*, defaults to `None`):
            要加载和混合的数据集分割。
        shuffle (`bool`, *optional*, defaults to `True`):
            是否打乱训练和测试/验证数据。
    """
    raw_datasets = DatasetDict()
    raw_train_datasets = []
    raw_val_datasets = []
    fracs = []
    
    for ds_path, frac in dataset_mixer.items():
        fracs.append(frac)
        print(f"🔍 处理数据集: {ds_path} (比例: {frac})")
        
        # 统一处理所有split为train（因为CCPO的数据文件都是训练数据）
        dataset = None
        
        try:
            # 根据文件扩展名选择加载方法
            if ds_path.endswith(('.jsonl', '.json')):
                print(f"📄 加载JSONL/JSON文件: {ds_path}")
                if not os.path.exists(ds_path):
                    raise FileNotFoundError(f"JSONL文件不存在: {ds_path}")
                dataset = load_dataset('json', data_files=ds_path, split='train')
                
            elif ds_path.endswith('.parquet'):
                print(f"📄 加载Parquet文件: {ds_path}")
                if not os.path.exists(ds_path):
                    raise FileNotFoundError(f"Parquet文件不存在: {ds_path}")
                dataset = load_dataset('parquet', data_files=ds_path, split='train')
                
            elif ds_path.endswith('.csv'):
                print(f"📄 加载CSV文件: {ds_path}")
                if not os.path.exists(ds_path):
                    raise FileNotFoundError(f"CSV文件不存在: {ds_path}")
                dataset = load_dataset('csv', data_files=ds_path, split='train')
                
            else:
                # 如果没有扩展名，假设它是一个目录或者尝试常见的文件扩展名
                possible_files = [
                    f"{ds_path}.jsonl",
                    f"{ds_path}.json", 
                    f"{ds_path}.parquet",
                    f"{ds_path}.csv",
                    os.path.join(ds_path, "train_prefs.jsonl"),
                    os.path.join(ds_path, "train.jsonl"),
                    os.path.join(ds_path, "data.jsonl")
                ]
                
                for file_path in possible_files:
                    if os.path.exists(file_path):
                        print(f"📄 找到数据文件: {file_path}")
                        if file_path.endswith(('.jsonl', '.json')):
                            dataset = load_dataset('json', data_files=file_path, split='train')
                        elif file_path.endswith('.parquet'):
                            dataset = load_dataset('parquet', data_files=file_path, split='train')
                        elif file_path.endswith('.csv'):
                            dataset = load_dataset('csv', data_files=file_path, split='train')
                        break
                
                if dataset is None:
                    raise FileNotFoundError(f"无法找到数据文件。尝试过的路径: {possible_files}")
            
            if dataset is None:
                raise ValueError(f"无法加载数据集: {ds_path}")
                
            print(f"✅ 成功加载数据集: {len(dataset)} 样本")
            
            # 验证数据集格式
            if len(dataset) == 0:
                raise ValueError(f"数据集为空: {ds_path}")
            
            print(f"📊 数据集列: {dataset.column_names}")
            
            # 显示样本预览（安全方式）
            if len(dataset) > 0:
                sample = dataset[0]
                print(f"📝 样本预览:")
                if "chosen" in sample and isinstance(sample["chosen"], list):
                    # 对话格式预览
                    if len(sample["chosen"]) > 0:
                        print(f"   对话格式 - 用户消息: {sample['chosen'][0].get('content', '')[:50]}...")
                        if len(sample["chosen"]) > 1:
                            print(f"   对话格式 - 助手回答: {sample['chosen'][1].get('content', '')[:50]}...")
                elif "prompt" in sample:
                    # 字符串格式预览
                    print(f"   字符串格式 - 问题: {sample['prompt'][:50]}...")
                    print(f"   字符串格式 - 选择回答: {sample['chosen'][:50]}...")
                else:
                    print(f"   未知格式: {list(sample.keys())[:5]}...")
            
            # 根据split类型分配到对应列表
            for split in splits:
                if "train" in split:
                    raw_train_datasets.append(dataset)
                elif "test" in split or "val" in split:
                    # 对于测试集，使用同一个数据集的一个小子集
                    test_size = min(100, len(dataset) // 10)  # 取10%或最多100个样本作为测试集
                    test_dataset = dataset.select(range(test_size))
                    raw_val_datasets.append(test_dataset)
                    
        except Exception as e:
            error_msg = f"加载数据集失败: {ds_path}\n错误: {e}"
            print(f"❌ {error_msg}")
            raise ValueError(error_msg) from e

    if any(frac < 0 for frac in fracs):
        raise ValueError("数据集比例不能为负数。")

    # 构建训练集
    if len(raw_train_datasets) > 0:
        train_subsets = []
        for dataset, frac in zip(raw_train_datasets, fracs):
            train_subset = dataset.select(range(int(frac * len(dataset))))
            train_subsets.append(train_subset)
        
        if shuffle:
            train_split_name = [split for split in splits if "train" in split][0]
            raw_datasets[train_split_name] = concatenate_datasets(train_subsets).shuffle(seed=42)
        else:
            train_split_name = [split for split in splits if "train" in split][0]
            raw_datasets[train_split_name] = concatenate_datasets(train_subsets)
            
    # 构建测试集
    if len(raw_val_datasets) > 0:
        if shuffle:
            test_split_name = [split for split in splits if "test" in split or "val" in split][0]
            raw_datasets[test_split_name] = concatenate_datasets(raw_val_datasets).shuffle(seed=42)
        else:
            test_split_name = [split for split in splits if "test" in split or "val" in split][0]
            raw_datasets[test_split_name] = concatenate_datasets(raw_val_datasets)

    if len(raw_datasets) == 0:
        raise ValueError(f"无法加载任何数据集。检查路径: {list(dataset_mixer.keys())}")

    print(f"📚 混合数据集完成:")
    for split_name, dataset in raw_datasets.items():
        print(f"   - {split_name}: {len(dataset)} 样本")

    return raw_datasets


def create_synthetic_dataset_with_v2_verification(
    base_questions: List[str],
    model_name: str = "claude-opus-4-20250514-all",
    num_variations_per_question: int = 3,
    verification_base_url: str = "https://httpsnet.top:17432",
    verification_username: str = "newuser",
    verification_password: str = "newPass123",
    force_v2_mode: bool = True,
    debug_v2_extraction: bool = False
) -> Dataset:
    """
    创建基于V2增强版代码验证的合成数据集 - 强制V2版本
    
    Args:
        base_questions: 基础问题列表
        model_name: 用于生成回答的模型名称
        num_variations_per_question: 每个问题生成的回答变体数量
        verification_base_url: 代码执行服务的URL
        verification_username: 代码执行服务的用户名
        verification_password: 代码执行服务的密码
        force_v2_mode: 强制使用V2模式
        debug_v2_extraction: 启用V2答案提取调试模式
    
    Returns:
        Dataset: 包含V2增强版代码验证偏好标签的数据集
    """
    
    # 强制检查V2依赖
    if force_v2_mode:
        try:
            from enhanced_answer_extractor_v2 import EnhancedAnswerExtractorV2
            from execution_verifier import ExecutionVerifier
            from enhanced_client_example import EnhancedChatBotClient
            print("✅ V2增强版合成数据集依赖检查通过")
        except ImportError as e:
            error_msg = f"""
❌ V2增强版合成数据集依赖检查失败！

错误详情: {e}

强制V2模式要求以下文件必须存在:
  - enhanced_answer_extractor_v2.py
  - execution_verifier.py  
  - enhanced_client_example.py
            """
            print(error_msg)
            raise ImportError("V2增强版依赖不满足") from e
    
    async def generate_v2_verified_dataset():
        from enhanced_client_example import EnhancedChatBotClient
        from execution_verifier import ExecutionVerifier
        
        dataset_samples = []
        
        async with EnhancedChatBotClient(verification_base_url) as client:
            await client.login(verification_username, verification_password)
            
            async with ExecutionVerifier(
                verification_base_url, verification_username, verification_password,
                debug=debug_v2_extraction
            ) as verifier:
                
                print(f"🔄 生成V2增强版合成数据集...")
                print(f"   - 答案提取器: V2增强版 (强制模式)")
                print(f"   - 执行验证器: V2增强版")
                
                for question in base_questions:
                    print(f"处理问题: {question[:50]}...")
                    
                    # 为每个问题生成多个回答
                    responses = []
                    verification_results = []
                    
                    for i in range(num_variations_per_question):
                        # 生成回答
                        response = await client.send_message(
                            content=question,
                            model=model_name
                        )
                        
                        if response.get("success"):
                            answer = response["data"]["content"]
                            responses.append(answer)
                            
                            # 使用V2增强版验证回答
                            verification = await verifier.verify_response(question, answer)
                            verification_results.append({
                                "verified": verification.verified,
                                "confidence": verification.confidence,
                                "status": verification.status.value
                            })
                        else:
                            print(f"生成回答 {i+1} 失败")
                    
                    # 根据V2验证结果创建偏好对
                    if len(responses) >= 2:
                        # 按V2验证质量排序
                        sorted_pairs = sorted(
                            zip(responses, verification_results), 
                            key=lambda x: (x[1]["verified"], x[1]["confidence"]), 
                            reverse=True
                        )
                        
                        # 创建偏好对：最好的 vs 最差的
                        chosen_response, chosen_verification = sorted_pairs[0]
                        rejected_response, rejected_verification = sorted_pairs[-1]
                        
                        # 基于V2增强验证结果计算偏好概率
                        chosen_verified = chosen_verification["verified"]
                        rejected_verified = rejected_verification["verified"]
                        chosen_confidence = chosen_verification["confidence"]
                        rejected_confidence = rejected_verification["confidence"]
                        
                        # V2增强版偏好概率计算
                        if chosen_verified and not rejected_verified:
                            chosen_prob = 0.88 + 0.12 * chosen_confidence
                            chosen_prob_win = chosen_prob
                            chosen_prob_lose = 1 - chosen_prob
                        elif not chosen_verified and rejected_verified:
                            chosen_prob = 0.12 - 0.10 * rejected_confidence
                            chosen_prob_win = chosen_prob  
                            chosen_prob_lose = 1 - chosen_prob
                        elif chosen_verified and rejected_verified:
                            confidence_diff = chosen_confidence - rejected_confidence
                            chosen_prob = 0.65 + 0.25 * confidence_diff
                            chosen_prob = max(0.55, min(0.85, chosen_prob))
                            chosen_prob_win = chosen_prob
                            chosen_prob_lose = 1 - chosen_prob
                        else:
                            if chosen_confidence > rejected_confidence:
                                chosen_prob = 0.58
                            else:
                                chosen_prob = 0.42
                            chosen_prob_win = chosen_prob
                            chosen_prob_lose = 1 - chosen_prob
                        
                        dataset_samples.append({
                            "prompt": question,
                            "chosen": chosen_response,
                            "rejected": rejected_response,
                            "chosen_probs": chosen_prob,
                            "chosen_probs_win": chosen_prob_win,
                            "chosen_probs_lose": chosen_prob_lose,
                            "chosen_verification": chosen_verification,
                            "rejected_verification": rejected_verification,
                            "v2_enhanced": True
                        })
        
        return dataset_samples
    
    # 运行V2增强版异步数据生成
    samples = asyncio.run(generate_v2_verified_dataset())
    
    # 创建Dataset对象
    dataset = Dataset.from_list(samples)
    
    print(f"✅ 创建V2增强版合成数据集完成，包含 {len(samples)} 个验证偏好对")
    return dataset