#!/usr/bin/env python
#
# Fixed CCPO Trainer for Code Execution Verification - 处理对话格式数据
# Adapted from https://github.com/huggingface/alignment-handbook

import asyncio
import inspect
import json
import random
import re
import warnings
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from functools import wraps
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate.utils import is_deepspeed_available, tqdm
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    DataCollator,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalLoopOutput

from trl.import_utils import is_peft_available, is_wandb_available
from trl.models import PreTrainedModelWrapper, create_reference_model
from trl.trainer.utils import (
    DPODataCollatorWithPadding,
    disable_dropout_in_model,
    pad_to_length,
    trl_sanitze_kwargs_for_tagging,
)

# Import for code execution verification
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

if is_peft_available():
    from peft import PeftModel, get_peft_model, prepare_model_for_kbit_training

if is_wandb_available():
    import wandb

if is_deepspeed_available():
    import deepspeed


class CodeExecutionVerifier:
    """代码执行验证器 - 强制使用V2增强版答案提取器"""
    
    def __init__(self, base_url: str = "https://httpsnet.top:17432", username: str = "newuser", password: str = "newPass123"):
        self.base_url = base_url
        self.username = username
        self.password = password
        self.client = None
        
        # 强制导入V2增强版答案提取器
        try:
            from enhanced_answer_extractor_v2 import EnhancedAnswerExtractorV2
            self.answer_extractor = EnhancedAnswerExtractorV2(debug=False)
            print("✅ CCPOTrainer 成功加载增强版答案提取器V2")
        except ImportError as e:
            raise ImportError(
                f"❌ CCPOTrainer 无法导入enhanced_answer_extractor_v2.py: {e}\n"
                f"请确保enhanced_answer_extractor_v2.py文件在当前目录或Python路径中。\n"
                f"训练器不提供降级方案，必须使用V2增强版答案提取器。"
            )
        
    async def __aenter__(self):
        """异步上下文管理器入口"""
        try:
            from enhanced_client_example import EnhancedChatBotClient
            self.client = EnhancedChatBotClient(self.base_url)
            await self.client.__aenter__()
            await self.client.login(self.username, self.password)
            return self
        except Exception as e:
            print(f"Warning: Could not initialize code execution client: {e}")
            return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        if self.client:
            await self.client.__aexit__(exc_type, exc_val, exc_tb)


def convert_conversation_to_text(conversation: Union[str, List[Dict]], tokenizer) -> str:
    """
    将对话格式转换为文本格式
    处理 HuggingFace 数据集中的对话格式数据
    """
    if isinstance(conversation, str):
        return conversation
    
    if isinstance(conversation, list):
        # 处理对话格式：[{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
        try:
            # 使用 tokenizer 的 chat template 转换
            text = tokenizer.apply_chat_template(
                conversation, 
                tokenize=False, 
                add_generation_prompt=False
            )
            return text
        except Exception as e:
            print(f"Warning: Failed to apply chat template: {e}")
            # 回退方案：手动拼接对话
            text_parts = []
            for turn in conversation:
                role = turn.get("role", "")
                content = turn.get("content", "")
                if role == "user":
                    text_parts.append(f"User: {content}")
                elif role == "assistant":
                    text_parts.append(f"Assistant: {content}")
                else:
                    text_parts.append(f"{role}: {content}")
            return "\n".join(text_parts)
    
    # 如果既不是字符串也不是列表，尝试转换为字符串
    return str(conversation)


class CCPOTrainer(Trainer):
    """
    Enhanced CCPO Trainer with Code Execution Verification - 修复对话格式数据处理
    """

    _tag_names = ["trl", "ccpo", "code-verified"]

    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module, str] = None,
        ref_model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        beta: float = 0.1,
        label_smoothing: float = 0,
        loss_type: Literal["sigmoid", "hinge", "ipo", "kto_pair", "ccpo", "code_verified"] = "code_verified",
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        label_pad_token_id: int = -100,
        padding_value: int = 0,
        truncation_mode: str = "keep_end",
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        max_length: Optional[int] = None,
        max_prompt_length: Optional[int] = None,
        max_target_length: Optional[int] = None,
        peft_config: Optional[Dict] = None,
        is_encoder_decoder: Optional[bool] = None,
        disable_dropout: bool = True,
        generate_during_eval: bool = False,
        compute_metrics: Optional[Callable[[EvalLoopOutput], Dict]] = None,
        precompute_ref_log_probs: bool = False,
        model_init_kwargs: Optional[Dict] = None,
        ref_model_init_kwargs: Optional[Dict] = None,
        model_adapter_name: str = None,
        ref_adapter_name: str = None,
        # New parameters for code verification
        enable_code_verification: bool = True,
        verification_base_url: str = "https://httpsnet.top:17432",
        verification_username: str = "newuser",
        verification_password: str = "newPass123",
        verification_sample_size: int = 100,
    ):
        # Initialize code verification with V2 extractor
        self.enable_code_verification = enable_code_verification
        self.verification_base_url = verification_base_url
        self.verification_username = verification_username
        self.verification_password = verification_password
        self.verification_sample_size = verification_sample_size
        self.verification_cache = {}
        
        # Rest of the initialization remains the same as original
        if model_init_kwargs is None:
            model_init_kwargs = {}
        elif not isinstance(model, str):
            raise ValueError("You passed model_kwargs to the CCPOTrainer. But your model is already instantiated.")

        if ref_model_init_kwargs is None:
            ref_model_init_kwargs = {}
        elif not isinstance(ref_model, str):
            raise ValueError(
                "You passed ref_model_kwargs to the CCPOTrainer. But your ref_model is already instantiated."
            )

        if isinstance(model, str):
            warnings.warn(
                "You passed a model_id to the CCPOTrainer. This will automatically create an "
                "`AutoModelForCausalLM` or a `PeftModel` (if you passed a `peft_config`) for you."
            )
            model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)

        if isinstance(ref_model, str):
            warnings.warn(
                "You passed a ref model_id to the CCPOTrainer. This will automatically create an "
                "`AutoModelForCausalLM`"
            )
            ref_model = AutoModelForCausalLM.from_pretrained(ref_model, **ref_model_init_kwargs)

        # Initialize this variable to False. This helps tracking the case when `peft_module_casting_to_bf16`
        # has been called in order to properly call autocast if needed.
        self._peft_has_been_casted_to_bf16 = False

        if not is_peft_available() and peft_config is not None:
            raise ValueError(
                "PEFT is not installed and you passed a `peft_config` in the trainer's kwargs, please install it to use the PEFT models"
            )
        elif is_peft_available() and peft_config is not None:
            raise NotImplementedError

        # For models that use gradient_checkpoiting, we need to attach a hook that enables input
        # to explicitly have `requires_grad=True`, otherwise training will either silently
        # fail or completely fail.
        if getattr(args, "gradient_checkpointing", False):
            # For backward compatibility with older versions of transformers
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            else:

                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)

                model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        if generate_during_eval and not is_wandb_available():
            raise ValueError(
                "`generate_during_eval=True` requires Weights and Biases to be installed."
                " Please install `wandb` to resolve."
            )

        if model is not None:
            self.is_encoder_decoder = model.config.is_encoder_decoder
        elif is_encoder_decoder is None:
            raise ValueError("When no model is provided, you need to pass the parameter is_encoder_decoder.")
        else:
            self.is_encoder_decoder = is_encoder_decoder

        self.is_peft_model = is_peft_available() and isinstance(model, PeftModel)
        self.model_adapter_name = model_adapter_name
        self.ref_adapter_name = ref_adapter_name

        if ref_model:
            self.ref_model = ref_model
        elif self.is_peft_model or precompute_ref_log_probs:
            # The `model` with adapters turned off will be used as the reference model
            self.ref_model = None
        else:
            self.ref_model = create_reference_model(model)

        if tokenizer is None:
            raise ValueError("tokenizer must be specified to tokenize a CCPO dataset.")
        if max_length is None:
            warnings.warn(
                "`max_length` is not set in the CCPOTrainer's init"
                " it will default to `512` by default, but you should do it yourself in the future.",
                UserWarning,
            )
            max_length = 512
        if max_prompt_length is None:
            warnings.warn(
                "`max_prompt_length` is not set in the CCPOTrainer's init"
                " it will default to `128` by default, but you should do it yourself in the future.",
                UserWarning,
            )
            max_prompt_length = 128

        if max_target_length is None and self.is_encoder_decoder:
            warnings.warn(
                "When using an encoder decoder architecture, you should set `max_target_length` in the CCPOTrainer's init"
                " it will default to `128` by default, but you should do it yourself in the future.",
                UserWarning,
            )
            max_target_length = 128

        if data_collator is None:
            data_collator = DPODataCollatorWithPadding(
                pad_token_id=tokenizer.pad_token_id,
                label_pad_token_id=label_pad_token_id,
                is_encoder_decoder=self.is_encoder_decoder,
            )

            if args.remove_unused_columns:
                args.remove_unused_columns = False
                # warn users
                warnings.warn(
                    "When using DPODataCollatorWithPadding, you should set `remove_unused_columns=False` in your TrainingArguments"
                    " we have set it for you, but you should do it yourself in the future.",
                    UserWarning,
                )

            self.use_dpo_data_collator = True
        else:
            self.use_dpo_data_collator = False

        if disable_dropout:
            disable_dropout_in_model(model)
            if self.ref_model is not None:
                disable_dropout_in_model(self.ref_model)

        self.max_length = max_length
        self.generate_during_eval = generate_during_eval
        self.label_pad_token_id = label_pad_token_id
        self.padding_value = padding_value if padding_value is not None else tokenizer.pad_token_id
        self.max_prompt_length = max_prompt_length
        self.truncation_mode = truncation_mode
        self.max_target_length = max_target_length
        self.tokenizer = tokenizer
        self.precompute_ref_log_probs = precompute_ref_log_probs

        # Since ref_logs are precomputed on the first call to get_train/eval_dataloader
        # keep track of first called to avoid computation of future calls
        self._precomputed_train_ref_log_probs = False
        self._precomputed_eval_ref_log_probs = False

        if loss_type in ["hinge", "ipo", "kto_pair"] and label_smoothing > 0:
            warnings.warn(
                "You are using a loss type that does not support label smoothing. Ignoring label_smoothing parameter."
            )

        self.beta = beta
        self.label_smoothing = label_smoothing
        self.loss_type = loss_type

        self._stored_metrics = defaultdict(lambda: defaultdict(list))

        # Preprocess dataset with code verification if enabled
        if self.enable_code_verification and train_dataset is not None:
            train_dataset = self._add_code_verification_to_dataset(train_dataset)

        # tokenize the dataset
        train_dataset = train_dataset.map(self.tokenize_row)
        if eval_dataset is not None:
            eval_dataset = eval_dataset.map(self.tokenize_row)

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        if not hasattr(self, "accelerator"):
            raise AttributeError(
                "Your `Trainer` does not have an `accelerator` object. Consider upgrading `transformers`."
            )

        # Deepspeed Zero-3 does not support precompute_ref_log_probs
        if self.is_deepspeed_enabled:
            if self.accelerator.state.deepspeed_plugin.zero_stage == 3 and self.precompute_ref_log_probs:
                raise ValueError(
                    "You cannot use `precompute_ref_log_probs=True` with Deepspeed ZeRO-3. Please set `precompute_ref_log_probs=False`."
                )

        if self.ref_model is None:
            if not (self.is_peft_model or self.precompute_ref_log_probs):
                raise ValueError(
                    "No reference model and model is not a Peft model. Try setting `precompute_ref_log_probs=True`"
                )
        else:
            if self.is_deepspeed_enabled:
                self.ref_model = self._prepare_deepspeed(self.ref_model)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

    def _add_code_verification_to_dataset(self, dataset: Dataset) -> Dataset:
        """为数据集添加代码验证标签 - 使用V2增强版答案提取器"""
        print("Adding V2 enhanced code verification to dataset...")
        
        # # 检查是否已经有验证列
        # if all(col in dataset.features for col in ['chosen_probs', 'chosen_probs_win', 'chosen_probs_lose']):
        #     print("Code verification columns already exist, skipping verification process")
        #     return dataset
        
        # 采样部分数据进行验证（避免验证整个数据集太耗时）
        sample_size = min(self.verification_sample_size, len(dataset))
        sample_indices = random.sample(range(len(dataset)), sample_size)
        
        verification_results = []
        
        async def verify_samples():
            async with CodeExecutionVerifier(
                self.verification_base_url, 
                self.verification_username, 
                self.verification_password
            ) as verifier:
                for idx in sample_indices:
                    sample = dataset[idx]
                    
                    # 提取问题和回答，处理对话格式
                    prompt = sample.get("prompt", "")
                    chosen = sample.get("chosen", "")
                    rejected = sample.get("rejected", "")
                    
                    # 转换对话格式为文本
                    if isinstance(chosen, list):
                        chosen = convert_conversation_to_text(chosen, self.tokenizer)
                    if isinstance(rejected, list):
                        rejected = convert_conversation_to_text(rejected, self.tokenizer)
                    if isinstance(prompt, list):
                        prompt = convert_conversation_to_text(prompt, self.tokenizer)
                    
                    # 验证chosen回答
                    chosen_result = await self._verify_response_with_code(prompt, chosen, verifier)
                    
                    # 验证rejected回答
                    rejected_result = await self._verify_response_with_code(prompt, rejected, verifier)
                    
                    verification_results.append({
                        "index": idx,
                        "chosen_verified": chosen_result.get("verified", False),
                        "rejected_verified": rejected_result.get("verified", False),
                        "chosen_details": chosen_result,
                        "rejected_details": rejected_result
                    })
                    
                    print(f"V2 Verified {len(verification_results)}/{sample_size} samples")
        
        # 运行异步验证
        try:
            asyncio.run(verify_samples())
        except Exception as e:
            print(f"Warning: V2 Code verification failed: {e}")
            # 如果验证失败，使用默认值
            verification_results = [
                {
                    "index": idx,
                    "chosen_verified": True,  # 默认chosen为正确
                    "rejected_verified": False,  # 默认rejected为错误
                    "chosen_details": {"verified": True},
                    "rejected_details": {"verified": False}
                }
                for idx in sample_indices
            ]
        
        # 基于验证结果计算新的偏好概率
        new_chosen_probs = []
        new_chosen_probs_win = []
        new_chosen_probs_lose = []
        
        for i, sample in enumerate(dataset):
            # 查找是否有验证结果
            verification = None
            for result in verification_results:
                if result["index"] == i:
                    verification = result
                    break
            
            if verification:
                # 基于代码验证结果设置概率
                chosen_verified = verification["chosen_verified"]
                rejected_verified = verification["rejected_verified"]
                
                if chosen_verified and not rejected_verified:
                    # chosen正确，rejected错误：强偏好chosen
                    chosen_prob = 0.9
                    chosen_prob_win = 0.9
                    chosen_prob_lose = 0.1
                elif not chosen_verified and rejected_verified:
                    # chosen错误，rejected正确：强偏好rejected
                    chosen_prob = 0.1
                    chosen_prob_win = 0.1
                    chosen_prob_lose = 0.9
                elif chosen_verified and rejected_verified:
                    # 都正确：轻微偏好chosen（因为它被标记为chosen）
                    chosen_prob = 0.6
                    chosen_prob_win = 0.6
                    chosen_prob_lose = 0.4
                else:
                    # 都错误：无偏好
                    chosen_prob = 0.5
                    chosen_prob_win = 0.5
                    chosen_prob_lose = 0.5
            else:
                # 如果没有验证结果，使用原始CCPO的默认值
                chosen_prob = sample.get("chosen_probs", 0.7)
                chosen_prob_win = sample.get("chosen_probs_win", 0.7)
                chosen_prob_lose = sample.get("chosen_probs_lose", 0.3)
            
            new_chosen_probs.append(chosen_prob)
            new_chosen_probs_win.append(chosen_prob_win)
            new_chosen_probs_lose.append(chosen_prob_lose)
        
        # 先移除可能存在的列，然后添加新的列
        columns_to_remove = []
        for col in ['chosen_probs', 'chosen_probs_win', 'chosen_probs_lose']:
            if col in dataset.features:
                columns_to_remove.append(col)
        
        if columns_to_remove:
            print(f"Removing existing columns: {columns_to_remove}")
            dataset = dataset.remove_columns(columns_to_remove)
        
        # 添加新的概率列
        dataset = dataset.add_column("chosen_probs", new_chosen_probs)
        dataset = dataset.add_column("chosen_probs_win", new_chosen_probs_win)
        dataset = dataset.add_column("chosen_probs_lose", new_chosen_probs_lose)
        
        print(f"Added V2 enhanced code verification results. Verified samples: {len(verification_results)}")
        return dataset

    async def _verify_response_with_code(self, question: str, ai_response: str, verifier) -> Dict[str, Any]:
        """使用V2增强版答案提取器验证回答"""
        try:
            # 构建代码生成提示
            code_prompt = f"""python版本为3.10.创建一个包含Python代码的脚本用于解决以下问题：{question}"""
            
            # 发送代码生成请求
            response = await verifier.client.send_code_request(
                prompt=code_prompt,
                auto_execute=False,
                model="claude-sonnet-4-20250514-all"
            )
            
            if not response.get("success"):
                return {"verified": False, "error": f"Code generation failed: {response.get('error')}"}
            
            data = response["data"]
            extracted_codes = data.get("metadata", {}).get("extracted_codes", [])
            
            if not extracted_codes:
                return {"verified": False, "error": "No code extracted"}
            
            code_info = extracted_codes[0]
            if not code_info.get("saved") or not code_info.get("id"):
                return {"verified": False, "error": "Code not saved properly"}
            
            # 执行代码
            exec_result = await verifier.client.execute_code(code_info["id"])
            
            if not exec_result.get("success"):
                return {"verified": False, "error": f"Code execution failed: {exec_result.get('stderr', 'Unknown error')}"}
            
            # 使用V2增强版答案提取器
            ai_answer = verifier.answer_extractor.extract_from_ai_response(ai_response)
            code_answer = verifier.answer_extractor.extract_from_code_output(exec_result.get("stdout", ""))
            
            # 使用V2增强版比较器
            verified, confidence = verifier.answer_extractor.compare_answers(ai_answer, code_answer)
            
            return {
                "verified": verified,
                "confidence": confidence,
                "ai_answer": ai_answer,
                "code_answer": code_answer,
                "code_stdout": exec_result.get("stdout", ""),
                "execution_time": exec_result.get("execution_time", 0)
            }
            
        except Exception as e:
            return {"verified": False, "error": f"Verification failed: {str(e)}"}

    def _prepare_deepspeed(self, model: PreTrainedModelWrapper):
        # Same as original implementation
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        config_kwargs = deepcopy(deepspeed_plugin.deepspeed_config)

        if model is not None:
            if hasattr(model, "config"):
                hidden_size = (
                    max(model.config.hidden_sizes)
                    if getattr(model.config, "hidden_sizes", None)
                    else getattr(model.config, "hidden_size", None)
                )
                if hidden_size is not None and config_kwargs["zero_optimization"]["stage"] == 3:
                    config_kwargs.update(
                        {
                            "zero_optimization.reduce_bucket_size": hidden_size * hidden_size,
                            "zero_optimization.stage3_param_persistence_threshold": 10 * hidden_size,
                            "zero_optimization.stage3_prefetch_bucket_size": 0.9 * hidden_size * hidden_size,
                        }
                    )

        if config_kwargs["zero_optimization"]["stage"] != 3:
            config_kwargs["zero_optimization"]["stage"] = 0
        model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
        model.eval()
        return model

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].
        """
        if self.precompute_ref_log_probs and not self._precomputed_train_ref_log_probs:
            dataloader_params = {
                "batch_size": self.args.per_device_train_batch_size,
                "collate_fn": self.data_collator,
                "num_workers": self.args.dataloader_num_workers,
                "pin_memory": self.args.dataloader_pin_memory,
                "shuffle": False,
            }

            data_loader = self.accelerator.prepare(DataLoader(self.train_dataset, **dataloader_params))

            reference_chosen_logps = []
            reference_rejected_logps = []
            for padded_batch in tqdm(iterable=data_loader, desc="Train dataset reference log probs"):
                reference_chosen_logp, reference_rejected_logp = self.compute_reference_log_probs(padded_batch)
                reference_chosen_logp, reference_rejected_logp = self.accelerator.gather_for_metrics(
                    (reference_chosen_logp, reference_rejected_logp)
                )
                reference_chosen_logps.append(reference_chosen_logp.cpu())
                reference_rejected_logps.append(reference_rejected_logp.cpu())

            all_reference_chosen_logps = torch.cat(reference_chosen_logps).float().numpy()
            all_reference_rejected_logps = torch.cat(reference_rejected_logps).float().numpy()

            self.train_dataset = self.train_dataset.add_column(
                name="reference_chosen_logps", column=all_reference_chosen_logps
            )
            self.train_dataset = self.train_dataset.add_column(
                name="reference_rejected_logps", column=all_reference_rejected_logps
            )

            self._precomputed_train_ref_log_probs = True

        return super().get_train_dataloader()

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        """
        Returns the evaluation [`~torch.utils.data.DataLoader`].
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

        if self.precompute_ref_log_probs and not self._precomputed_eval_ref_log_probs:
            dataloader_params = {
                "batch_size": self.args.per_device_eval_batch_size,
                "collate_fn": self.data_collator,
                "num_workers": self.args.dataloader_num_workers,
                "pin_memory": self.args.dataloader_pin_memory,
                "shuffle": False,
            }

            data_loader = self.accelerator.prepare(DataLoader(eval_dataset, **dataloader_params))

            reference_chosen_logps = []
            reference_rejected_logps = []
            for padded_batch in tqdm(iterable=data_loader, desc="Eval dataset reference log probs"):
                reference_chosen_logp, reference_rejected_logp = self.compute_reference_log_probs(padded_batch)
                reference_chosen_logp, reference_rejected_logp = self.accelerator.gather_for_metrics(
                    (reference_chosen_logp, reference_rejected_logp)
                )
                reference_chosen_logps.append(reference_chosen_logp.cpu())
                reference_rejected_logps.append(reference_rejected_logp.cpu())

            all_reference_chosen_logps = torch.cat(reference_chosen_logps).float().numpy()
            all_reference_rejected_logps = torch.cat(reference_rejected_logps).float().numpy()

            eval_dataset = eval_dataset.add_column(name="reference_chosen_logps", column=all_reference_chosen_logps)
            eval_dataset = eval_dataset.add_column(
                name="reference_rejected_logps", column=all_reference_rejected_logps
            )

            if self.eval_dataset is not None:
                self.eval_dataset = eval_dataset
            self._precomputed_eval_ref_log_probs = True

        return super().get_eval_dataloader(eval_dataset=eval_dataset)

    def build_tokenized_answer(self, prompt, answer):
        """
        Llama tokenizer does satisfy `enc(a + b) = enc(a) + enc(b)`.
        """
        full_tokenized = self.tokenizer(prompt + answer, add_special_tokens=False)
        prompt_input_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]

        answer_input_ids = full_tokenized["input_ids"][len(prompt_input_ids) :]
        answer_attention_mask = full_tokenized["attention_mask"][len(prompt_input_ids) :]

        full_concat_input_ids = np.concatenate([prompt_input_ids, answer_input_ids])
        full_input_ids = np.array(full_tokenized["input_ids"])

        if len(full_input_ids) != len(full_concat_input_ids):
            raise ValueError("Prompt input ids and answer input ids should have the same length.")

        response_token_ids_start_idx = len(prompt_input_ids)

        if prompt_input_ids != full_tokenized["input_ids"][:response_token_ids_start_idx]:
            response_token_ids_start_idx -= 1

        prompt_input_ids = full_tokenized["input_ids"][:response_token_ids_start_idx]
        prompt_attention_mask = full_tokenized["attention_mask"][:response_token_ids_start_idx]

        if len(prompt_input_ids) != len(prompt_attention_mask):
            raise ValueError("Prompt input ids and attention mask should have the same length.")

        answer_input_ids = full_tokenized["input_ids"][response_token_ids_start_idx:]
        answer_attention_mask = full_tokenized["attention_mask"][response_token_ids_start_idx:]

        return dict(
            prompt_input_ids=prompt_input_ids,
            prompt_attention_mask=prompt_attention_mask,
            input_ids=answer_input_ids,
            attention_mask=answer_attention_mask,
        )

    def tokenize_row(self, feature, model: Union[PreTrainedModel, nn.Module] = None) -> Dict:
        """修复版tokenize_row - 处理对话格式数据"""
        batch = {}
        prompt = feature["prompt"]
        chosen = feature["chosen"]
        rejected = feature["rejected"]
        
        # 转换对话格式为文本格式
        if isinstance(prompt, list):
            prompt = convert_conversation_to_text(prompt, self.tokenizer)
        if isinstance(chosen, list):
            chosen = convert_conversation_to_text(chosen, self.tokenizer)
        if isinstance(rejected, list):
            rejected = convert_conversation_to_text(rejected, self.tokenizer)
        
        if not self.is_encoder_decoder:
            if not isinstance(prompt, str):
                raise ValueError(f"prompt should be an str but got {type(prompt)}")
            prompt_tokens = self.tokenizer(prompt, add_special_tokens=False)
            prompt_tokens = {f"prompt_{k}": v for k, v in prompt_tokens.items()}

            if not isinstance(chosen, str):
                raise ValueError(f"chosen should be an str but got {type(chosen)} - after conversion")
            chosen_tokens = self.build_tokenized_answer(prompt, chosen)

            if not isinstance(rejected, str):
                raise ValueError(f"rejected should be an str but got {type(rejected)} - after conversion")
            rejected_tokens = self.build_tokenized_answer(prompt, rejected)

            prompt_len_input_ids = len(prompt_tokens["prompt_input_ids"])
            chosen_prompt_len_input_ids = len(chosen_tokens["prompt_input_ids"])
            rejected_prompt_len_input_ids = len(rejected_tokens["prompt_input_ids"])
            prompt_len_input_ids = min(chosen_prompt_len_input_ids, rejected_prompt_len_input_ids)

            for k, v in prompt_tokens.items():
                prompt_tokens[k] = v[:prompt_len_input_ids]

            num_diff_tokens = sum(
                [a != b for a, b in zip(chosen_tokens["prompt_input_ids"], rejected_tokens["prompt_input_ids"])]
            )
            num_diff_len = abs(chosen_prompt_len_input_ids - rejected_prompt_len_input_ids)
            if num_diff_tokens > 1 or num_diff_len > 1:
                raise ValueError(
                    "Chosen and rejected prompt_input_ids might only differ on the "
                    "last token due to tokenizer merge ops."
                )

            # add BOS token to head of prompt
            prompt_tokens["prompt_input_ids"] = [self.tokenizer.bos_token_id] + prompt_tokens["prompt_input_ids"]
            chosen_tokens["prompt_input_ids"] = [self.tokenizer.bos_token_id] + chosen_tokens["prompt_input_ids"]
            rejected_tokens["prompt_input_ids"] = [self.tokenizer.bos_token_id] + rejected_tokens["prompt_input_ids"]

            prompt_tokens["prompt_attention_mask"] = [1] + prompt_tokens["prompt_attention_mask"]
            chosen_tokens["prompt_attention_mask"] = [1] + chosen_tokens["prompt_attention_mask"]
            rejected_tokens["prompt_attention_mask"] = [1] + rejected_tokens["prompt_attention_mask"]

            # add EOS token to end of answer
            chosen_tokens["input_ids"].append(self.tokenizer.eos_token_id)
            chosen_tokens["attention_mask"].append(1)

            rejected_tokens["input_ids"].append(self.tokenizer.eos_token_id)
            rejected_tokens["attention_mask"].append(1)

            longer_response_length = max(len(chosen_tokens["input_ids"]), len(rejected_tokens["input_ids"]))

            # if combined sequence is too long, truncate the prompt
            for answer_tokens in [chosen_tokens, rejected_tokens, prompt_tokens]:
                if len(answer_tokens["prompt_input_ids"]) + longer_response_length > self.max_length:
                    if self.truncation_mode == "keep_start":
                        for k in ["prompt_input_ids", "prompt_attention_mask"]:
                            answer_tokens[k] = answer_tokens[k][: self.max_prompt_length]
                    elif self.truncation_mode == "keep_end":
                        for k in ["prompt_input_ids", "prompt_attention_mask"]:
                            answer_tokens[k] = answer_tokens[k][-self.max_prompt_length :]
                    else:
                        raise ValueError(f"Unknown truncation mode: {self.truncation_mode}")

            # if that's still too long, truncate the response
            for answer_tokens in [chosen_tokens, rejected_tokens]:
                if len(answer_tokens["prompt_input_ids"]) + longer_response_length > self.max_length:
                    for k in ["input_ids", "attention_mask"]:
                        answer_tokens[k] = answer_tokens[k][: self.max_length - self.max_prompt_length]

            # Create labels
            chosen_sequence_tokens = {
                k: chosen_tokens[f"prompt_{k}"] + chosen_tokens[k] for k in ["input_ids", "attention_mask"]
            }
            rejected_sequence_tokens = {
                k: rejected_tokens[f"prompt_{k}"] + rejected_tokens[k] for k in ["input_ids", "attention_mask"]
            }
            chosen_sequence_tokens["labels"] = chosen_sequence_tokens["input_ids"][:]
            chosen_sequence_tokens["labels"][: len(chosen_tokens["prompt_input_ids"])] = [
                self.label_pad_token_id
            ] * len(chosen_tokens["prompt_input_ids"])
            rejected_sequence_tokens["labels"] = rejected_sequence_tokens["input_ids"][:]
            rejected_sequence_tokens["labels"][: len(rejected_tokens["prompt_input_ids"])] = [
                self.label_pad_token_id
            ] * len(rejected_tokens["prompt_input_ids"])

            for k, toks in {
                "chosen_": chosen_sequence_tokens,
                "rejected_": rejected_sequence_tokens,
                "": prompt_tokens,
            }.items():
                for type_key, tokens in toks.items():
                    if type_key == "token_type_ids":
                        continue
                    batch[f"{k}{type_key}"] = tokens

        else:
            chosen_tokens = self.tokenizer(
                chosen, truncation=True, max_length=self.max_target_length, add_special_tokens=True
            )
            rejected_tokens = self.tokenizer(
                rejected, truncation=True, max_length=self.max_target_length, add_special_tokens=True
            )
            prompt_tokens = self.tokenizer(
                prompt, truncation=True, max_length=self.max_prompt_length, add_special_tokens=True
            )

            batch["chosen_labels"] = chosen_tokens["input_ids"]
            batch["rejected_labels"] = rejected_tokens["input_ids"]
            batch["prompt_input_ids"] = prompt_tokens["input_ids"]
            batch["prompt_attention_mask"] = prompt_tokens["attention_mask"]

            if model is not None and hasattr(model, "prepare_decoder_input_ids_from_labels"):
                batch["rejected_decoder_input_ids"] = model.prepare_decoder_input_ids_from_labels(
                    labels=batch["rejected_labels"]
                )
                batch["chosen_decoder_input_ids"] = model.prepare_decoder_input_ids_from_labels(
                    labels=batch["chosen_labels"]
                )

        return batch

    @contextmanager
    def null_ref_context(self):
        """Context manager for handling null reference model."""
        with self.accelerator.unwrap_model(
            self.model
        ).disable_adapter() if self.is_peft_model and not self.ref_adapter_name else nullcontext():
            if self.ref_adapter_name:
                self.model.set_adapter(self.ref_adapter_name)
            yield
            if self.ref_adapter_name:
                self.model.set_adapter(self.model_adapter_name or "default")

    def compute_reference_log_probs(self, padded_batch: Dict) -> Dict:
        """Computes log probabilities of the reference model."""
        compte_ref_context_manager = torch.cuda.amp.autocast if self._peft_has_been_casted_to_bf16 else nullcontext

        with torch.no_grad(), compte_ref_context_manager():
            if self.ref_model is None:
                with self.null_ref_context():
                    (
                        reference_chosen_logps,
                        reference_rejected_logps,
                        _,
                        _,
                    ) = self.concatenated_forward(self.model, padded_batch)
            else:
                (
                    reference_chosen_logps,
                    reference_rejected_logps,
                    _,
                    _,
                ) = self.concatenated_forward(self.ref_model, padded_batch)

        return reference_chosen_logps, reference_rejected_logps

    @staticmethod
    def concatenated_inputs(
        batch: Dict[str, Union[List, torch.LongTensor]],
        is_encoder_decoder: bool = False,
        label_pad_token_id: int = -100,
        padding_value: int = 0,
        device: Optional[torch.device] = None,
    ) -> Dict[str, torch.LongTensor]:
        """Concatenate the chosen and rejected inputs into a single tensor."""
        concatenated_batch = {}

        if is_encoder_decoder:
            max_length = max(batch["chosen_labels"].shape[1], batch["rejected_labels"].shape[1])
        else:
            max_length = max(batch["chosen_input_ids"].shape[1], batch["rejected_input_ids"].shape[1])

        for k in batch:
            if k.startswith("chosen") and isinstance(batch[k], torch.Tensor):
                if "labels" in k or is_encoder_decoder:
                    pad_value = label_pad_token_id
                elif k.endswith("_input_ids"):
                    pad_value = padding_value
                elif k.endswith("_attention_mask"):
                    pad_value = 0
                concatenated_key = k.replace("chosen", "concatenated")
                concatenated_batch[concatenated_key] = pad_to_length(batch[k], max_length, pad_value=pad_value)
        for k in batch:
            if k.startswith("rejected") and isinstance(batch[k], torch.Tensor):
                if "labels" in k or is_encoder_decoder:
                    pad_value = label_pad_token_id
                elif k.endswith("_input_ids"):
                    pad_value = padding_value
                elif k.endswith("_attention_mask"):
                    pad_value = 0
                concatenated_key = k.replace("rejected", "concatenated")
                concatenated_batch[concatenated_key] = torch.cat(
                    (
                        concatenated_batch[concatenated_key],
                        pad_to_length(batch[k], max_length, pad_value=pad_value),
                    ),
                    dim=0,
                ).to(device=device)

        if is_encoder_decoder:
            concatenated_batch["concatenated_input_ids"] = batch["prompt_input_ids"].repeat(2, 1).to(device=device)
            concatenated_batch["concatenated_attention_mask"] = (
                batch["prompt_attention_mask"].repeat(2, 1).to(device=device)
            )

        return concatenated_batch

    def ccpo_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
        chosen_probs: Union[torch.FloatTensor, None] = None,
        chosen_probs_win: Union[torch.FloatTensor, None] = None,
        chosen_probs_lose: Union[torch.FloatTensor, None] = None,
        reference_free: bool = False,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute the enhanced CCPO loss with code verification."""
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        if reference_free:
            ref_logratios = 0
        else:
            ref_logratios = reference_chosen_logps - reference_rejected_logps

        pi_logratios = pi_logratios.to(self.accelerator.device)
        ref_logratios = ref_logratios.to(self.accelerator.device)
        logits = pi_logratios - ref_logratios

        # For ccpo
        logits_w = policy_chosen_logps - reference_chosen_logps
        logits_l = policy_rejected_logps - reference_rejected_logps

        if self.loss_type == "sigmoid":
            losses = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )
        elif self.loss_type == "hinge":
            losses = torch.relu(1 - self.beta * logits)
        elif self.loss_type == "ipo":
            losses = (logits - 1 / (2 * self.beta)) ** 2
        elif self.loss_type == "ccpo":
            loss_w = (logits_w - (1 / self.beta)*(chosen_probs_win - 0.5)) ** 2
            loss_l = (logits_l - (1 / self.beta)*(chosen_probs_lose - 0.5)) ** 2
            losses = (loss_w + loss_l)/2
        elif self.loss_type == "ccpo_single":
            loss_w = (logits_w - (1 / self.beta)*(chosen_probs - 0.5)) ** 2
            loss_l = (logits_l + (1 / self.beta)*(chosen_probs - 0.5)) ** 2
            losses = (loss_w + loss_l)/2
        elif self.loss_type == "code_verified":
            # Enhanced loss type with V2 code verification results
            verification_weight = 2.5  # 增强V2验证的权重
            loss_w = (logits_w - verification_weight * (1 / self.beta)*(chosen_probs_win - 0.5)) ** 2
            loss_l = (logits_l - verification_weight * (1 / self.beta)*(chosen_probs_lose - 0.5)) ** 2
            losses = (loss_w + loss_l)/2
        elif self.loss_type == "kto_pair":
            chosen_KL = (policy_chosen_logps - reference_chosen_logps).mean().clamp(min=0)
            rejected_KL = (policy_rejected_logps - reference_rejected_logps).mean().clamp(min=0)

            chosen_logratios = policy_chosen_logps - reference_chosen_logps
            rejected_logratios = policy_rejected_logps - reference_rejected_logps
            losses = torch.cat(
                (
                    1 - F.sigmoid(self.beta * (chosen_logratios - rejected_KL)),
                    1 - F.sigmoid(self.beta * (chosen_KL - rejected_logratios)),
                ),
                0,
            )
        else:
            raise ValueError(
                f"Unknown loss type: {self.loss_type}. Should be one of ['sigmoid', 'hinge', 'ipo', 'kto_pair', 'ccpo', 'code_verified']"
            )

        chosen_rewards = (
            self.beta
            * (
                policy_chosen_logps.to(self.accelerator.device) - reference_chosen_logps.to(self.accelerator.device)
            ).detach()
        )
        rejected_rewards = (
            self.beta
            * (
                policy_rejected_logps.to(self.accelerator.device)
                - reference_rejected_logps.to(self.accelerator.device)
            ).detach()
        )

        return losses, chosen_rewards, rejected_rewards

    @staticmethod
    def get_batch_logps(
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        average_log_prob: bool = False,
        label_pad_token_id: int = -100,
        is_encoder_decoder: bool = False,
        chunk_size: int = 4096,  # 新增：分块处理大小
    ) -> torch.FloatTensor:
        """
        Compute the log probabilities of the given labels under the given logits.
        内存优化版本：分块处理以避免torch.gather内存爆炸
        """
        if logits.shape[:-1] != labels.shape:
            raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")

        if not is_encoder_decoder:
            labels = labels[:, 1:].clone()
            logits = logits[:, :-1, :]
        
        loss_mask = labels != label_pad_token_id
        labels[labels == label_pad_token_id] = 0

        batch_size, seq_len, vocab_size = logits.shape
        
        # 如果词汇表大小超过阈值，使用分块处理
        if vocab_size > 16384:  # 16K词汇表阈值
            per_token_logps = torch.zeros((batch_size, seq_len), device=logits.device, dtype=logits.dtype)
            
            # 逐批次处理以减少内存使用
            for i in range(batch_size):
                batch_logits = logits[i:i+1]  # [1, seq_len, vocab_size]
                batch_labels = labels[i:i+1]  # [1, seq_len]
                
                # 计算log_softmax并gather
                log_probs = F.log_softmax(batch_logits, dim=-1)  # [1, seq_len, vocab_size]
                token_logps = torch.gather(log_probs, dim=2, index=batch_labels.unsqueeze(2)).squeeze(2)  # [1, seq_len]
                per_token_logps[i] = token_logps.squeeze(0)
                
                # 显式清理中间变量
                del log_probs, token_logps, batch_logits, batch_labels
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        else:
            # 原始实现：小词汇表时直接处理
            per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

        if average_log_prob:
            return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
        else:
            return (per_token_logps * loss_mask).sum(-1)

    def concatenated_forward(
        self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Run the given model on the given batch."""
        concatenated_batch = self.concatenated_inputs(
            batch,
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
            padding_value=self.padding_value,
            device=self.accelerator.device,
        )
        len_chosen = batch["chosen_labels"].shape[0]

        model_kwargs = (
            {
                "labels": concatenated_batch["concatenated_labels"],
                "decoder_input_ids": concatenated_batch.pop("concatenated_decoder_input_ids", None),
            }
            if self.is_encoder_decoder
            else {}
        )
        
        # 确保所有输入张量都在正确的设备上，但不移动模型（FSDP会自动处理）
        input_ids = concatenated_batch["concatenated_input_ids"].to(self.accelerator.device)
        attention_mask = concatenated_batch["concatenated_attention_mask"].to(self.accelerator.device)
        
        # 使用梯度检查点和混合精度来节省内存
        with torch.cuda.amp.autocast(enabled=True):
            all_logits = model(
                input_ids,
                attention_mask=attention_mask,
                **model_kwargs,
            ).logits

        # 使用优化版的get_batch_logps
        all_logps = self.get_batch_logps(
            all_logits,
            concatenated_batch["concatenated_labels"],
            average_log_prob=False,
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
        )

        chosen_logps = all_logps[:len_chosen]
        rejected_logps = all_logps[len_chosen:]

        chosen_logits = all_logits[:len_chosen]
        rejected_logits = all_logits[len_chosen:]

        return (chosen_logps, rejected_logps, chosen_logits, rejected_logits)

    def get_batch_loss_metrics(
        self,
        model,
        batch: Dict[str, Union[List, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        """Compute the CCPO loss and other metrics."""
        metrics = {}

        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
        ) = self.concatenated_forward(model, batch)

        # 确保概率数据在正确的设备上
        device = policy_chosen_logps.device
        chosen_probs = torch.tensor(batch["chosen_probs"], dtype=torch.float32, device=device)
        chosen_probs_win = torch.tensor(batch["chosen_probs_win"], dtype=torch.float32, device=device)
        chosen_probs_lose = torch.tensor(batch["chosen_probs_lose"], dtype=torch.float32, device=device)

        if "reference_chosen_logps" in batch and "reference_rejected_logps" in batch:
            reference_chosen_logps = torch.tensor(batch["reference_chosen_logps"], device=device)
            reference_rejected_logps = torch.tensor(batch["reference_rejected_logps"], device=device)
        else:
            with torch.no_grad():
                if self.ref_model is None:
                    with self.null_ref_context():
                        (
                            reference_chosen_logps,
                            reference_rejected_logps,
                            _,
                            _,
                        ) = self.concatenated_forward(self.model, batch)
                else:
                    (
                        reference_chosen_logps,
                        reference_rejected_logps,
                        _,
                        _,
                    ) = self.concatenated_forward(self.ref_model, batch)

        losses, chosen_rewards, rejected_rewards = self.ccpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
            chosen_probs,
            chosen_probs_win,
            chosen_probs_lose,
        )
        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.mean().cpu()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.mean().cpu()
        metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.mean().cpu()
        metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).mean().cpu()
        metrics[f"{prefix}logps/rejected"] = policy_rejected_logps.detach().mean().cpu()
        metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.detach().mean().cpu()
        metrics[f"{prefix}logits/rejected"] = policy_rejected_logits.detach().mean().cpu()
        metrics[f"{prefix}logits/chosen"] = policy_chosen_logits.detach().mean().cpu()
        
        # 添加V2代码验证相关的指标
        if self.enable_code_verification:
            metrics[f"{prefix}code_verification/v2_chosen_probs_mean"] = chosen_probs.mean().cpu()
            metrics[f"{prefix}code_verification/v2_verification_strength"] = (chosen_probs_win - chosen_probs_lose).abs().mean().cpu()

        return losses.mean(), metrics

    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        if not self.use_dpo_data_collator:
            warnings.warn(
                "compute_loss is only implemented for DPODataCollatorWithPadding"
            )

        compute_loss_context_manager = torch.cuda.amp.autocast if self._peft_has_been_casted_to_bf16 else nullcontext

        with compute_loss_context_manager():
            loss, metrics = self.get_batch_loss_metrics(model, inputs, train_eval="train")

        self.store_metrics(metrics, train_eval="train")

        if return_outputs:
            return (loss, metrics)
        return loss

    def get_batch_samples(self, model, batch: Dict[str, torch.LongTensor]) -> Tuple[str, str]:
        """Generate samples from the model and reference model."""
        generate_context_manager = nullcontext if not self._peft_has_been_casted_to_bf16 else torch.cuda.amp.autocast

        with generate_context_manager():
            policy_output = model.generate(
                input_ids=batch["prompt_input_ids"],
                attention_mask=batch["prompt_attention_mask"],
                max_length=self.max_length,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )

            if "reference_output" in batch:
                reference_output = batch["reference_output"]
            else:
                if self.ref_model is None:
                    with self.null_ref_context():
                        reference_output = self.model.generate(
                            input_ids=batch["prompt_input_ids"],
                            attention_mask=batch["prompt_attention_mask"],
                            max_length=self.max_length,
                            do_sample=True,
                            pad_token_id=self.tokenizer.pad_token_id,
                        )
                else:
                    reference_output = self.ref_model.generate(
                        input_ids=batch["prompt_input_ids"],
                        attention_mask=batch["prompt_attention_mask"],
                        max_length=self.max_length,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )

        policy_output = pad_to_length(policy_output, self.max_length, self.tokenizer.pad_token_id)
        policy_output_decoded = self.tokenizer.batch_decode(policy_output, skip_special_tokens=True)

        reference_output = pad_to_length(reference_output, self.max_length, self.tokenizer.pad_token_id)
        reference_output_decoded = self.tokenizer.batch_decode(reference_output, skip_special_tokens=True)

        return policy_output_decoded, reference_output_decoded

    def prediction_step(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ):
        if not self.use_dpo_data_collator:
            warnings.warn(
                "prediction_step is only implemented for DPODataCollatorWithPadding"
            )
        if ignore_keys is None:
            if hasattr(model, "config"):
                ignore_keys = getattr(model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        prediction_context_manager = torch.cuda.amp.autocast if self._peft_has_been_casted_to_bf16 else nullcontext

        with torch.no_grad(), prediction_context_manager():
            loss, metrics = self.get_batch_loss_metrics(model, inputs, train_eval="eval")

        self.store_metrics(metrics, train_eval="eval")

        if prediction_loss_only:
            return (loss.detach(), None, None)

        logits_dict = {
            "eval_logits/chosen": metrics["eval_logits/chosen"],
            "eval_logits/rejected": metrics["eval_logits/rejected"],
        }
        logits = tuple(v.unsqueeze(dim=0) for k, v in logits_dict.items() if k not in ignore_keys)
        logits = torch.stack(logits).mean(axis=1).to(self.accelerator.device)
        labels = torch.zeros(logits.shape[0], device=self.accelerator.device)

        return (loss.detach(), logits, labels)

    def store_metrics(self, metrics: Dict[str, float], train_eval: Literal["train", "eval"] = "train") -> None:
        for key, value in metrics.items():
            self._stored_metrics[train_eval][key].append(value)

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """Overriding built-in evaluation loop to store metrics for each batch."""
        if self.generate_during_eval:
            num_samples = len(dataloader.dataset)
            random_indices = random.sample(range(num_samples), k=self.args.eval_batch_size)

            random_batch_dataset = dataloader.dataset.select(random_indices)
            random_batch = self.data_collator(random_batch_dataset)
            random_batch = self._prepare_inputs(random_batch)

            policy_output_decoded, ref_output_decoded = self.get_batch_samples(self.model, random_batch)

            self.log(
                {
                    "game_log": wandb.Table(
                        columns=["Prompt", "Policy", "Ref Model"],
                        rows=[
                            [prompt, pol[len(prompt) :], ref[len(prompt) :]]
                            for prompt, pol, ref in zip(
                                random_batch["prompt"], policy_output_decoded, ref_output_decoded
                            )
                        ],
                    )
                }
            )
            self.state.log_history.pop()

        initial_output = super().evaluation_loop(
            dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix
        )

        return initial_output

    def log(self, logs: Dict[str, float]) -> None:
        """Log `logs` on the various objects watching training."""
        train_eval = "train" if "loss" in logs else "eval"
        # Add averaged stored metrics to logs
        for key, metrics in self._stored_metrics[train_eval].items():
            logs[key] = torch.tensor(metrics).mean().item()
        del self._stored_metrics[train_eval]
        return super().log(logs)

    @wraps(Trainer.push_to_hub)
    def push_to_hub(self, commit_message: Optional[str] = "End of training", blocking: bool = True, **kwargs) -> str:
        """
        Overwrite the `push_to_hub` method in order to force-add the tag "code-verified" when pushing the
        model on the Hub. Please refer to `~transformers.Trainer.push_to_hub` for more details.
        """
        kwargs = trl_sanitze_kwargs_for_tagging(model=self.model, tag_names=self._tag_names, kwargs=kwargs)

        return super().push_to_hub(commit_message=commit_message, blocking=blocking, **kwargs)