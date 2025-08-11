#!/usr/bin/env python
"""
Clean CCPO Trainer - 基于嵌入式架构分离原则的重构版本
职责清晰：只负责CCPO算法，数据验证前置处理
"""

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

if is_peft_available():
    from peft import PeftModel

if is_wandb_available():
    import wandb

if is_deepspeed_available():
    import deepspeed


def convert_conversation_to_text(conversation: Union[str, List[Dict]], tokenizer) -> str:
    """
    对话格式转换工具函数 - 独立于Trainer
    """
    if isinstance(conversation, str):
        return conversation
    
    if isinstance(conversation, list):
        try:
            return tokenizer.apply_chat_template(
                conversation, 
                tokenize=False, 
                add_generation_prompt=False
            )
        except Exception:
            # 回退方案
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
    
    return str(conversation)


class OptimizedDataCollator(DPODataCollatorWithPadding):
    """
    简化的数据收集器 - 移除自动tensor转换，保持原版行为
    """
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # 只做标准的DPO数据整理，不预处理概率值
        return super().__call__(features)


class CCPOTrainer(Trainer):
    """
    清理版CCPO训练器 - 遵循单一职责原则
    只负责CCPO算法实现，不处理数据验证
    """

    _tag_names = ["trl", "ccpo"]

    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module, str] = None,
        ref_model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        beta: float = 0.1,
        label_smoothing: float = 0,
        loss_type: Literal["sigmoid", "hinge", "ipo", "kto_pair", "ccpo", "ccpo_single"] = "ccpo",
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
    ):
        # 验证必需的数据列
        if train_dataset is not None:
            self._validate_dataset_format(train_dataset)
        
        # 模型初始化逻辑
        if model_init_kwargs is None:
            model_init_kwargs = {}
        elif not isinstance(model, str):
            raise ValueError("You passed model_kwargs but model is already instantiated.")

        if ref_model_init_kwargs is None:
            ref_model_init_kwargs = {}
        elif not isinstance(ref_model, str):
            raise ValueError("You passed ref_model_kwargs but ref_model is already instantiated.")

        if isinstance(model, str):
            model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)

        if isinstance(ref_model, str):
            ref_model = AutoModelForCausalLM.from_pretrained(ref_model, **ref_model_init_kwargs)

        self._peft_has_been_casted_to_bf16 = False

        # 梯度检查点设置
        if getattr(args, "gradient_checkpointing", False):
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            else:
                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)
                model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        # 模型配置
        if model is not None:
            self.is_encoder_decoder = model.config.is_encoder_decoder
        elif is_encoder_decoder is None:
            raise ValueError("When no model is provided, you need to pass is_encoder_decoder parameter.")
        else:
            self.is_encoder_decoder = is_encoder_decoder

        self.is_peft_model = is_peft_available() and isinstance(model, PeftModel)
        self.model_adapter_name = model_adapter_name
        self.ref_adapter_name = ref_adapter_name

        # 参考模型设置
        if ref_model:
            self.ref_model = ref_model
        elif self.is_peft_model or precompute_ref_log_probs:
            self.ref_model = None
        else:
            self.ref_model = create_reference_model(model)

        if tokenizer is None:
            raise ValueError("tokenizer must be specified")
        
        # 默认参数设置
        self.max_length = max_length or 512
        self.max_prompt_length = max_prompt_length or 128
        self.max_target_length = max_target_length or 128 if self.is_encoder_decoder else None

        # 数据收集器设置 - 使用优化版本
        if data_collator is None:
            data_collator = OptimizedDataCollator(
                pad_token_id=tokenizer.pad_token_id,
                label_pad_token_id=label_pad_token_id,
                is_encoder_decoder=self.is_encoder_decoder,
            )
            if args.remove_unused_columns:
                args.remove_unused_columns = False
                warnings.warn("Set remove_unused_columns=False for DPODataCollatorWithPadding")
            self.use_dpo_data_collator = True
        else:
            self.use_dpo_data_collator = False

        if disable_dropout:
            disable_dropout_in_model(model)
            if self.ref_model is not None:
                disable_dropout_in_model(self.ref_model)

        # 训练参数
        self.generate_during_eval = generate_during_eval
        self.label_pad_token_id = label_pad_token_id
        self.padding_value = padding_value if padding_value is not None else tokenizer.pad_token_id
        self.truncation_mode = truncation_mode
        self.tokenizer = tokenizer
        self.precompute_ref_log_probs = precompute_ref_log_probs

        # 损失函数参数
        if loss_type in ["hinge", "ipo", "kto_pair"] and label_smoothing > 0:
            warnings.warn("Loss type doesn't support label smoothing. Ignoring parameter.")

        self.beta = beta
        self.label_smoothing = label_smoothing
        self.loss_type = loss_type

        self._stored_metrics = defaultdict(lambda: defaultdict(list))
        self._precomputed_train_ref_log_probs = False
        self._precomputed_eval_ref_log_probs = False

        # 数据预处理 - 只处理格式转换，不做验证
        if train_dataset is not None:
            train_dataset = train_dataset.map(self.tokenize_row)
        if eval_dataset is not None:
            eval_dataset = eval_dataset.map(self.tokenize_row)

        # 调用父类初始化
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

        # DeepSpeed兼容性检查
        if self.is_deepspeed_enabled:
            if self.accelerator.state.deepspeed_plugin.zero_stage == 3 and self.precompute_ref_log_probs:
                raise ValueError("Cannot use precompute_ref_log_probs=True with Deepspeed ZeRO-3")

        # 参考模型准备
        if self.ref_model is None:
            if not (self.is_peft_model or self.precompute_ref_log_probs):
                raise ValueError("No reference model and model is not Peft. Try precompute_ref_log_probs=True")
        else:
            if self.is_deepspeed_enabled:
                self.ref_model = self._prepare_deepspeed(self.ref_model)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

    def _validate_dataset_format(self, dataset: Dataset):
        """验证数据集格式 - 确保包含必需的概率列"""
        required_columns = ['prompt', 'chosen', 'rejected', 'chosen_probs']
        missing_columns = [col for col in required_columns if col not in dataset.features]
        
        if missing_columns:
            raise ValueError(
                f"Dataset missing required columns: {missing_columns}. "
                f"Please ensure data preprocessing (including code verification) is completed before training."
            )
        
        print(f"✅ Dataset validation passed. Found {len(dataset)} samples with required probability columns.")

    def tokenize_row(self, feature, model: Union[PreTrainedModel, nn.Module] = None) -> Dict:
        """
        优化的tokenize函数 - 处理对话格式转换
        """
        batch = {}
        
        # 对话格式转换
        prompt = feature["prompt"]
        chosen = feature["chosen"] 
        rejected = feature["rejected"]
        
        if isinstance(prompt, list):
            prompt = convert_conversation_to_text(prompt, self.tokenizer)
        if isinstance(chosen, list):
            chosen = convert_conversation_to_text(chosen, self.tokenizer)
        if isinstance(rejected, list):
            rejected = convert_conversation_to_text(rejected, self.tokenizer)

        if not self.is_encoder_decoder:
            # 标准的DPO tokenization逻辑
            prompt_tokens = self.tokenizer(prompt, add_special_tokens=False)
            prompt_tokens = {f"prompt_{k}": v for k, v in prompt_tokens.items()}

            chosen_tokens = self.build_tokenized_answer(prompt, chosen)
            rejected_tokens = self.build_tokenized_answer(prompt, rejected)

            # 处理prompt长度一致性
            prompt_len_input_ids = len(prompt_tokens["prompt_input_ids"])
            chosen_prompt_len_input_ids = len(chosen_tokens["prompt_input_ids"])
            rejected_prompt_len_input_ids = len(rejected_tokens["prompt_input_ids"])
            prompt_len_input_ids = min(chosen_prompt_len_input_ids, rejected_prompt_len_input_ids)

            for k, v in prompt_tokens.items():
                prompt_tokens[k] = v[:prompt_len_input_ids]

            # 验证token一致性
            num_diff_tokens = sum([a != b for a, b in zip(chosen_tokens["prompt_input_ids"], rejected_tokens["prompt_input_ids"])])
            num_diff_len = abs(chosen_prompt_len_input_ids - rejected_prompt_len_input_ids)
            if num_diff_tokens > 1 or num_diff_len > 1:
                raise ValueError("Chosen and rejected prompts differ by more than one token")

            # 添加特殊token
            for tokens in [prompt_tokens, chosen_tokens, rejected_tokens]:
                if "prompt_input_ids" in tokens:
                    tokens["prompt_input_ids"] = [self.tokenizer.bos_token_id] + tokens["prompt_input_ids"]
                    tokens["prompt_attention_mask"] = [1] + tokens["prompt_attention_mask"]

            # 添加EOS token
            for tokens in [chosen_tokens, rejected_tokens]:
                tokens["input_ids"].append(self.tokenizer.eos_token_id)
                tokens["attention_mask"].append(1)

            # 长度截断逻辑
            longer_response_length = max(len(chosen_tokens["input_ids"]), len(rejected_tokens["input_ids"]))
            
            for answer_tokens in [chosen_tokens, rejected_tokens, prompt_tokens]:
                if len(answer_tokens["prompt_input_ids"]) + longer_response_length > self.max_length:
                    if self.truncation_mode == "keep_start":
                        for k in ["prompt_input_ids", "prompt_attention_mask"]:
                            answer_tokens[k] = answer_tokens[k][:self.max_prompt_length]
                    elif self.truncation_mode == "keep_end":
                        for k in ["prompt_input_ids", "prompt_attention_mask"]:
                            answer_tokens[k] = answer_tokens[k][-self.max_prompt_length:]

            for answer_tokens in [chosen_tokens, rejected_tokens]:
                if len(answer_tokens["prompt_input_ids"]) + longer_response_length > self.max_length:
                    for k in ["input_ids", "attention_mask"]:
                        answer_tokens[k] = answer_tokens[k][:self.max_length - self.max_prompt_length]

            # 创建标签
            chosen_sequence_tokens = {
                k: chosen_tokens[f"prompt_{k}"] + chosen_tokens[k] for k in ["input_ids", "attention_mask"]
            }
            rejected_sequence_tokens = {
                k: rejected_tokens[f"prompt_{k}"] + rejected_tokens[k] for k in ["input_ids", "attention_mask"]
            }
            
            chosen_sequence_tokens["labels"] = chosen_sequence_tokens["input_ids"][:]
            chosen_sequence_tokens["labels"][:len(chosen_tokens["prompt_input_ids"])] = [self.label_pad_token_id] * len(chosen_tokens["prompt_input_ids"])
            
            rejected_sequence_tokens["labels"] = rejected_sequence_tokens["input_ids"][:]
            rejected_sequence_tokens["labels"][:len(rejected_tokens["prompt_input_ids"])] = [self.label_pad_token_id] * len(rejected_tokens["prompt_input_ids"])

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
            # Encoder-decoder处理
            chosen_tokens = self.tokenizer(chosen, truncation=True, max_length=self.max_target_length, add_special_tokens=True)
            rejected_tokens = self.tokenizer(rejected, truncation=True, max_length=self.max_target_length, add_special_tokens=True)
            prompt_tokens = self.tokenizer(prompt, truncation=True, max_length=self.max_prompt_length, add_special_tokens=True)

            batch["chosen_labels"] = chosen_tokens["input_ids"]
            batch["rejected_labels"] = rejected_tokens["input_ids"]
            batch["prompt_input_ids"] = prompt_tokens["input_ids"]
            batch["prompt_attention_mask"] = prompt_tokens["attention_mask"]

        return batch

    def build_tokenized_answer(self, prompt, answer):
        """构建tokenized答案 - 处理tokenizer特性"""
        full_tokenized = self.tokenizer(prompt + answer, add_special_tokens=False)
        prompt_input_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]

        answer_input_ids = full_tokenized["input_ids"][len(prompt_input_ids):]
        answer_attention_mask = full_tokenized["attention_mask"][len(prompt_input_ids):]

        full_concat_input_ids = np.concatenate([prompt_input_ids, answer_input_ids])
        full_input_ids = np.array(full_tokenized["input_ids"])

        if len(full_input_ids) != len(full_concat_input_ids):
            raise ValueError("Prompt and answer input ids length mismatch")

        response_token_ids_start_idx = len(prompt_input_ids)
        if prompt_input_ids != full_tokenized["input_ids"][:response_token_ids_start_idx]:
            response_token_ids_start_idx -= 1

        prompt_input_ids = full_tokenized["input_ids"][:response_token_ids_start_idx]
        prompt_attention_mask = full_tokenized["attention_mask"][:response_token_ids_start_idx]

        answer_input_ids = full_tokenized["input_ids"][response_token_ids_start_idx:]
        answer_attention_mask = full_tokenized["attention_mask"][response_token_ids_start_idx:]

        return dict(
            prompt_input_ids=prompt_input_ids,
            prompt_attention_mask=prompt_attention_mask,
            input_ids=answer_input_ids,
            attention_mask=answer_attention_mask,
        )

    def ccpo_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
        chosen_probs: torch.FloatTensor,
        chosen_probs_win: Optional[torch.FloatTensor] = None,
        chosen_probs_lose: Optional[torch.FloatTensor] = None,
        reference_free: bool = False,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """
        核心CCPO损失函数 - 简化版本，只保留必要的损失类型
        """
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        if reference_free:
            ref_logratios = 0
        else:
            ref_logratios = reference_chosen_logps - reference_rejected_logps

        pi_logratios = pi_logratios.to(self.accelerator.device)
        ref_logratios = ref_logratios.to(self.accelerator.device)
        logits = pi_logratios - ref_logratios

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
            # 使用代码验证增强的概率
            if chosen_probs_win is not None and chosen_probs_lose is not None:
                loss_w = (logits_w - (1 / self.beta) * (chosen_probs_win - 0.5)) ** 2
                loss_l = (logits_l - (1 / self.beta) * (chosen_probs_lose - 0.5)) ** 2
            else:
                # 回退到单一概率
                loss_w = (logits_w - (1 / self.beta) * (chosen_probs - 0.5)) ** 2
                loss_l = (logits_l + (1 / self.beta) * (chosen_probs - 0.5)) ** 2
            losses = (loss_w + loss_l) / 2
        elif self.loss_type == "ccpo_single":
            loss_w = (logits_w - (1 / self.beta) * (chosen_probs - 0.5)) ** 2
            loss_l = (logits_l + (1 / self.beta) * (chosen_probs - 0.5)) ** 2
            losses = (loss_w + loss_l) / 2
        elif self.loss_type == "kto_pair":
            chosen_KL = (policy_chosen_logps - reference_chosen_logps).mean().clamp(min=0)
            rejected_KL = (policy_rejected_logps - reference_rejected_logps).mean().clamp(min=0)
            chosen_logratios = policy_chosen_logps - reference_chosen_logps
            rejected_logratios = policy_rejected_logps - reference_rejected_logps
            losses = torch.cat((
                1 - F.sigmoid(self.beta * (chosen_logratios - rejected_KL)),
                1 - F.sigmoid(self.beta * (chosen_KL - rejected_logratios)),
            ), 0)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).detach()

        return losses, chosen_rewards, rejected_rewards

    def get_batch_loss_metrics(
        self,
        model,
        batch: Dict[str, Union[List, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        """计算批次损失和指标 - 优化的tensor处理"""
        metrics = {}

        (policy_chosen_logps, policy_rejected_logps, policy_chosen_logits, policy_rejected_logits,) = self.concatenated_forward(model, batch)

        # 确保完全按照原版逻辑处理概率值
        chosen_probs = torch.tensor(batch["chosen_probs"], dtype=float, device=self.accelerator.device)
        chosen_probs_win = torch.tensor(batch["chosen_probs_win"], dtype=float, device=self.accelerator.device) 
        chosen_probs_lose = torch.tensor(batch["chosen_probs_lose"], dtype=float, device=self.accelerator.device)

        # 获取参考模型logits
        if "reference_chosen_logps" in batch and "reference_rejected_logps" in batch:
            reference_chosen_logps = batch["reference_chosen_logps"]
            reference_rejected_logps = batch["reference_rejected_logps"]
        else:
            with torch.no_grad():
                if self.ref_model is None:
                    with self.null_ref_context():
                        (reference_chosen_logps, reference_rejected_logps, _, _,) = self.concatenated_forward(self.model, batch)
                else:
                    (reference_chosen_logps, reference_rejected_logps, _, _,) = self.concatenated_forward(self.ref_model, batch)

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

        # 代码验证相关指标
        metrics[f"{prefix}code_verification/chosen_probs_mean"] = chosen_probs.mean().cpu()
        metrics[f"{prefix}code_verification/verification_strength"] = (chosen_probs_win - chosen_probs_lose).abs().mean().cpu()

        return losses.mean(), metrics

    # === 以下是标准的训练器方法，保持不变 ===
    
    @contextmanager
    def null_ref_context(self):
        """参考模型上下文管理器"""
        with self.accelerator.unwrap_model(self.model).disable_adapter() if self.is_peft_model and not self.ref_adapter_name else nullcontext():
            if self.ref_adapter_name:
                self.model.set_adapter(self.ref_adapter_name)
            yield
            if self.ref_adapter_name:
                self.model.set_adapter(self.model_adapter_name or "default")

    @staticmethod
    def concatenated_inputs(batch, is_encoder_decoder=False, label_pad_token_id=-100, padding_value=0, device=None):
        """连接chosen和rejected输入"""
        concatenated_batch = {}
        if is_encoder_decoder:
            max_length = max(batch["chosen_labels"].shape[1], batch["rejected_labels"].shape[1])
        else:
            max_length = max(batch["chosen_input_ids"].shape[1], batch["rejected_input_ids"].shape[1])

        for k in batch:
            if k.startswith("chosen") and isinstance(batch[k], torch.Tensor):
                pad_value = label_pad_token_id if "labels" in k or is_encoder_decoder else (padding_value if k.endswith("_input_ids") else 0)
                concatenated_key = k.replace("chosen", "concatenated")
                concatenated_batch[concatenated_key] = pad_to_length(batch[k], max_length, pad_value=pad_value)
                
        for k in batch:
            if k.startswith("rejected") and isinstance(batch[k], torch.Tensor):
                pad_value = label_pad_token_id if "labels" in k or is_encoder_decoder else (padding_value if k.endswith("_input_ids") else 0)
                concatenated_key = k.replace("rejected", "concatenated")
                concatenated_batch[concatenated_key] = torch.cat((
                    concatenated_batch[concatenated_key],
                    pad_to_length(batch[k], max_length, pad_value=pad_value),
                ), dim=0).to(device=device)

        if is_encoder_decoder:
            concatenated_batch["concatenated_input_ids"] = batch["prompt_input_ids"].repeat(2, 1).to(device=device)
            concatenated_batch["concatenated_attention_mask"] = batch["prompt_attention_mask"].repeat(2, 1).to(device=device)

        return concatenated_batch

    def concatenated_forward(self, model, batch):
        """连接前向传播"""
        concatenated_batch = self.concatenated_inputs(
            batch, self.is_encoder_decoder, self.label_pad_token_id, self.padding_value, self.accelerator.device
        )
        len_chosen = batch["chosen_labels"].shape[0]

        model_kwargs = (
            {"labels": concatenated_batch["concatenated_labels"], "decoder_input_ids": concatenated_batch.pop("concatenated_decoder_input_ids", None)}
            if self.is_encoder_decoder else {}
        )
        all_logits = model(
            concatenated_batch["concatenated_input_ids"],
            attention_mask=concatenated_batch["concatenated_attention_mask"],
            **model_kwargs,
        ).logits

        all_logps = self.get_batch_logps(
            all_logits, concatenated_batch["concatenated_labels"], False, self.is_encoder_decoder, self.label_pad_token_id
        )

        return all_logps[:len_chosen], all_logps[len_chosen:], all_logits[:len_chosen], all_logits[len_chosen:]

    @staticmethod
    def get_batch_logps(logits, labels, average_log_prob=False, is_encoder_decoder=False, label_pad_token_id=-100):
        """计算批次log概率"""
        if logits.shape[:-1] != labels.shape:
            raise ValueError("Logits and labels shape mismatch")

        if not is_encoder_decoder:
            labels = labels[:, 1:].clone()
            logits = logits[:, :-1, :]
        
        loss_mask = labels != label_pad_token_id
        labels[labels == label_pad_token_id] = 0
        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

        if average_log_prob:
            return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
        else:
            return (per_token_logps * loss_mask).sum(-1)

    def compute_loss(self, model, inputs, return_outputs=False):
        """计算损失"""
        if not self.use_dpo_data_collator:
            warnings.warn("compute_loss only implemented for DPODataCollatorWithPadding")

        compute_loss_context_manager = torch.cuda.amp.autocast if self._peft_has_been_casted_to_bf16 else nullcontext

        with compute_loss_context_manager():
            loss, metrics = self.get_batch_loss_metrics(model, inputs, train_eval="train")

        self.store_metrics(metrics, train_eval="train")
        return (loss, metrics) if return_outputs else loss

    def store_metrics(self, metrics, train_eval="train"):
        """存储指标"""
        for key, value in metrics.items():
            self._stored_metrics[train_eval][key].append(value)

    def log(self, logs):
        """日志记录"""
        train_eval = "train" if "loss" in logs else "eval"
        for key, metrics in self._stored_metrics[train_eval].items():
            logs[key] = torch.tensor(metrics).mean().item()
        del self._stored_metrics[train_eval]
        return super().log(logs)

    def _prepare_deepspeed(self, model):
        """DeepSpeed准备"""
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        config_kwargs = deepcopy(deepspeed_plugin.deepspeed_config)

        if model and hasattr(model, "config"):
            hidden_size = max(model.config.hidden_sizes) if getattr(model.config, "hidden_sizes", None) else getattr(model.config, "hidden_size", None)
            if hidden_size and config_kwargs["zero_optimization"]["stage"] == 3:
                config_kwargs.update({
                    "zero_optimization.reduce_bucket_size": hidden_size * hidden_size,
                    "zero_optimization.stage3_param_persistence_threshold": 10 * hidden_size,
                    "zero_optimization.stage3_prefetch_bucket_size": 0.9 * hidden_size * hidden_size,
                })

        if config_kwargs["zero_optimization"]["stage"] != 3:
            config_kwargs["zero_optimization"]["stage"] = 0
        model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
        model.eval()
        return model

    # 省略其他标准方法实现...
    def get_train_dataloader(self): 
        if self.precompute_ref_log_probs and not self._precomputed_train_ref_log_probs:
            # 预计算参考log概率的逻辑
            pass
        return super().get_train_dataloader()

    def compute_reference_log_probs(self, padded_batch):
        """计算参考log概率"""
        compte_ref_context_manager = torch.cuda.amp.autocast if self._peft_has_been_casted_to_bf16 else nullcontext
        with torch.no_grad(), compte_ref_context_manager():
            if self.ref_model is None:
                with self.null_ref_context():
                    return self.concatenated_forward(self.model, padded_batch)[:2]
            else:
                return self.concatenated_forward(self.ref_model, padded_batch)[:2]

    @wraps(Trainer.push_to_hub)
    def push_to_hub(self, commit_message="End of training", blocking=True, **kwargs):
        """推送到Hub"""
        kwargs = trl_sanitze_kwargs_for_tagging(model=self.model, tag_names=self._tag_names, kwargs=kwargs)
        return super().push_to_hub(commit_message=commit_message, blocking=blocking, **kwargs)