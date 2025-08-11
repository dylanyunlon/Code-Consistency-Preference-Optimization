#!/usr/bin/env python3
"""
Code-Verified CCPO Trainer Patch
在原有CCPOTrainer基础上添加代码验证功能
"""

import asyncio
import logging
from typing import Dict, List, Optional, Union, Any
import torch
import torch.nn.functional as F
from datasets import Dataset

# 导入原有的CCPOTrainer
from trainer import CCPOTrainer
from execution_verifier import ExecutionVerifier, batch_verify_responses

logger = logging.getLogger(__name__)

class CodeVerifiedCCPOTrainer(CCPOTrainer):
    """代码验证增强的CCPO训练器"""
    
    def __init__(
        self,
        model,
        ref_model=None,
        beta: float = 0.1,
        args=None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer=None,
        # 代码验证相关参数
        enable_code_verification: bool = False,
        verification_base_url: str = "https://8.134.217.190:17432",
        verification_username: str = "newuser", 
        verification_password: str = "newPass123",
        verification_batch_size: int = 5,
        verification_cache: bool = True,
        **kwargs
    ):
        # 初始化原始CCPO训练器
        super().__init__(
            model=model,
            ref_model=ref_model,
            beta=beta,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            **kwargs
        )
        
        # 代码验证配置
        self.enable_code_verification = enable_code_verification
        self.verification_config = {
            'base_url': verification_base_url,
            'username': verification_username,
            'password': verification_password
        }
        self.verification_batch_size = verification_batch_size
        self.verification_cache = verification_cache
        
        # 验证缓存
        self.cached_verifications = {}
        
        logger.info(f"CodeVerifiedCCPOTrainer initialized. Code verification: {enable_code_verification}")
    
    def ccpo_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
        chosen_probs: Union[torch.FloatTensor, None] = None,
        chosen_probs_win: Union[torch.FloatTensor, None] = None,
        chosen_probs_lose: Union[torch.FloatTensor, None] = None,
        # 新增代码验证相关参数
        chosen_verification_scores: Union[torch.FloatTensor, None] = None,
        rejected_verification_scores: Union[torch.FloatTensor, None] = None,
        reference_free: bool = False,
    ):
        """
        增强的CCPO损失函数，集成代码验证分数
        """
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        if reference_free:
            ref_logratios = 0
        else:
            ref_logratios = reference_chosen_logps - reference_rejected_logps

        pi_logratios = pi_logratios.to(self.accelerator.device)
        ref_logratios = ref_logratios.to(self.accelerator.device)
        logits = pi_logratios - ref_logratios

        # For CCPO with code verification
        logits_w = policy_chosen_logps - reference_chosen_logps
        logits_l = policy_rejected_logps - reference_rejected_logps

        if self.loss_type == "code_verified" and self.enable_code_verification:
            # 使用代码验证分数增强的损失函数
            if chosen_verification_scores is not None and rejected_verification_scores is not None:
                # 基于验证分数调整偏好概率
                verification_diff = chosen_verification_scores - rejected_verification_scores
                adjusted_chosen_probs = 0.5 + verification_diff * 0.4
                adjusted_chosen_probs = torch.clamp(adjusted_chosen_probs, 0.1, 0.9)
                
                # 计算代码验证增强的损失
                loss_w = (logits_w - (1 / self.beta) * (adjusted_chosen_probs - 0.5)) ** 2
                loss_l = (logits_l - (1 / self.beta) * (1 - adjusted_chosen_probs - 0.5)) ** 2
                
                # 添加验证质量权重
                verification_quality = (chosen_verification_scores + rejected_verification_scores) / 2
                quality_weight = 0.5 + verification_quality * 0.5  # 0.5-1.0范围
                
                losses = (loss_w + loss_l) * quality_weight / 2
            else:
                # 回退到原始CCPO损失
                logger.warning("代码验证分数不可用，使用原始CCPO损失")
                loss_w = (logits_w - (1 / self.beta) * (chosen_probs - 0.5)) ** 2
                loss_l = (logits_l + (1 / self.beta) * (chosen_probs - 0.5)) ** 2
                losses = (loss_w + loss_l) / 2
        
        elif self.loss_type == "ccpo":
            # 原始CCPO损失
            loss_w = (logits_w - (1 / self.beta) * (chosen_probs_win - 0.5)) ** 2
            loss_l = (logits_l - (1 / self.beta) * (chosen_probs_lose - 0.5)) ** 2
            losses = (loss_w + loss_l) / 2
        
        elif self.loss_type == "ccpo_single":
            # 单一CCPO损失
            loss_w = (logits_w - (1 / self.beta) * (chosen_probs - 0.5)) ** 2
            loss_l = (logits_l + (1 / self.beta) * (chosen_probs - 0.5)) ** 2
            losses = (loss_w + loss_l) / 2
        
        else:
            # 其他损失类型（sigmoid, hinge, ipo等）
            return super().ccpo_loss(
                policy_chosen_logps, policy_rejected_logps,
                reference_chosen_logps, reference_rejected_logps,
                chosen_probs, chosen_probs_win, chosen_probs_lose,
                reference_free
            )

        chosen_rewards = (
            self.beta * (
                policy_chosen_logps.to(self.accelerator.device) - 
                reference_chosen_logps.to(self.accelerator.device)
            ).detach()
        )
        rejected_rewards = (
            self.beta * (
                policy_rejected_logps.to(self.accelerator.device) - 
                reference_rejected_logps.to(self.accelerator.device)
            ).detach()
        )

        return losses, chosen_rewards, rejected_rewards
    
    def get_batch_loss_metrics(
        self,
        model,
        batch: Dict[str, Union[List, torch.LongTensor]],
        train_eval: str = "train",
    ):
        """计算批次损失指标，集成代码验证"""
        metrics = {}

        # 获取策略模型的logits
        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
        ) = self.concatenated_forward(model, batch)

        # 提取验证分数（如果可用）
        chosen_verification_scores = None
        rejected_verification_scores = None
        
        if "chosen_verification_score" in batch:
            chosen_verification_scores = torch.tensor(
                batch["chosen_verification_score"], 
                dtype=torch.float32, 
                device=policy_chosen_logps.device
            )
        
        if "rejected_verification_score" in batch:
            rejected_verification_scores = torch.tensor(
                batch["rejected_verification_score"], 
                dtype=torch.float32, 
                device=policy_chosen_logps.device
            )

        # 获取偏好概率
        chosen_probs = torch.tensor(batch["chosen_probs"], dtype=float, device=policy_chosen_logps.device)
        chosen_probs_win = torch.tensor(batch.get("chosen_probs_win", batch["chosen_probs"]), 
                                       dtype=float, device=policy_chosen_logps.device)
        chosen_probs_lose = torch.tensor(batch.get("chosen_probs_lose", 1 - batch["chosen_probs"]), 
                                        dtype=float, device=policy_chosen_logps.device)

        # 获取参考模型的logits
        if "reference_chosen_logps" in batch and "reference_rejected_logps" in batch:
            reference_chosen_logps = batch["reference_chosen_logps"]
            reference_rejected_logps = batch["reference_rejected_logps"]
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

        # 计算损失
        losses, chosen_rewards, rejected_rewards = self.ccpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
            chosen_probs,
            chosen_probs_win,
            chosen_probs_lose,
            chosen_verification_scores,
            rejected_verification_scores,
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
        
        # 添加代码验证相关指标
        if chosen_verification_scores is not None:
            metrics[f"{prefix}verification/chosen_score"] = chosen_verification_scores.mean().cpu()
        if rejected_verification_scores is not None:
            metrics[f"{prefix}verification/rejected_score"] = rejected_verification_scores.mean().cpu()
        if chosen_verification_scores is not None and rejected_verification_scores is not None:
            metrics[f"{prefix}verification/score_margin"] = (
                chosen_verification_scores - rejected_verification_scores
            ).mean().cpu()

        return losses.mean(), metrics

    async def verify_batch_responses(
        self, 
        questions: List[str], 
        responses: List[str]
    ) -> List[Dict[str, Any]]:
        """批量验证响应的代码执行结果"""
        if not self.enable_code_verification:
            return [{"verified": True, "confidence": 1.0, "score": 1.0} for _ in responses]
        
        try:
            async with ExecutionVerifier(**self.verification_config) as verifier:
                verification_results = await batch_verify_responses(
                    questions, responses, verifier, 
                    max_concurrent=self.verification_batch_size
                )
                
                # 转换为字典格式
                results = []
                for result in verification_results:
                    results.append({
                        "verified": result.verified,
                        "confidence": result.confidence,
                        "score": self._calculate_verification_score(result),
                        "ai_answer": result.ai_answer,
                        "code_answer": result.code_answer,
                        "status": result.status.value
                    })
                
                return results
                
        except Exception as e:
            logger.error(f"批量验证失败: {e}")
            # 返回默认结果
            return [{"verified": False, "confidence": 0.5, "score": 0.5} for _ in responses]
    
    def _calculate_verification_score(self, result) -> float:
        """计算验证分数（与rank.py中的函数保持一致）"""
        base_score = 0.1
        
        if result.verified:
            confidence_score = result.confidence * 0.8
            verification_bonus = 0.2
            base_score = confidence_score + verification_bonus
        else:
            if result.status.value == "success":
                base_score = 0.3 + result.confidence * 0.2
            elif result.status.value == "execution_failed":
                base_score = 0.1
            elif result.status.value == "no_code_generated":
                base_score = 0.05
            else:
                base_score = 0.1
        
        if result.execution_time > 30:
            base_score *= 0.9
        elif result.execution_time > 60:
            base_score *= 0.8
        
        return min(1.0, max(0.0, base_score))

def create_code_verified_trainer(
    model,
    tokenizer,
    training_args,
    train_dataset,
    eval_dataset=None,
    verification_config: Optional[Dict[str, Any]] = None
) -> CodeVerifiedCCPOTrainer:
    """
    创建代码验证增强的CCPO训练器的便捷函数
    
    Args:
        model: 预训练模型
        tokenizer: 分词器
        training_args: 训练参数
        train_dataset: 训练数据集
        eval_dataset: 评估数据集
        verification_config: 验证配置
    
    Returns:
        CodeVerifiedCCPOTrainer实例
    """
    verification_config = verification_config or {}
    
    trainer = CodeVerifiedCCPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        beta=training_args.beta,
        max_length=training_args.max_length,
        max_prompt_length=training_args.max_prompt_length,
        loss_type=getattr(training_args, 'loss_type', 'code_verified'),
        enable_code_verification=verification_config.get('enable', True),
        verification_base_url=verification_config.get('base_url', 'https://8.134.217.190:17432'),
        verification_username=verification_config.get('username', 'newuser'),
        verification_password=verification_config.get('password', 'newPass123'),
        verification_batch_size=verification_config.get('batch_size', 5),
        precompute_ref_log_probs=True
    )
    
    return trainer