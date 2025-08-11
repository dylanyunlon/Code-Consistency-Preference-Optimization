#!/usr/bin/env python3
"""
CCPO Architecture B Execution Verifier
实现核心创新：用服务器大模型按照7B推理思路生成并执行代码
"""

import asyncio
import re
import json
import hashlib
import time
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VerificationStatus(Enum):
    SUCCESS = "success"
    EXECUTION_FAILED = "execution_failed"
    PARSE_FAILED = "parse_failed"
    NO_CODE_GENERATED = "no_code_generated"
    TIMEOUT = "timeout"
    ERROR = "error"
    REASONING_FAILED = "reasoning_failed"  # 新增：推理过程转换失败

@dataclass
class VerificationResult:
    """验证结果数据类"""
    verified: bool
    status: VerificationStatus
    ai_answer: Optional[str]
    code_answer: Optional[str]
    confidence: float
    execution_time: float
    code_generated: str
    code_id: Optional[str]
    stdout: str
    stderr: str
    error_message: str
    verification_id: str
    raw_ai_response: str
    reasoning_process: Optional[str] = None  # 新增：保存原始推理过程

class ExecutionVerifier:
    """
    CCPO Architecture B 执行验证器
    核心创新：用服务器大模型按照7B推理思路生成并执行代码
    """
    
    def __init__(
        self,
        base_url: str = "https://8.134.217.190:17432",
        username: str = "newuser",
        password: str = "newPass123",
        timeout: int = 60,
        debug: bool = False
    ):
        self.base_url = base_url
        self.username = username
        self.password = password
        self.timeout = timeout
        self.debug = debug
        
        # 初始化改进的答案比较器
        try:
            from improved_answer_comparison import ImprovedAnswerComparator, RateLimitHandler, robust_api_call_with_retry
            self.answer_comparator = ImprovedAnswerComparator(debug=debug)
            self.rate_limiter = RateLimitHandler(
                base_delay=4.0,
                max_delay=120.0,
                backoff_factor=2.5
            )
            self.robust_api_call = robust_api_call_with_retry
            print("✅ 使用改进的答案比较器和限流处理")
        except ImportError:
            self.answer_comparator = None
            self.rate_limiter = None
            self.robust_api_call = None
            print("⚠️  使用基础答案比较器")
        
        # 初始化答案提取器
        try:
            from enhanced_answer_extractor_v2 import EnhancedAnswerExtractorV2
            self.answer_extractor = EnhancedAnswerExtractorV2(debug=debug)
            logger.info("✅ 使用增强版答案提取器V2")
        except ImportError:
            from enhanced_answer_extractor import EnhancedAnswerExtractor
            self.answer_extractor = EnhancedAnswerExtractor()
            logger.info("⚠️  使用原版增强答案提取器")
        
        self.client = None
        self.verification_cache = {}
        
    async def __aenter__(self):
        """异步上下文管理器入口"""
        try:
            import sys
            import os
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            
            from enhanced_client_example import EnhancedChatBotClient
            
            try:
                from enhanced_answer_extractor_v2 import EnhancedAnswerExtractorV2
                self.answer_extractor = EnhancedAnswerExtractorV2(debug=self.debug)
                logger.info("✅ 使用增强版答案提取器V2")
            except ImportError:
                from enhanced_answer_extractor import EnhancedAnswerExtractor
                self.answer_extractor = EnhancedAnswerExtractor()
                logger.info("⚠️  使用原版增强答案提取器")
            
            self.client = EnhancedChatBotClient(self.base_url)
            await self.client.__aenter__()
            await self.client.login(self.username, self.password)
            
            logger.info("✅ CCPO ExecutionVerifier初始化成功")
            return self
            
        except ImportError as e:
            logger.error(f"导入失败: {e}")
            raise ImportError(f"请确保相关文件在当前目录中: {e}")
        except Exception as e:
            logger.error(f"初始化失败: {e}")
            raise
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        if self.client:
            await self.client.__aexit__(exc_type, exc_val, exc_tb)
    
    def _generate_verification_id(self, question: str, reasoning: str) -> str:
        """生成验证ID用于缓存 - 基于reasoning而非最终回答"""
        content = f"{question}||{reasoning}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def build_reasoning_based_prompt(self, question: str, reasoning: str) -> str:
        """
        构建基于reasoning的代码生成prompt - CCPO核心创新
        让服务器按照7B的推理思路生成代码
        """
        prompt = f"""使用Python 3.10，基于以下解题思路创建代码：

解题思路：『{reasoning}』

目标问题：{question}

要求：
1. 严格按照思路中的每个推理步骤编写代码
2. 不要在代码中进行额外的理论分析
3. 思路中的答案不一定正确，请按思路逻辑执行
4. 代码必须体现思路中的每个推理步骤
5. 只print计算的中间结果和最终结果
6. 不要硬编码思路中的结论，要通过计算得出
7. 在最后的时候输出结果，确保结果是print的最后一行

请生成完整的可执行Python代码。"""
        
        return prompt
    
    def build_exact_prompt(self, question: str) -> str:
        """构建传统prompt格式（兜底方案）"""
        return f"""python版本为3.10.创建一个包含Python代码的脚本用于解决以下问题,在最后的时候输出结果,后续别再验证，保证输出结果是print的最后一行：{question}"""
    
    def extract_answers(self, ground_truth: str, code_output: str) -> Tuple[Optional[str], Optional[str]]:
        """
        提取Ground Truth和代码输出中的答案 - CCPO Architecture B修正
        注意：不提取推理过程中的答案，只提取ground_truth和代码执行结果
        """
        
        # 提取Ground Truth答案（直接使用，可能需要标准化）
        if hasattr(self.answer_extractor, 'extract_from_ai_response'):
            gt_answer = self.answer_extractor.extract_from_ai_response(ground_truth)
        else:
            # 简单清理ground truth
            gt_answer = str(ground_truth).strip()
        
        # 从代码输出中提取答案
        if hasattr(self.answer_extractor, 'extract_from_code_output'):
            code_answer = self.answer_extractor.extract_from_code_output(code_output)
        else:
            # 回退方法
            import re
            patterns = [
                r"出现了\s*(\d+)\s*次",
                r"有\s*(\d+)\s*个",
                r"结果[：:]?\s*(\d+(?:\.\d+)?)",
                r"=\s*(\d+(?:\.\d+)?)",
            ]
            for pattern in patterns:
                matches = re.findall(pattern, code_output, re.IGNORECASE)
                if matches:
                    code_answer = matches[-1]
                    break
            else:
                numbers = re.findall(r'\d+(?:\.\d+)?', code_output)
                code_answer = numbers[-1] if numbers else None
        
        return gt_answer, code_answer
    
    def compare_answers(self, ai_answer: str, code_answer: str) -> Tuple[bool, float, str]:
        """比较答案是否匹配"""
        if self.answer_comparator:
            return self.answer_comparator.compare_answers(ai_answer, code_answer)
        else:
            if not ai_answer or not code_answer:
                return False, 0.0, "答案为空"
            
            try:
                ai_num = float(ai_answer)
                code_num = float(code_answer)
                is_match = abs(ai_num - code_num) < 1e-6
                confidence = 1.0 if is_match else 0.0
                reason = "数字比较"
            except (ValueError, TypeError):
                is_match = ai_answer.strip() == code_answer.strip()
                confidence = 0.8 if is_match else 0.0
                reason = "字符串比较"
            
            return is_match, confidence, reason
    
    async def verify_reasoning_process(
        self,
        question: str,
        reasoning_process: str,
        ground_truth: Optional[str] = None,  # 新增ground_truth参数
        use_cache: bool = True,
        model: str = "claude-sonnet-4-20250514-all"
    ) -> VerificationResult:
        """
        CCPO核心方法：验证推理过程的质量
        让服务器按照7B的reasoning思路生成并执行代码
        """
        verification_id = self._generate_verification_id(question, reasoning_process)
        
        # 检查缓存
        if use_cache and verification_id in self.verification_cache:
            logger.info(f"使用缓存的推理验证结果: {verification_id}")
            return self.verification_cache[verification_id]
        
        start_time = time.time()
        
        if self.debug:
            print(f"\n🧠 CCPO推理验证")
            print(f"问题: {question}")
            print(f"7B推理过程: {reasoning_process[:200]}...")
            print("-" * 60)
        
        try:
            # 步骤1: 构建reasoning-based prompt - CCPO核心创新
            prompt = self.build_reasoning_based_prompt(question, reasoning_process)
            if self.debug:
                print(f"📝 CCPO Prompt:\n{prompt}")
            
            # 步骤2: 发送代码生成请求
            if self.debug:
                print("🔄 发送reasoning-based代码生成请求...")
            
            if self.robust_api_call and self.rate_limiter:
                success, response, error = await self.robust_api_call(
                    self.client.send_code_request,
                    self.rate_limiter,
                    max_retries=3,
                    timeout=90,
                    prompt=prompt,
                    auto_execute=False,
                    model=model
                )
                
                if not success:
                    return self._create_error_result(
                        verification_id, 
                        VerificationStatus.REASONING_FAILED,
                        f"推理转换请求失败: {error}",
                        time.time() - start_time,
                        reasoning_process
                    )
            else:
                await asyncio.sleep(3)
                response = await self.client.send_code_request(
                    prompt=prompt,
                    auto_execute=False,
                    model=model
                )
            
            if not response.get("success"):
                error_msg = f"推理转换失败: {response.get('error')}"
                if self.debug:
                    print(f"❌ {error_msg}")
                return self._create_error_result(
                    verification_id, 
                    VerificationStatus.REASONING_FAILED,
                    error_msg,
                    time.time() - start_time,
                    reasoning_process
                )
            
            # 步骤3: 解析响应
            data = response["data"]
            ai_full_response = data.get("content", "")
            extracted_codes = data.get("metadata", {}).get("extracted_codes", [])
            
            if self.debug:
                print(f"🤖 服务器基于推理生成的回答:\n{ai_full_response[:200]}...")
            
            if not extracted_codes:
                error_msg = "服务器未能根据推理过程生成代码"
                if self.debug:
                    print(f"❌ {error_msg}")
                return self._create_error_result(
                    verification_id,
                    VerificationStatus.NO_CODE_GENERATED,
                    error_msg,
                    time.time() - start_time,
                    reasoning_process
                )
            
            # 步骤4: 选择Python代码块
            python_codes = [
                code for code in extracted_codes 
                if code.get("language", "").lower() in ["python", "py"]
            ]

            if not python_codes:
                error_msg = f"服务器没有生成Python代码，只有: {[code.get('language') for code in extracted_codes]}"
                if self.debug:
                    print(f"❌ {error_msg}")
                return self._create_error_result(
                    verification_id,
                    VerificationStatus.NO_CODE_GENERATED,
                    error_msg,
                    time.time() - start_time,
                    reasoning_process
                )

            code_info = python_codes[0]
            code_id = code_info.get("id")
            code_content = code_info.get("content", "")

            if self.debug:
                print(f"✅ 服务器基于推理生成的代码块: {code_id}")
                print(f"📝 代码内容预览:\n{code_content[:300]}...")
            
            if not code_id:
                error_msg = "代码块ID获取失败"
                if self.debug:
                    print(f"❌ {error_msg}")
                return self._create_error_result(
                    verification_id,
                    VerificationStatus.ERROR,
                    error_msg,
                    time.time() - start_time,
                    reasoning_process
                )
            
            # 步骤5: 执行基于推理生成的代码
            if self.debug:
                print(f"⚡ 执行基于推理的代码: /exec {code_id}")
            
            if self.robust_api_call and self.rate_limiter:
                success, exec_result, error = await self.robust_api_call(
                    self.client.execute_code,
                    self.rate_limiter,
                    max_retries=3,
                    timeout=120,
                    code_id=code_id
                )
                
                if not success:
                    return self._create_error_result(
                        verification_id,
                        VerificationStatus.ERROR,
                        f"推理代码执行请求失败: {error}",
                        time.time() - start_time,
                        reasoning_process
                    )
            else:
                await asyncio.sleep(5)
                exec_result = await self.client.execute_code(code_id)
            
            execution_time = time.time() - start_time
            
            if not exec_result.get("success"):
                error_data = exec_result.get("data", {})
                error_msg = f"推理代码执行失败: {error_data}"
                if self.debug:
                    print(f"❌ {error_msg}")
                return self._create_error_result(
                    verification_id,
                    VerificationStatus.EXECUTION_FAILED,
                    error_msg,
                    execution_time,
                    reasoning_process
                )
            
            # 步骤6: 分析执行结果
            result_data = exec_result.get("data", {})
            result_info = result_data.get("result", {})
            stdout = result_info.get("stdout", "")
            stderr = result_info.get("stderr", "")
            
            if self.debug:
                print(f"✅ 推理代码执行成功!")
                print(f"📤 执行输出:")
                print(stdout[:500] + "..." if len(stdout) > 500 else stdout)
                
                if stderr:
                    print(f"⚠️  错误输出:")
                    print(stderr)
            
            # 步骤7: 提取答案 - 修复版本
            if ground_truth is not None:
                # 如果有ground_truth，从ground_truth中提取标准答案
                gt_extracted, _ = self.extract_answers(ground_truth, "")
                # 从代码执行结果中提取答案
                _, code_extracted = self.extract_answers("", stdout)
            else:
                # 兼容模式：从reasoning_process中提取答案（保持向后兼容）
                gt_extracted, code_extracted = self.extract_answers(reasoning_process, stdout)
            
            if self.debug:
                print(f"\n🔍 CCPO答案提取结果:")
                if ground_truth is not None:
                    print(f"Ground Truth提取答案: '{gt_extracted}'")
                else:
                    print(f"推理过程提取答案: '{gt_extracted}'")
                print(f"执行结果提取答案: '{code_extracted}'")
            
            # 步骤8: 验证推理质量 - CCPO Architecture B核心
            if gt_extracted and code_extracted:
                is_match, confidence, reason = self.compare_answers(gt_extracted, code_extracted)
                
                if self.debug:
                    if ground_truth is not None:
                        print(f"\n📊 CCPO Architecture B推理质量验证:")
                        print(f"比较方法: {reason}")
                        print(f"置信度: {confidence:.3f}")
                        
                        if is_match:
                            print(f"✅ 推理验证通过! 这是高质量的推理过程")
                            print(f"   Ground Truth: {gt_extracted}")
                            print(f"   执行结果: {code_extracted}")
                        else:
                            print(f"❌ 推理验证失败! 推理过程有问题")
                            print(f"   Ground Truth: {gt_extracted}")
                            print(f"   执行结果: {code_extracted}")
                    else:
                        print(f"\n📊 兼容模式验证:")
                        print(f"比较方法: {reason}")
                        print(f"置信度: {confidence:.3f}")
                
                verification_success = is_match
                status = VerificationStatus.SUCCESS if is_match else VerificationStatus.PARSE_FAILED
            else:
                if self.debug:
                    print(f"❌ CCPO答案提取失败!")
                    print(f"   标准答案提取结果: {gt_extracted}")
                    print(f"   代码执行提取结果: {code_extracted}")
                verification_success = False
                confidence = 0.0
                status = VerificationStatus.PARSE_FAILED
            
            if self.debug:
                print(f"⏱️  CCPO总验证时间: {execution_time:.2f}s")
            
            # 创建结果对象
            result = VerificationResult(
                verified=verification_success,
                status=status,
                ai_answer=gt_extracted,  # 修正：存储ground_truth答案
                code_answer=code_extracted,
                confidence=confidence,
                execution_time=execution_time,
                code_generated=code_content,
                code_id=code_id,
                stdout=stdout,
                stderr=stderr,
                error_message="",
                verification_id=verification_id,
                raw_ai_response=ai_full_response,
                reasoning_process=reasoning_process
            )
            
            # 缓存结果
            if use_cache:
                self.verification_cache[verification_id] = result
            
            return result
            
        except Exception as e:
            error_msg = f"CCPO推理验证异常: {e}"
            logger.error(error_msg)
            if self.debug:
                logger.exception("CCPO推理验证异常详情")
            
            return self._create_error_result(
                verification_id,
                VerificationStatus.ERROR,
                error_msg,
                time.time() - start_time,
                reasoning_process
            )
    
    async def verify_response(
        self,
        question: str,
        ai_response: str,
        use_cache: bool = True,
        model: str = "claude-sonnet-4-20250514-all"
    ) -> VerificationResult:
        """
        兼容性方法：同时支持传统验证和CCPO推理验证
        优先使用CCPO推理验证
        """
        # 优先使用CCPO推理验证
        return await self.verify_reasoning_process(
            question=question,
            reasoning_process=ai_response,
            use_cache=use_cache,
            model=model
        )
    
    def _create_error_result(
        self,
        verification_id: str,
        status: VerificationStatus,
        error_message: str,
        execution_time: float,
        reasoning_process: str
    ) -> VerificationResult:
        """创建错误结果"""
        return VerificationResult(
            verified=False,
            status=status,
            ai_answer=None,
            code_answer=None,
            confidence=0.0,
            execution_time=execution_time,
            code_generated="",
            code_id=None,
            stdout="",
            stderr="",
            error_message=error_message,
            verification_id=verification_id,
            raw_ai_response="",
            reasoning_process=reasoning_process
        )


# 新增：批量验证函数
async def batch_verify_responses(
    questions: List[str], 
    responses: List[str], 
    verifier: ExecutionVerifier, 
    max_concurrent: int = 5,
    ground_truths: Optional[List[str]] = None
) -> List[VerificationResult]:
    """
    批量验证响应的代码执行结果
    
    Args:
        questions: 问题列表
        responses: 响应列表（推理过程）
        verifier: 验证器实例
        max_concurrent: 最大并发数
        ground_truths: 可选的ground truth列表
    
    Returns:
        验证结果列表
    """
    if len(questions) != len(responses):
        raise ValueError("问题和响应数量不匹配")
    
    if ground_truths and len(ground_truths) != len(questions):
        raise ValueError("ground_truths和问题数量不匹配")
    
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def verify_single(idx: int) -> Tuple[int, VerificationResult]:
        async with semaphore:
            question = questions[idx]
            response = responses[idx]
            ground_truth = ground_truths[idx] if ground_truths else None
            
            try:
                result = await verifier.verify_reasoning_process(
                    question=question,
                    reasoning_process=response,
                    ground_truth=ground_truth,
                    use_cache=True
                )
                return idx, result
            except Exception as e:
                logger.error(f"批量验证第{idx}项失败: {e}")
                error_result = VerificationResult(
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
                    error_message=f"批量验证异常: {e}",
                    verification_id=f"batch_error_{idx}",
                    raw_ai_response="",
                    reasoning_process=response
                )
                return idx, error_result
    
    # 并发执行验证
    tasks = [verify_single(i) for i in range(len(questions))]
    results_with_idx = await asyncio.gather(*tasks)
    
    # 按原始顺序排序结果
    results_with_idx.sort(key=lambda x: x[0])
    results = [result for _, result in results_with_idx]
    
    logger.info(f"批量验证完成: {len(results)}个结果")
    return results


# 新增：便捷函数，直接在trainer中使用
async def verify_batch_with_context(
    questions: List[str],
    responses: List[str],
    base_url: str = "https://8.134.217.190:17432",
    username: str = "newuser",
    password: str = "newPass123",
    max_concurrent: int = 5,
    ground_truths: Optional[List[str]] = None,
    debug: bool = False
) -> List[VerificationResult]:
    """
    带上下文管理的批量验证便捷函数
    """
    async with ExecutionVerifier(
        base_url=base_url,
        username=username,
        password=password,
        debug=debug
    ) as verifier:
        return await batch_verify_responses(
            questions=questions,
            responses=responses,
            verifier=verifier,
            max_concurrent=max_concurrent,
            ground_truths=ground_truths
        )


# 测试函数
async def test_ccpo_verifier():
    """测试CCPO推理验证器"""
    print("🧪 测试CCPO Architecture B 推理验证器")
    print("=" * 70)
    
    # CCPO测试案例：包含完整的推理过程
    test_cases = [
        {
            "question": "strawberry 中有几个r",
            "reasoning": "要计算strawberry中字母r的个数，我需要逐个检查每个字母。strawberry的拼写是s-t-r-a-w-b-e-r-r-y。检查每个位置：第3位是r，第8位是r，第9位是r。所以总共有3个r。"
        },
        {
            "question": "计算 2 + 3 × 4 的结果",
            "reasoning": "根据运算顺序，先算乘法再算加法。首先计算3 × 4 = 12，然后计算2 + 12 = 14。所以最终结果是14。"
        },
        {
            "question": "一个正方形边长是5米，面积是多少平方米",
            "reasoning": "正方形的面积公式是边长的平方。边长是5米，所以面积 = 5 × 5 = 25平方米。"
        }
    ]
    
    try:
        async with ExecutionVerifier(debug=True) as verifier:
            print("✅ CCPO推理验证器初始化成功")
            
            success_count = 0
            total_count = len(test_cases)
            
            for i, case in enumerate(test_cases, 1):
                print(f"\n{'='*20} CCPO测试案例 {i}/{total_count} {'='*20}")
                
                result = await verifier.verify_reasoning_process(
                    case["question"],
                    case["reasoning"]
                )
                
                success = result.verified
                
                if success:
                    success_count += 1
                
                print(f"CCPO案例 {i} 结果: {'✅ 推理质量优秀' if success else '❌ 推理质量不佳'}")
                print(f"  验证ID: {result.verification_id}")
                print(f"  推理质量置信度: {result.confidence:.3f}")
                print(f"  验证时间: {result.execution_time:.2f}s")
                print(f"  推理答案: {result.ai_answer}")
                print(f"  执行答案: {result.code_answer}")
                print(f"  状态: {result.status.value}")
                
                if result.error_message:
                    print(f"  错误信息: {result.error_message}")
            
            # 最终统计
            print(f"\n{'='*20} CCPO验证统计 {'='*20}")
            print(f"总推理过程数: {total_count}")
            print(f"高质量推理数: {success_count}")
            print(f"推理质量率: {success_count/total_count*100:.1f}%")
            
            if success_count == total_count:
                print("🎉 所有推理过程都是高质量的! CCPO验证器工作正常!")
            else:
                print("⚠️  部分推理过程质量不佳，这正是CCPO要解决的问题")
                print("💡 CCPO将用这些验证结果来训练更好的推理能力")
    
    except Exception as e:
        print(f"❌ CCPO验证测试失败: {e}")
        logger.error("CCPO验证测试失败", exc_info=True)


if __name__ == "__main__":
    asyncio.run(test_ccpo_verifier())