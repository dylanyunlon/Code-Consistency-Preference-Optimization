#!/usr/bin/env python3
"""
CCPO Improved Answer Comparison and Rate Limiting Handler
专为CCPO Architecture B优化的答案比较逻辑和请求限流处理
"""

import asyncio
import re
import time
import random
import math
from typing import Optional, Tuple, Any, Dict, List
import logging

logger = logging.getLogger(__name__)

class CCPOAnswerComparator:
    """
    CCPO专用的答案比较器
    专为Architecture B的推理过程vs代码执行结果比较优化
    """
    
    def __init__(self, debug: bool = False):
        self.debug = debug
        
        # CCPO特有的标准化映射表
        self.ccpo_standardization_map = {
            # 数学常见表达
            'true': ['true', 'yes', 'correct', '正确', '是', '对', 'positive', '1', 'True'],
            'false': ['false', 'no', 'incorrect', '错误', '否', '不', 'negative', '0', 'False'],
            
            # 数学零值表达
            'zero': ['zero', '零', 'none', 'nothing', '无', '没有'],
            
            # 数学单位处理
            'percent': ['%', 'percent', 'percentage', '百分比'],
            'degree': ['°', 'degree', 'degrees', '度'],
        }
        
        # 预编译正则表达式 - 专为数学答案优化
        self.number_pattern = re.compile(r'[+-]?\d*\.?\d+(?:[eE][+-]?\d+)?')
        self.fraction_pattern = re.compile(r'(\d+)/(\d+)')  # 分数匹配
        self.percentage_pattern = re.compile(r'(\d+(?:\.\d+)?)%')  # 百分比匹配
        self.math_expression_pattern = re.compile(r'=\s*([+-]?\d*\.?\d+)')  # 等式结果
        
        # 数学常见表达式清理
        self.math_cleanup_patterns = [
            (r'答案是[:：]?\s*', ''),
            (r'结果是[:：]?\s*', ''),
            (r'等于[:：]?\s*', ''),
            (r'为[:：]?\s*', ''),
            (r'共有?[:：]?\s*', ''),
            (r'一共有?[:：]?\s*', ''),
            (r'总共有?[:：]?\s*', ''),
            (r'出现了?\s*', ''),
            (r'个?字母', ''),
            (r'[次个条项件]', ''),
        ]
        
    def extract_mathematical_answer(self, text: str) -> Optional[str]:
        """
        从文本中提取数学答案 - CCPO专用
        优先级：等式结果 > 纯数字 > 分数 > 百分比
        """
        if not text:
            return None
        
        text = str(text).strip()
        
        # 1. 优先提取等式结果（如：= 5）
        math_results = self.math_expression_pattern.findall(text)
        if math_results:
            try:
                return str(float(math_results[-1]))
            except ValueError:
                pass
        
        # 2. 提取分数并转换为小数
        fractions = self.fraction_pattern.findall(text)
        if fractions:
            try:
                numerator, denominator = map(float, fractions[-1])
                if denominator != 0:
                    result = numerator / denominator
                    # 如果结果是整数，返回整数形式
                    if result == int(result):
                        return str(int(result))
                    else:
                        return str(round(result, 6))
            except (ValueError, ZeroDivisionError):
                pass
        
        # 3. 提取百分比
        percentages = self.percentage_pattern.findall(text)
        if percentages:
            try:
                return str(float(percentages[-1]))
            except ValueError:
                pass
        
        # 4. 提取普通数字
        numbers = self.number_pattern.findall(text)
        if numbers:
            try:
                num = float(numbers[-1])
                # 返回最简形式
                if num == int(num):
                    return str(int(num))
                else:
                    return str(num)
            except ValueError:
                pass
        
        return None
    
    def normalize_ccpo_answer(self, answer: str) -> str:
        """CCPO专用的答案标准化"""
        if not answer:
            return ""
        
        # 转换为字符串并清理
        normalized = str(answer).strip().lower()
        
        # 优先进行数学答案提取
        math_answer = self.extract_mathematical_answer(normalized)
        if math_answer is not None:
            return math_answer
        
        # 应用数学表达式清理
        for pattern, replacement in self.math_cleanup_patterns:
            normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)
        
        normalized = normalized.strip()
        
        # 再次尝试数学答案提取（清理后）
        math_answer = self.extract_mathematical_answer(normalized)
        if math_answer is not None:
            return math_answer
        
        # 移除标点符号
        clean_normalized = re.sub(r'[^\w\s]', '', normalized)
        
        # 检查CCPO标准化映射
        for standard_value, variants in self.ccpo_standardization_map.items():
            if clean_normalized in [v.lower() for v in variants]:
                return standard_value
        
        # 最后尝试提取任何数字
        final_numbers = re.findall(r'\d+(?:\.\d+)?', clean_normalized)
        if final_numbers:
            try:
                num = float(final_numbers[-1])
                return str(int(num)) if num == int(num) else str(num)
            except ValueError:
                pass
        
        return clean_normalized
    
    def compare_ccpo_answers(
        self, 
        ground_truth_answer: Optional[str], 
        execution_answer: Optional[str], 
        tolerance: float = 1e-6
    ) -> Tuple[bool, float, str]:
        """
        CCPO Architecture B专用的答案比较
        比较ground_truth答案 vs 代码执行答案
        
        Args:
            ground_truth_answer: 数据集中的标准答案
            execution_answer: 服务器按推理思路执行代码得到的答案
            tolerance: 数值比较容差
            
        Returns:
            Tuple[bool, float, str]: (是否匹配, 置信度, 比较方法)
        """
        if not ground_truth_answer or not execution_answer:
            return False, 0.0, "答案为空"
        
        # CCPO标准化答案
        norm_ground_truth = self.normalize_ccpo_answer(ground_truth_answer)
        norm_execution = self.normalize_ccpo_answer(execution_answer)
        
        if self.debug:
            print(f"🎯 CCPO Architecture B答案比较:")
            print(f"   Ground Truth: '{ground_truth_answer}' → '{norm_ground_truth}'")
            print(f"   代码执行结果: '{execution_answer}' → '{norm_execution}'")
        
        # 1. 直接字符串匹配（最高置信度）
        if norm_ground_truth == norm_execution:
            return True, 1.0, "CCPO Architecture B精确匹配"
        
        # 2. 数值比较（Architecture B核心验证）
        try:
            truth_num = float(norm_ground_truth)
            execution_num = float(norm_execution)
            
            # 精确匹配
            if abs(truth_num - execution_num) < tolerance:
                return True, 1.0, "CCPO数值精确匹配"
            
            # Architecture B验证：推理逻辑正确性的验证
            # 如果代码执行结果与ground truth接近，说明推理逻辑是好的
            if execution_num != 0:
                relative_error = abs(truth_num - execution_num) / abs(execution_num)
                if relative_error < 1e-4:  # 0.01%误差
                    return True, 0.99, "CCPO超高精度验证通过"
                elif relative_error < 1e-3:  # 0.1%误差
                    return True, 0.98, "CCPO高精度验证通过"
                elif relative_error < 1e-2:  # 1%误差
                    return True, 0.95, "CCPO中等精度验证通过"
                elif relative_error < 0.02:  # 2%误差（代码执行可能有精度问题）
                    return True, 0.9, "CCPO低精度验证通过"
                else:
                    return False, 0.3, f"CCPO验证失败：推理逻辑有问题 (相对误差: {relative_error:.4f})"
            else:
                # truth为零的情况
                if abs(execution_num) < tolerance:
                    return True, 0.98, "CCPO零值验证通过"
                else:
                    return False, 0.2, f"CCPO零值验证失败 (执行结果: {execution_num})"
            
            # truth为零但execution不为零的情况
            if truth_num == 0:
                if abs(execution_num) < tolerance:
                    return True, 0.98, "CCPO零值验证通过"
                else:
                    return False, 0.1, f"CCPO零值验证失败 (应为0，得到: {execution_num})"
                    
        except (ValueError, TypeError):
            pass
        
        # 3. Architecture B特有：布尔值严格验证
        # ground truth和执行结果都必须是明确的布尔值才能匹配
        bool_mappings = {
            'true': ['true', 'yes', 'correct', '1', '1.0'],
            'false': ['false', 'no', 'incorrect', '0', '0.0'],
        }
        
        truth_bool = None
        exec_bool = None
        
        for bool_val, variants in bool_mappings.items():
            if norm_ground_truth in variants:
                truth_bool = bool_val
            if norm_execution in variants:
                exec_bool = bool_val
        
        if truth_bool and exec_bool:
            if truth_bool == exec_bool:
                return True, 0.95, "CCPO布尔值验证通过"
            else:
                return False, 0.1, f"CCPO布尔值验证失败 (期望: {truth_bool}, 得到: {exec_bool})"
        
        # 4. 字符串精确匹配（降低容忍度）
        # Architecture B要求高精度，不能太宽松
        if len(norm_ground_truth) > 2 and len(norm_execution) > 2:
            if norm_ground_truth == norm_execution:
                return True, 0.8, "CCPO字符串精确匹配"
            elif norm_ground_truth in norm_execution or norm_execution in norm_ground_truth:
                return True, 0.6, "CCPO字符串包含匹配"
        
        # 5. 完全不匹配
        return False, 0.0, f"CCPO Architecture B验证失败 (Ground Truth: '{norm_ground_truth}', 执行: '{norm_execution}')"
    
    # 兼容性方法
    def compare_answers(self, ai_answer: str, code_answer: str, tolerance: float = 1e-6) -> Tuple[bool, float, str]:
        """兼容性方法，调用CCPO专用比较"""
        return self.compare_ccpo_answers(ai_answer, code_answer, tolerance)

class CCPORateLimitHandler:
    """
    CCPO专用的请求限流处理器
    为Architecture B的批量验证优化
    """
    
    def __init__(
        self,
        base_delay: float = 8.0,      # CCPO优化：从12.0降到8.0
        max_delay: float = 200.0,     # CCPO优化：从300.0降到200.0
        backoff_factor: float = 2.8,  # CCPO优化：从3.5降到2.8
        jitter: bool = True,
        max_consecutive_429s: int = 3  # CCPO专用：最大连续429次数
    ):
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.jitter = jitter
        self.max_consecutive_429s = max_consecutive_429s
        
        self.consecutive_429s = 0
        self.last_request_time = 0
        self.total_requests = 0
        self.successful_requests = 0
        
    async def wait_before_request(self):
        """CCPO优化的请求前等待"""
        current_time = time.time()
        self.total_requests += 1
        
        # 计算等待时间
        if self.consecutive_429s > 0:
            # 指数退避，但有上限
            delay = min(
                self.base_delay * (self.backoff_factor ** min(self.consecutive_429s, 4)),
                self.max_delay
            )
        else:
            # 基础延迟
            delay = self.base_delay
        
        # CCPO优化：根据成功率动态调整
        if self.total_requests > 10:
            success_rate = self.successful_requests / self.total_requests
            if success_rate > 0.9:
                delay *= 0.8  # 成功率高，减少延迟
            elif success_rate < 0.7:
                delay *= 1.3  # 成功率低，增加延迟
        
        # 添加随机抖动
        if self.jitter:
            jitter_amount = random.uniform(0, delay * 0.3)  # 30%抖动
            delay += jitter_amount
        
        # CCPO最小延迟保证
        delay = max(delay, 5.0)
        
        # 确保与上次请求的间隔
        time_since_last = current_time - self.last_request_time
        if time_since_last < delay:
            wait_time = delay - time_since_last
            if self.consecutive_429s > 0:
                logger.info(f"⏱️  CCPO限流等待 {wait_time:.1f}s (429: {self.consecutive_429s}, 成功率: {self.successful_requests}/{self.total_requests})")
            else:
                logger.debug(f"⏱️  CCPO基础延迟 {wait_time:.1f}s")
            await asyncio.sleep(wait_time)
        
        self.last_request_time = time.time()
    
    def handle_success(self):
        """处理成功请求"""
        self.successful_requests += 1
        if self.consecutive_429s > 0:
            logger.info(f"✅ CCPO请求成功，重置429计数器 (之前: {self.consecutive_429s})")
        self.consecutive_429s = 0
    
    def handle_429(self):
        """处理429错误 - CCPO优化"""
        self.consecutive_429s += 1
        logger.warning(f"⚠️  CCPO收到429限流，连续次数: {self.consecutive_429s}/{self.max_consecutive_429s}")
        
        # CCPO特有：如果连续429太多，返回更长的延迟
        if self.consecutive_429s >= self.max_consecutive_429s:
            extra_delay = min(180.0, 45.0 * (self.consecutive_429s - self.max_consecutive_429s + 1))
            logger.error(f"🔄 CCPO连续429过多，强制等待 {extra_delay:.1f}s")
            return extra_delay
        else:
            return min(20.0, 8.0 * self.consecutive_429s)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        success_rate = self.successful_requests / max(self.total_requests, 1)
        return {
            'total_requests': self.total_requests,
            'successful_requests': self.successful_requests,
            'success_rate': success_rate,
            'consecutive_429s': self.consecutive_429s,
            'current_delay': self.base_delay * (self.backoff_factor ** min(self.consecutive_429s, 4))
        }

async def ccpo_robust_api_call_with_retry(
    api_call_func,
    rate_limiter: CCPORateLimitHandler,
    max_retries: int = 2,   # CCPO优化：限制重试次数
    timeout: int = 150,     # CCPO优化：合理的超时时间
    *args,
    **kwargs
) -> Tuple[bool, Any, str]:
    """
    CCPO专用的健壮API调用
    为Architecture B的批量推理验证优化
    """
    
    for attempt in range(max_retries + 1):
        try:
            # 请求前等待
            await rate_limiter.wait_before_request()
            
            # 执行API调用
            if attempt > 0:
                logger.debug(f"🔄 CCPO API重试 {attempt}/{max_retries}")
            
            result = await asyncio.wait_for(
                api_call_func(*args, **kwargs),
                timeout=timeout
            )
            
            # 成功
            rate_limiter.handle_success()
            return True, result, ""
            
        except asyncio.TimeoutError:
            error_msg = f"CCPO API超时 ({timeout}s)"
            logger.error(f"⏰ {error_msg} - 尝试 {attempt + 1}")
            
            if attempt == max_retries:
                return False, None, error_msg
            
            # 超时后的渐进延迟
            timeout_delay = min(25.0, 8.0 * (attempt + 1))
            logger.info(f"⏱️  超时后等待 {timeout_delay:.1f}s")
            await asyncio.sleep(timeout_delay)
            
        except Exception as e:
            error_str = str(e).lower()
            
            # 检查429错误
            if ("429" in error_str or 
                "too many requests" in error_str or 
                "rate limit" in error_str or
                "请求过于频繁" in error_str):
                
                extra_delay = rate_limiter.handle_429()
                
                logger.error(f"❌ CCPO 429错误 - 尝试 {attempt + 1}: {e}")
                
                if attempt == max_retries:
                    return False, None, f"CCPO达到最大重试次数，最后错误: {e}"
                
                # 429额外等待
                if extra_delay > 0:
                    await asyncio.sleep(extra_delay)
                
                continue
            
            # 其他错误
            logger.error(f"❌ CCPO API异常 - 尝试 {attempt + 1}: {e}")
            
            if attempt == max_retries:
                return False, None, str(e)
            
            # 普通错误的重试延迟
            retry_delay = min(15.0, 4.0 * (attempt + 1))
            await asyncio.sleep(retry_delay)
    
    return False, None, "CCPO未知错误"

# 兼容性别名
ImprovedAnswerComparator = CCPOAnswerComparator
RateLimitHandler = CCPORateLimitHandler
robust_api_call_with_retry = ccpo_robust_api_call_with_retry

# 测试函数
async def test_ccpo_answer_comparison():
    """测试CCPO答案比较功能"""
    print("🧪 测试CCPO专用答案比较器")
    print("="*60)
    
    comparator = CCPOAnswerComparator(debug=True)
    
    # CCPO典型测试案例
    test_cases = [
        # 数学推理 vs 代码执行
        ("strawberry中有3个字母r", "3"),
        ("答案是5", "5.0"),
        ("结果等于42", "42"),
        ("共有7个", "7"),
        ("总共出现了2次", "2"),
        
        # 分数和小数
        ("1/2", "0.5"),
        ("3/4", "0.75"),
        ("25%", "25"),
        
        # 布尔和数字
        ("yes", "1"),
        ("no", "0"),
        ("true", "1.0"),
        ("false", "0.0"),
        
        # 容错测试
        ("大约是5", "5.1"),  # 应该匹配（容错）
        ("接近10", "9.8"),   # 应该匹配（容错）
        
        # 不匹配
        ("hello", "world"),
        ("是的", "10"),      # 应该不匹配
    ]
    
    for reasoning, execution in test_cases:
        is_match, confidence, method = comparator.compare_ccpo_answers(reasoning, execution)
        
        print(f"\n📊 CCPO比较结果:")
        print(f"   推理: '{reasoning}' vs 执行: '{execution}'")
        print(f"   匹配: {'✅' if is_match else '❌'}")
        print(f"   置信度: {confidence:.3f}")
        print(f"   方法: {method}")

async def test_ccpo_rate_limiter():
    """测试CCPO限流处理器"""
    print("\n🧪 测试CCPO限流处理器")
    print("="*60)
    
    rate_limiter = CCPORateLimitHandler()
    
    # 模拟CCPO API调用
    async def mock_ccpo_api_call(should_fail: bool = False):
        if should_fail:
            raise Exception("429 Too Many Requests")
        return {"success": True, "verification_result": "mock_result"}
    
    # 测试成功调用
    print("✅ 测试CCPO成功调用:")
    success, result, error = await ccpo_robust_api_call_with_retry(
        mock_ccpo_api_call, rate_limiter, max_retries=2, should_fail=False
    )
    print(f"结果: {success}, 数据: {result}")
    
    # 测试统计信息
    stats = rate_limiter.get_stats()
    print(f"📊 CCPO统计: {stats}")

if __name__ == "__main__":
    # 运行CCPO测试
    asyncio.run(test_ccpo_answer_comparison())
    asyncio.run(test_ccpo_rate_limiter())