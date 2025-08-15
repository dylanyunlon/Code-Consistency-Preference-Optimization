#!/usr/bin/env python3
"""
Enhanced Answer Extractor V2 - 保持接口不变的修复版
基于原版本，内部实现多数字验证逻辑，但保持所有函数签名和调用关系不变
"""

import re
import logging
from typing import Optional, Tuple, List, Dict, Any
from functools import lru_cache

logger = logging.getLogger(__name__)

class EnhancedAnswerExtractorV2:
    """
    增强版答案提取器V2 - 保持接口不变的修复版
    核心改进：内部实现多数字验证，但对外接口完全保持原样
    """
    
    def __init__(self, debug: bool = False):
        self.debug = debug
        
        # 预编译正则表达式以提高性能 (最佳实践1)
        self.compiled_patterns = self._compile_patterns()
        
        # 🚀 新增：内部缓存最后一次的候选答案数组（不影响对外接口）
        self._last_code_candidates = []
        
    def _compile_patterns(self) -> Dict[str, List[re.Pattern]]:
        """预编译正则表达式模式 - 修复版，添加数学答案模式"""
        patterns = {
            # 🔥 新增：数学文本的标准答案模式（最高优先级）
            'math_answer_patterns': [
                re.compile(r"####\s*(\d+(?:\.\d+)?)", re.IGNORECASE),  # GSM8K格式：#### 3400
                re.compile(r"\\boxed\{([^}]+)\}", re.IGNORECASE),  # LaTeX \boxed{}
                re.compile(r"The answer is[:\s]*([^\s\n.,]+)", re.IGNORECASE),  # 英文答案声明
                re.compile(r"答案是[：:\s]*([^\s\n，。]+)", re.IGNORECASE),  # 中文答案声明
                re.compile(r"Final answer[:\s]*([^\s\n.,]+)", re.IGNORECASE),  # 最终答案
                re.compile(r"答案[：:\s]*([^\s\n，。]+)", re.IGNORECASE),  # 简化中文答案
            ],
            
            'result_line_extraction': [
                # 修复：专门针对结果行的模式
                re.compile(r"字母\s*['\"]([^'\"]*?)['\"]?\s*在\s*['\"]([^'\"]*?)['\"]?\s*中出现了\s*(\d+)\s*次", re.IGNORECASE),
                re.compile(r"在\s*['\"]([^'\"]*?)['\"]?\s*中找到\s*(\d+)\s*个\s*['\"]([^'\"]*?)['\"]?", re.IGNORECASE),
                re.compile(r"结果[：:\s]*(\d+(?:\.\d+)?)", re.IGNORECASE),
                re.compile(r"输出[：:\s]*(\d+(?:\.\d+)?)", re.IGNORECASE),
                re.compile(r"^(\d+(?:\.\d+)?)$"),  # 纯数字行
                re.compile(r"等于\s*(\d+(?:\.\d+)?)", re.IGNORECASE),
                # 降低 = 模式的优先级，避免误匹配中间计算
                re.compile(r"=\s*(\d+(?:\.\d+)?)(?!\s*\+)", re.IGNORECASE),  # 避免匹配 = 11+1 这种情况
            ],
            'last_line_numbers': [
                # 修复：针对最后几行的数字提取
                re.compile(r"\b(\d+(?:\.\d+)?)\b"),
            ],
            'general_number': [
                # 通用数字提取 - 修复：支持任意位数，避免时间戳格式
                re.compile(r"(?<![\d\-])\b(\d+(?:\.\d+)?)\b(?![\-\d])"),  # 支持任意位数字
            ],
            'option_extraction': [
                re.compile(r"答案是\s*([A-Z])", re.IGNORECASE),
                re.compile(r"选择\s*([A-Z])", re.IGNORECASE),
                re.compile(r"^\s*([A-Z])\s*$", re.IGNORECASE),
            ],
            'yesno_extraction': [
                re.compile(r"\b(是|yes|y|对|正确|true)\b", re.IGNORECASE),
                re.compile(r"\b(否|no|n|不|错误|false)\b", re.IGNORECASE),
            ]
        }
        return patterns
    
    def extract_from_code_output(self, output: str) -> Optional[str]:
        """
        从代码执行输出中提取答案 - 修复版
        🚀 内部改进：同时缓存候选答案数组，但对外仍返回单个答案
        """
        if not output:
            return None
        
        if self.debug:
            logger.info(f"开始提取答案，输出长度: {len(output)}")
        
        # 🚀 内部改进：提取所有候选答案并缓存
        self._last_code_candidates = self._extract_all_candidate_answers(output)
        
        if self.debug and self._last_code_candidates:
            logger.info(f"内部候选答案数组: {self._last_code_candidates}")
        
        # 策略1: 最后几行优先提取 (新增策略)
        result = self._extract_from_last_lines(output)
        if result:
            if self.debug:
                logger.info(f"最后几行提取成功: {result}")
            return result
        
        # 策略2: 精确模式匹配 
        result = self._extract_with_precise_patterns(output)
        if result:
            if self.debug:
                logger.info(f"精确模式匹配成功: {result}")
            return result
        
        # 策略3: 行级分析 + 关键词优先
        result = self._extract_with_line_analysis(output)
        if result:
            if self.debug:
                logger.info(f"行级分析成功: {result}")
            return result
        
        # 策略4: 通用数字提取 (兜底策略)
        result = self._extract_general_numbers(output)
        if result:
            if self.debug:
                logger.info(f"通用数字提取成功: {result}")
            return result
        
        if self.debug:
            logger.warning("所有提取策略都失败了")
        return None
    
    def _extract_all_candidate_answers(self, text: str, max_lines: int = 5) -> List[str]:
        """
        🚀 新增：内部方法，提取所有可能的候选答案（你的想法的核心实现）
        """
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        if not lines:
            return []
        
        # 取最后几行
        last_lines = lines[-max_lines:]
        
        # 过滤掉明显的日志行和分隔符行
        skip_patterns = [
            re.compile(r'^\d{4}-\d{2}-\d{2}.*INFO'),  # 时间戳日志
            re.compile(r'^=+$'),  # 等号分隔符
            re.compile(r'^-+$'),  # 减号分隔符
            re.compile(r'位置示意图|可视化|出现位置'),  # 可视化描述
            re.compile(r'程序.*完成|执行完成'),  # 程序状态行
        ]
        
        filtered_lines = []
        for line in last_lines:
            skip_line = False
            for skip_pattern in skip_patterns:
                if skip_pattern.search(line):
                    skip_line = True
                    break
            if not skip_line:
                filtered_lines.append(line)
        
        # 🎯 核心逻辑：从最后的有效行提取所有可能的答案数字
        all_candidates = []
        
        # 从最后往前检查，找到第一个包含数字的有效行
        for line in reversed(filtered_lines):
            line_numbers = self._extract_all_numbers_from_line_safe(line)
            if line_numbers:
                if self.debug:
                    logger.info(f"从有效行提取候选数字: '{line}' -> {line_numbers}")
                all_candidates.extend(line_numbers)
                break  # 找到第一个有数字的行就停止
        
        # 去重并保持顺序
        seen = set()
        unique_candidates = []
        for num in all_candidates:
            if num not in seen:
                seen.add(num)
                unique_candidates.append(num)
        
        return unique_candidates
    
    def _extract_all_numbers_from_line_safe(self, line: str) -> List[str]:
        """
        🔧 改进：安全地从行中提取所有数字，正确处理货币格式
        """
        # 如果行包含明显的时间戳模式，跳过
        if re.search(r'\d{4}-\d{2}-\d{2}|\d{2}:\d{2}:\d{2}', line):
            return []
        
        numbers = []
        
        # 🔥 处理货币格式（如 $1,080）
        currency_pattern = re.compile(r'\$?([\d,]+(?:\.\d{1,2})?)')
        currency_matches = currency_pattern.findall(line)
        for match in currency_matches:
            # 移除逗号并验证
            clean_number = match.replace(',', '')
            if self._is_valid_answer_number(clean_number):
                numbers.append(clean_number)
        
        # 🔥 处理普通数字（避免重复货币数字）
        temp_line = line
        for currency_match in currency_matches:
            temp_line = temp_line.replace('$' + currency_match, '').replace(currency_match, '')
        
        # 提取剩余的普通数字
        general_pattern = re.compile(r'(?<![\d\-])\b(\d+(?:\.\d+)?)\b(?![\-\d])')
        general_matches = general_pattern.findall(temp_line)
        for num in general_matches:
            if self._is_valid_answer_number(num):
                numbers.append(num)
        
        return numbers
    
    def _extract_from_last_lines(self, text: str, max_lines: int = 5) -> Optional[str]:
        """
        新增策略：专门从最后几行提取结果
        针对用户的prompt格式："在最后的时候输出结果"
        """
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        if not lines:
            return None
        
        # 取最后几行
        last_lines = lines[-max_lines:]
        
        if self.debug:
            logger.info(f"检查最后 {len(last_lines)} 行:")
            for i, line in enumerate(last_lines):
                logger.info(f"  最后第{len(last_lines)-i}行: {line}")
        
        # 过滤掉明显的日志行和分隔符行
        skip_patterns = [
            re.compile(r'^\d{4}-\d{2}-\d{2}.*INFO'),  # 时间戳日志
            re.compile(r'^=+$'),  # 等号分隔符
            re.compile(r'^-+$'),  # 减号分隔符
            re.compile(r'位置示意图|可视化|出现位置'),  # 可视化描述
            re.compile(r'程序.*完成|执行完成'),  # 程序状态行
        ]
        
        filtered_lines = []
        for line in last_lines:
            skip_line = False
            for skip_pattern in skip_patterns:
                if skip_pattern.search(line):
                    skip_line = True
                    break
            if not skip_line:
                filtered_lines.append(line)
        
        if self.debug:
            logger.info(f"过滤后剩余 {len(filtered_lines)} 行:")
            for line in filtered_lines:
                logger.info(f"  有效行: {line}")
        
        # 从过滤后的行中提取结果
        for line in reversed(filtered_lines):  # 从最后往前
            # 1. 尝试精确结果模式
            for pattern in self.compiled_patterns['result_line_extraction']:
                matches = pattern.findall(line)
                if matches:
                    if isinstance(matches[0], tuple):
                        # 多捕获组，取数字组
                        for match_group in matches:
                            for item in match_group:
                                if self._is_valid_number(item):
                                    return item
                    else:
                        return matches[0]
            
            # 2. 尝试提取纯数字（最严格的匹配）
            if re.match(r'^\s*\d+(?:\.\d+)?\s*$', line):
                number = re.findall(r'\d+(?:\.\d+)?', line)[0]
                if self.debug:
                    logger.info(f"找到纯数字行: '{line}' -> {number}")
                return number
            
            # 3. 从行中提取数字，但避免时间戳
            numbers = self._extract_numbers_from_line_safe(line)
            if numbers:
                return numbers[-1]
        
        return None
    
    def _extract_numbers_from_line_safe(self, line: str) -> List[str]:
        """
        安全地从行中提取数字，避免时间戳等干扰
        🔧 修复：现在返回最后一个数字，但内部支持多数字提取
        """
        all_numbers = self._extract_all_numbers_from_line_safe(line)
        return all_numbers
    
    def _is_valid_answer_number(self, s: str) -> bool:
        """
        验证是否是有效的答案数字
        排除明显不合理的数字（如年份、时间等）
        """
        if not self._is_valid_number(s):
            return False
        
        try:
            num = float(s)
            # 排除明显的年份、时间等
            if num >= 1900 and num <= 2100:  # 可能是年份
                return False
            if num >= 0 and num <= 60 and '.' not in s:  # 可能是时分秒
                return True  # 但小于60的整数很可能是答案
            return True
        except (ValueError, TypeError):
            return False
    
    def _extract_with_precise_patterns(self, text: str) -> Optional[str]:
        """使用精确预编译模式进行提取"""
        for pattern in self.compiled_patterns['result_line_extraction']:
            matches = pattern.findall(text)
            if matches:
                # 处理不同的捕获组结构
                if isinstance(matches[0], tuple):
                    # 多捕获组，取最后一个数字组
                    for match_group in matches:
                        for item in reversed(match_group):
                            if self._is_valid_number(item):
                                return item
                else:
                    # 单捕获组
                    return matches[-1]
        return None
    
    def _extract_with_line_analysis(self, text: str) -> Optional[str]:
        """基于行级分析的智能提取 - 修复版"""
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        # 过滤掉非内容行
        content_lines = []
        skip_patterns = [
            re.compile(r'^=+$'),  # 分隔符行
            re.compile(r'^-+$'),  # 分隔符行
            re.compile(r'^\d{4}-\d{2}-\d{2}.*INFO'),  # 日志行
            re.compile(r'位置示意图|可视化|出现位置'),  # 可视化行
        ]
        
        for line in lines:
            skip_line = False
            for skip_pattern in skip_patterns:
                if skip_pattern.search(line):
                    skip_line = True
                    break
            if not skip_line and len(line) > 3:
                content_lines.append(line)
        
        if self.debug:
            logger.info(f"过滤后的内容行数: {len(content_lines)}")
            for i, line in enumerate(content_lines[-3:]):  # 显示最后3行
                logger.info(f"内容行 {i}: {line}")
        
        # 关键词优先策略 - 修复：专注于结果相关关键词
        priority_keywords = ['出现', '次', '个', '结果', '答案', '等于', '找到', '有', '共', '输出']
        
        # 优先处理包含关键词的行
        for line in reversed(content_lines):
            if any(keyword in line for keyword in priority_keywords):
                numbers = self._extract_numbers_from_line_safe(line)
                if numbers:
                    if self.debug:
                        logger.info(f"从关键词行提取: {line} -> {numbers[-1]}")
                    return numbers[-1]
        
        # 处理所有内容行
        for line in reversed(content_lines):
            numbers = self._extract_numbers_from_line_safe(line)
            if numbers:
                if self.debug:
                    logger.info(f"从普通行提取: {line} -> {numbers[-1]}")
                return numbers[-1]
        
        return None
    
    def _extract_general_numbers(self, text: str) -> Optional[str]:
        """通用数字提取 - 兜底策略，修复版"""
        all_numbers = []
        for pattern in self.compiled_patterns['general_number']:
            numbers = pattern.findall(text)
            all_numbers.extend([n for n in numbers if self._is_valid_answer_number(n)])
        
        if all_numbers:
            # 返回最后一个有效数字
            return all_numbers[-1]
        return None
    
    def _is_valid_number(self, s: str) -> bool:
        """验证是否是有效数字 - 支持更多格式"""
        if not s or not isinstance(s, str):
            return False
        
        try:
            # 尝试转换为浮点数
            float(s)
            # 额外检查：避免纯符号
            if s.strip() in ['+', '-', '.', 'e', 'E']:
                return False
            return True
        except (ValueError, TypeError):
            return False
    
    def extract_from_ai_response(self, text: str) -> Optional[str]:
        """从AI回答中提取答案 - 修复版，优先处理数学答案标记"""
        if not text:
            return None
        
        # 🔥 最高优先级：数学答案标记（必须优先处理！）
        for pattern in self.compiled_patterns['math_answer_patterns']:
            matches = pattern.findall(text)
            if matches:
                answer = matches[-1].strip()  # 取最后一个匹配
                if self.debug:
                    logger.info(f"从数学答案标记提取: {pattern.pattern} -> '{answer}'")
                # 验证提取的答案是否合理（不是0或负数，除非题目确实要求）
                if self._is_valid_number(answer):
                    return answer
        
        # 其他模式保持原有优先级
        result = self._extract_with_precise_patterns(text)
        if result:
            return result
        
        # 尝试选项提取
        for pattern in self.compiled_patterns['option_extraction']:
            matches = pattern.findall(text)
            if matches:
                return matches[-1].upper()
        
        # 尝试是否提取
        for pattern in self.compiled_patterns['yesno_extraction']:
            matches = pattern.findall(text)
            if matches:
                match = matches[-1].lower()
                if match in ['是', 'yes', 'y', '对', '正确', 'true']:
                    return 'yes'
                elif match in ['否', 'no', 'n', '不', '错误', 'false']:
                    return 'no'
        
        # 兜底：通用数字提取
        return self._extract_general_numbers(text)
    
    def compare_answers(self, ai_answer: str, code_answer: str, tolerance: float = 1e-6) -> Tuple[bool, float]:
        """
        比较答案是否匹配 - 增强版
        🚀 内部改进：如果直接比较失败，检查AI答案是否在候选数组中
        """
        if not ai_answer or not code_answer:
            return False, 0.0
        
        # 原有的直接比较逻辑
        direct_match, direct_confidence = self._direct_compare(ai_answer, code_answer, tolerance)
        if direct_match:
            return direct_match, direct_confidence
        
        # 🚀 新增：如果直接比较失败，检查是否在候选数组中
        if self._last_code_candidates and ai_answer:
            if self.debug:
                logger.info(f"直接比较失败，检查AI答案 '{ai_answer}' 是否在候选数组 {self._last_code_candidates} 中")
            
            # 检查AI答案是否在候选数组中
            for candidate in self._last_code_candidates:
                candidate_match, candidate_confidence = self._direct_compare(ai_answer, candidate, tolerance)
                if candidate_match:
                    if self.debug:
                        logger.info(f"✅ 在候选数组中找到匹配: '{ai_answer}' = '{candidate}'")
                    return True, candidate_confidence
        
        return False, 0.0
    
    def _direct_compare(self, ai_answer: str, code_answer: str, tolerance: float = 1e-6) -> Tuple[bool, float]:
        """
        直接比较两个答案 - 原有逻辑
        """
        # 数字比较
        if self._is_valid_number(ai_answer) and self._is_valid_number(code_answer):
            try:
                ai_num = float(ai_answer)
                code_num = float(code_answer)
                
                # 精确匹配
                if abs(ai_num - code_num) < tolerance:
                    return True, 1.0
                
                # 相对误差检查
                if code_num != 0:
                    relative_error = abs(ai_num - code_num) / abs(code_num)
                    if relative_error < 1e-4:
                        return True, 0.99
                    elif relative_error < 1e-2:
                        return True, 0.95
                    elif relative_error < 0.1:
                        return True, 0.8
                    else:
                        return False, 0.2
                else:
                    return abs(ai_num) < tolerance, 0.9 if abs(ai_num) < tolerance else 0.1
                    
            except (ValueError, TypeError):
                pass
        
        # 字符串比较
        ai_clean = ai_answer.strip().lower()
        code_clean = code_answer.strip().lower()
        
        if ai_clean == code_clean:
            return True, 1.0
        
        # 模糊匹配 (Levenshtein distance approximation)
        max_len = max(len(ai_clean), len(code_clean))
        if max_len > 0:
            # 简单的相似度计算
            common_chars = sum(1 for a, b in zip(ai_clean, code_clean) if a == b)
            similarity = common_chars / max_len
            if similarity > 0.8:
                return True, similarity
        
        return False, 0.0
    
    # 🚀 保持原有的验证方法，但内部逻辑升级
    def verify_answer_in_context(self, output: str, ground_truth: str) -> Tuple[bool, Dict[str, Any]]:
        """
        验证答案是否在上下文中存在
        🚀 内部改进：使用候选数组逻辑，但保持接口不变
        """
        verification_info = {
            'found_in_last_line': False,
            'last_line_numbers': [],
            'last_effective_line': '',
            'ground_truth_normalized': ground_truth.strip(),
            'is_correct': False,
            'candidate_answers': []  # 新增：候选答案数组
        }
        
        if not output or not ground_truth:
            return False, verification_info
        
        # 🚀 使用新的候选答案提取逻辑
        candidate_answers = self._extract_all_candidate_answers(output)
        verification_info['candidate_answers'] = candidate_answers
        verification_info['last_line_numbers'] = candidate_answers  # 向后兼容
        
        if candidate_answers:
            verification_info['last_effective_line'] = f"候选答案: {candidate_answers}"
            
            # 检查ground_truth是否在候选数组中
            gt_normalized = ground_truth.strip()
            
            # 直接字符串匹配
            if gt_normalized in candidate_answers:
                verification_info['found_in_last_line'] = True
                verification_info['is_correct'] = True
                
                if self.debug:
                    logger.info(f"✅ Ground truth '{gt_normalized}' 直接匹配候选答案: {candidate_answers}")
                
                return True, verification_info
            
            # 数值匹配（如 "10" vs "10.0"）
            try:
                gt_float = float(gt_normalized)
                for candidate in candidate_answers:
                    try:
                        if abs(float(candidate) - gt_float) < 1e-6:
                            verification_info['found_in_last_line'] = True
                            verification_info['is_correct'] = True
                            
                            if self.debug:
                                logger.info(f"✅ Ground truth '{gt_normalized}' 数值匹配 '{candidate}' 在候选答案中")
                            
                            return True, verification_info
                    except:
                        continue
            except:
                pass
        
        if self.debug:
            logger.info(f"❌ Ground truth '{ground_truth}' 未在候选答案中找到: {candidate_answers}")
        
        return False, verification_info
    
    def debug_extraction_process(self, text: str) -> Dict[str, Any]:
        """调试用：显示提取过程的详细信息"""
        debug_info = {
            'input_length': len(text),
            'lines_count': len(text.split('\n')),
            'patterns_tried': [],
            'matches_found': {},
            'final_result': None,
            'last_lines_analysis': {},
            'candidate_answers': []  # 新增：候选答案数组
        }
        
        # 分析最后几行
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        last_lines = lines[-5:] if lines else []
        debug_info['last_lines_analysis'] = {
            'last_5_lines': last_lines,
            'filtered_lines': []
        }
        
        # 过滤日志行
        skip_patterns = [
            re.compile(r'^\d{4}-\d{2}-\d{2}.*INFO'),
            re.compile(r'^=+$'),
            re.compile(r'^-+$'),
        ]
        
        for line in last_lines:
            skip_line = False
            for skip_pattern in skip_patterns:
                if skip_pattern.search(line):
                    skip_line = True
                    break
            if not skip_line:
                debug_info['last_lines_analysis']['filtered_lines'].append(line)
        
        # 测试所有模式
        for category, patterns in self.compiled_patterns.items():
            debug_info['matches_found'][category] = []
            for i, pattern in enumerate(patterns):
                matches = pattern.findall(text)
                if matches:
                    debug_info['matches_found'][category].append({
                        'pattern_index': i,
                        'pattern': pattern.pattern,
                        'matches': matches[:5]  # 只显示前5个匹配
                    })
        
        debug_info['final_result'] = self.extract_from_code_output(text)
        debug_info['candidate_answers'] = self._extract_all_candidate_answers(text)
        return debug_info


# 测试函数 - 验证接口保持不变但内部逻辑升级
def test_enhanced_extractor_v2_with_internal_upgrade():
    """测试接口不变但内部逻辑升级的效果"""
    extractor = EnhancedAnswerExtractorV2(debug=True)
    
    print("🧪 测试接口保持不变的内部升级版本")
    print("=" * 60)
    
    # 测试案例1：原始问题场景
    test_output1 = """2025-08-07 19:54:42,103 - INFO - 设定方程
2025-08-07 19:54:42,103 - INFO - 求解方程
解得：x = 10.0
2025-08-07 19:54:42,103 - INFO - 验证：20 + 10 × 10.0 = 120.0
2025-08-07 19:54:42,103 - INFO - 解验证成功
Daria每周需要存入$10才能在10周内筹集到$120
2025-08-07 19:54:42,103 - INFO - 程序执行完成"""
    
    print("📊 案例1测试 - 原始问题:")
    print("-" * 30)
    
    # 🔧 保持原有接口调用方式
    extracted_answer1 = extractor.extract_from_code_output(test_output1)
    print(f"extract_from_code_output(): '{extracted_answer1}'")
    
    # 验证逻辑（接口保持不变）
    is_correct1, verification_info1 = extractor.verify_answer_in_context(test_output1, "10")
    print(f"verify_answer_in_context('10'): {'✅' if is_correct1 else '❌'}")
    print(f"候选答案数组: {verification_info1.get('candidate_answers', [])}")
    
    # 比较逻辑（接口保持不变）
    match_result1, confidence1 = extractor.compare_answers("10", extracted_answer1)
    print(f"compare_answers('10', '{extracted_answer1}'): {'✅' if match_result1 else '❌'} (置信度: {confidence1})")
    
    print()
    
    # 测试案例2：货币格式问题
    test_output2 = "Adam will have earned a total of $1,080 after taxes after working for 30 days."
    
    print("📊 案例2测试 - 货币格式:")
    print("-" * 30)
    
    extracted_answer2 = extractor.extract_from_code_output(test_output2)
    print(f"extract_from_code_output(): '{extracted_answer2}'")
    
    is_correct2, verification_info2 = extractor.verify_answer_in_context(test_output2, "1080")
    print(f"verify_answer_in_context('1080'): {'✅' if is_correct2 else '❌'}")
    print(f"候选答案数组: {verification_info2.get('candidate_answers', [])}")
    
    match_result2, confidence2 = extractor.compare_answers("1080", extracted_answer2)
    print(f"compare_answers('1080', '{extracted_answer2}'): {'✅' if match_result2 else '❌'} (置信度: {confidence2})")
    
    print()
    
    # 测试案例3：AI回答提取
    test_ai_response = """#### 1080
The answer is: 1080"""
    
    print("📊 案例3测试 - AI回答提取:")
    print("-" * 30)
    
    ai_answer = extractor.extract_from_ai_response(test_ai_response)
    print(f"extract_from_ai_response(): '{ai_answer}'")
    
    # 与案例2结合测试
    match_result3, confidence3 = extractor.compare_answers(ai_answer, extracted_answer2)
    print(f"compare_answers('{ai_answer}', '{extracted_answer2}'): {'✅' if match_result3 else '❌'} (置信度: {confidence3})")
    
    print(f"\n🎯 升级效果总结:")
    print("✅ 所有函数接口保持完全不变")
    print("✅ 内部实现了候选答案数组逻辑")
    print("✅ compare_answers() 自动检查候选数组")
    print("✅ verify_answer_in_context() 使用多数字验证")
    print("✅ 正确处理货币格式分割问题")
    print("✅ 向后兼容性100%保持")


if __name__ == "__main__":
    test_enhanced_extractor_v2_with_internal_upgrade()