#!/usr/bin/env python3
"""
Enhanced Answer Extractor V2 - 基于最佳实践的答案提取器
参考了Stack Overflow和Python官方文档的最佳实践
修复版：专注于最后输出结果的提取
"""

import re
import logging
from typing import Optional, Tuple, List, Dict, Any
from functools import lru_cache

logger = logging.getLogger(__name__)

class EnhancedAnswerExtractorV2:
    """
    增强版答案提取器V2 - 基于社区最佳实践
    修复版：专注于代码最后输出结果的提取
    """
    
    def __init__(self, debug: bool = False):
        self.debug = debug
        
        # 预编译正则表达式以提高性能 (最佳实践1)
        self.compiled_patterns = self._compile_patterns()
        
    def _compile_patterns(self) -> Dict[str, List[re.Pattern]]:
        """预编译正则表达式模式 - 性能最佳实践"""
        patterns = {
            'result_line_extraction': [
                # 修复：专门针对结果行的模式
                re.compile(r"字母\s*['\"]([^'\"]*?)['\"]?\s*在\s*['\"]([^'\"]*?)['\"]?\s*中出现了\s*(\d+)\s*次", re.IGNORECASE),
                re.compile(r"在\s*['\"]([^'\"]*?)['\"]?\s*中找到\s*(\d+)\s*个\s*['\"]([^'\"]*?)['\"]?", re.IGNORECASE),
                re.compile(r"结果[：:\s]*(\d+(?:\.\d+)?)", re.IGNORECASE),
                re.compile(r"答案[：:\s]*(\d+(?:\.\d+)?)", re.IGNORECASE),
                re.compile(r"输出[：:\s]*(\d+(?:\.\d+)?)", re.IGNORECASE),
                re.compile(r"^(\d+(?:\.\d+)?)$"),  # 纯数字行
                re.compile(r"等于\s*(\d+(?:\.\d+)?)", re.IGNORECASE),
                re.compile(r"=\s*(\d+(?:\.\d+)?)", re.IGNORECASE),
            ],
            'last_line_numbers': [
                # 修复：针对最后几行的数字提取
                re.compile(r"\b(\d+(?:\.\d+)?)\b"),
            ],
            'general_number': [
                # 通用数字提取 - 避免时间戳格式
                re.compile(r"(?<![\d\-])\b(\d{1,3}(?:\.\d+)?)\b(?![\-\d])"),  # 避免时间戳
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
        专注于最后输出结果，避免时间戳干扰
        """
        if not output:
            return None
        
        if self.debug:
            logger.info(f"开始提取答案，输出长度: {len(output)}")
        
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
        filtered_lines = []
        skip_patterns = [
            re.compile(r'^\d{4}-\d{2}-\d{2}.*INFO'),  # 时间戳日志
            re.compile(r'^=+$'),  # 等号分隔符
            re.compile(r'^-+$'),  # 减号分隔符
            re.compile(r'位置示意图|可视化|出现位置'),  # 可视化描述
            re.compile(r'程序.*完成|执行完成'),  # 程序状态行
        ]
        
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
        """
        # 如果行包含明显的时间戳模式，跳过
        if re.search(r'\d{4}-\d{2}-\d{2}|\d{2}:\d{2}:\d{2}', line):
            return []
        
        numbers = []
        for pattern in self.compiled_patterns['general_number']:
            found = pattern.findall(line)
            for num in found:
                if self._is_valid_answer_number(num):
                    numbers.append(num)
        return numbers
    
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
        """从AI回答中提取答案"""
        if not text:
            return None
        
        # 首先尝试数字提取
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
        支持数字比较、字符串比较、模糊匹配
        """
        if not ai_answer or not code_answer:
            return False, 0.0
        
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
    
    def debug_extraction_process(self, text: str) -> Dict[str, Any]:
        """调试用：显示提取过程的详细信息"""
        debug_info = {
            'input_length': len(text),
            'lines_count': len(text.split('\n')),
            'patterns_tried': [],
            'matches_found': {},
            'final_result': None,
            'last_lines_analysis': {}
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
        return debug_info


# 测试函数
def test_enhanced_extractor_v2():
    """测试增强版提取器V2 - 修复版"""
    extractor = EnhancedAnswerExtractorV2(debug=True)
    
    # 模拟实际输出格式
    test_output = """2025-07-31 17:59:06,345 - **main** - INFO - 程序开始执行
2025-07-31 17:59:06,345 - **main** - INFO - 开始分析单词: 'strawberry'
2025-07-31 17:59:06,345 - **main** - INFO - 查找字符: 'r'
2025-07-31 17:59:06,345 - **main** - INFO - 忽略大小写: False
2025-07-31 17:59:06,345 - **main** - INFO - 分析完成: 在'strawberry'中找到3个'r'
==================================================
分析结果
==================================================
单词: strawberry
单词长度: 10个字符
查找字符: 'r'
忽略大小写: 否
结果: 字母'r'在'strawberry'中出现了 3 次
位置: 第 3, 8, 9 个字符
可视化:
 s  t [r] a  w  b  e [r][r] y
==================================================
5.0
2025-07-31 17:59:06,345 - **main** - INFO - 程序执行完成"""
    
    print("🧪 测试增强版提取器V2 - 修复版")
    print("=" * 50)
    
    # 基本提取测试
    result = extractor.extract_from_code_output(test_output)
    print(f"提取结果: '{result}'")
    
    # 调试信息
    debug_info = extractor.debug_extraction_process(test_output)
    print("\n🔧 调试信息:")
    print(f"输入长度: {debug_info['input_length']}")
    print(f"处理行数: {debug_info['lines_count']}")
    
    # 最后几行分析
    print(f"\n📋 最后几行分析:")
    for i, line in enumerate(debug_info['last_lines_analysis']['last_5_lines']):
        print(f"  最后第{5-i}行: {line}")
    
    print(f"\n过滤后的有效行:")
    for line in debug_info['last_lines_analysis']['filtered_lines']:
        print(f"  有效: {line}")
    
    # 模式匹配详情
    for category, matches in debug_info['matches_found'].items():
        if matches:
            print(f"\n📋 {category}: {len(matches)} 个匹配")
            for match_info in matches:
                print(f"  - 模式: {match_info['pattern']}")
                print(f"    匹配: {match_info['matches']}")
    
    print(f"\n🎯 最终结果: '{debug_info['final_result']}'")
    
    # 答案比较测试
    ai_answer = "5"
    is_match, confidence = extractor.compare_answers(ai_answer, result)
    print(f"\n✅ 答案比较:")
    print(f"AI答案: '{ai_answer}', 代码答案: '{result}'")
    print(f"匹配结果: {'✅' if is_match else '❌'}, 置信度: {confidence:.3f}")


if __name__ == "__main__":
    test_enhanced_extractor_v2()