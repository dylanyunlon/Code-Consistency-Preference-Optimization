#!/usr/bin/env python3
"""
Enhanced Answer Extractor - 增强版答案提取器
基于你提供的check函数逻辑重写
"""

import re
from typing import Optional, Tuple
from functools import lru_cache

class EnhancedAnswerExtractor:
    """基于你的check函数逻辑的增强版答案提取器"""
    
    def __init__(self):
        pass
    
    def extract_label(self, text: str, answer_type: str = None) -> Optional[str]:
        """
        从文本中提取标签/答案
        基于你的extract_label函数逻辑
        """
        if not text:
            return None
        
        # 清理文本
        text = text.strip()
        
        # 根据答案类型使用不同的提取策略
        if answer_type == 'digit':
            return self._extract_digit_answer(text)
        elif answer_type == 'option':
            return self._extract_option_answer(text)
        elif answer_type == 'yesorno':
            return self._extract_yesno_answer(text)
        elif answer_type == 'formula':
            return self._extract_formula_answer(text)
        else:
            # 自动检测类型并提取
            return self._auto_extract_answer(text)
    
    def _extract_digit_answer(self, text: str) -> Optional[str]:
        """提取数字类型答案"""
        # 数字答案的模式，按优先级排序
        patterns = [
            # 明确的答案声明
            r'答案是[：:]?\s*([+-]?\d+(?:\.\d+)?)',
            r'结果是[：:]?\s*([+-]?\d+(?:\.\d+)?)',
            r'最终答案[：:]?\s*([+-]?\d+(?:\.\d+)?)',
            
            # 计数相关
            r'出现了\s*([+-]?\d+)\s*次',
            r'有\s*([+-]?\d+)\s*个',
            r'共\s*([+-]?\d+)\s*个',
            r'总共\s*([+-]?\d+)\s*个',
            r'([+-]?\d+)\s*个',
            r'([+-]?\d+)\s*次',
            
            # 数学运算
            r'等于\s*([+-]?\d+(?:\.\d+)?)',
            r'为\s*([+-]?\d+(?:\.\d+)?)',
            r'得到\s*([+-]?\d+(?:\.\d+)?)',
            r'计算得\s*([+-]?\d+(?:\.\d+)?)',
            r'=\s*([+-]?\d+(?:\.\d+)?)',
            
            # 带单位的数值
            r'([+-]?\d+(?:\.\d+)?)\s*平方米',
            r'([+-]?\d+(?:\.\d+)?)\s*立方米',
            r'([+-]?\d+(?:\.\d+)?)\s*米',
            r'([+-]?\d+(?:\.\d+)?)\s*厘米',
            r'([+-]?\d+(?:\.\d+)?)\s*公里',
            r'([+-]?\d+(?:\.\d+)?)\s*元',
            r'([+-]?\d+(?:\.\d+)?)\s*秒',
            r'([+-]?\d+(?:\.\d+)?)\s*分钟',
            r'([+-]?\d+(?:\.\d+)?)\s*小时',
            
            # 百分比
            r'([+-]?\d+(?:\.\d+)?)\s*%',
            
            # 纯数字（最低优先级）
            r'^([+-]?\d+(?:\.\d+)?)$',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                answer = matches[-1]  # 取最后一个匹配
                # 验证是否是有效数字
                if self._is_valid_number(answer):
                    return answer
        
        # 如果没有匹配到特定模式，提取最后出现的数字
        numbers = re.findall(r'[+-]?\d+(?:\.\d+)?', text)
        if numbers:
            return numbers[-1]
        
        return None
    
    def _extract_option_answer(self, text: str) -> Optional[str]:
        """提取选项类型答案 (A, B, C, D等)"""
        # 查找选项模式
        patterns = [
            r'答案是\s*([A-Z])',
            r'选择\s*([A-Z])',
            r'选项\s*([A-Z])',
            r'^\s*([A-Z])\s*$',
            r'\b([A-Z])\b'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                return matches[-1].upper()
        
        return None
    
    def _extract_yesno_answer(self, text: str) -> Optional[str]:
        """提取是否类型答案"""
        text_lower = text.lower()
        
        # 是的变体
        yes_patterns = [
            r'\b(是|yes|y|对|正确|true|是的|对的)\b',
        ]
        
        # 否的变体  
        no_patterns = [
            r'\b(否|no|n|不|错误|false|不是|不对)\b',
        ]
        
        for pattern in yes_patterns:
            if re.search(pattern, text_lower):
                return 'yes'
        
        for pattern in no_patterns:
            if re.search(pattern, text_lower):
                return 'no'
        
        return None
    
    def _extract_formula_answer(self, text: str) -> Optional[str]:
        """提取公式类型答案"""
        # 移除可能的数学符号
        cleaned = text.replace('$', '').strip()
        
        # 查找公式模式
        patterns = [
            r'答案是[：:]?\s*(.+?)(?:\s|$)',
            r'结果是[：:]?\s*(.+?)(?:\s|$)',
            r'等于\s*(.+?)(?:\s|$)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, cleaned)
            if matches:
                return matches[-1].strip()
        
        return cleaned
    
    def _auto_extract_answer(self, text: str) -> Optional[str]:
        """自动检测答案类型并提取"""
        # 首先尝试提取数字
        digit_answer = self._extract_digit_answer(text)
        if digit_answer:
            return digit_answer
        
        # 然后尝试选项
        option_answer = self._extract_option_answer(text)
        if option_answer:
            return option_answer
        
        # 然后尝试是否
        yesno_answer = self._extract_yesno_answer(text)
        if yesno_answer:
            return yesno_answer
        
        # 最后尝试公式
        formula_answer = self._extract_formula_answer(text)
        if formula_answer:
            return formula_answer
        
        return None
    
    def _is_valid_number(self, s: str) -> bool:
        """验证是否是有效数字"""
        try:
            float(s)
            return True
        except (ValueError, TypeError):
            return False
    
    def determine_answer_type(self, gt_text: str) -> str:
        """确定答案类型"""
        if not gt_text:
            return 'unknown'
        
        gt_label = self.extract_label(gt_text)
        if not gt_label:
            return 'unknown'
        
        if gt_label.isdigit() or self._is_valid_number(gt_label):
            return 'digit'
        elif len(gt_label) == 1 and gt_label.isupper() and gt_label.isalpha():
            return 'option'
        elif gt_label.lower() in ['yes', 'no', 'y', 'n', '是', '否']:
            return 'yesorno'
        else:
            return 'formula'
    
    @lru_cache(maxsize=1024)
    def check_answers(self, ground_truth: str, predicted_answer: str) -> Tuple[bool, str, str, str]:
        """
        检查答案是否匹配 - 基于你的check函数逻辑
        
        Returns:
            (is_match, gt_label, pred_label, answer_type)
        """
        # 提取真实答案标签
        gt_label = self.extract_label(ground_truth)
        if not gt_label:
            return False, None, None, 'unknown'
        
        # 确定答案类型
        answer_type = self.determine_answer_type(ground_truth)
        
        # 基于类型提取预测答案标签
        pred_label = self.extract_label(predicted_answer, answer_type)
        
        if not pred_label:
            return False, gt_label, None, answer_type
        
        # 后处理标签
        if answer_type == 'option':
            pred_label = pred_label.strip()[0].upper()
            gt_label = gt_label.upper()
        elif answer_type == 'yesorno':
            pred_label = pred_label.lower()
            gt_label = gt_label.lower()
        elif answer_type == 'formula':
            pred_label = pred_label.replace('$', '')
            gt_label = gt_label.replace('$', '')
        
        # 执行匹配检查
        is_match = self._perform_match_check(gt_label, pred_label, answer_type)
        
        return is_match, gt_label, pred_label, answer_type
    
    def _perform_match_check(self, gt_label: str, pred_label: str, answer_type: str) -> bool:
        """执行匹配检查"""
        if gt_label is None or pred_label is None:
            return False
        
        if answer_type == 'digit':
            return self._numeric_match(gt_label, pred_label)
        elif answer_type == 'option':
            return gt_label.upper() == pred_label.upper()
        elif answer_type == 'yesorno':
            return self._yesno_match(gt_label, pred_label)
        elif answer_type == 'formula':
            return self._formula_match(gt_label, pred_label)
        else:
            return gt_label == pred_label
    
    def _numeric_match(self, gt: str, pred: str) -> bool:
        """数字匹配检查"""
        try:
            gt_float = float(gt)
            pred_float = float(pred)
            
            # 对于大数使用相对误差，小数使用绝对误差
            if abs(gt_float) > 1.0:
                # 1%相对误差
                return abs(pred_float - gt_float) / max(1.0, abs(gt_float)) < 0.01
            else:
                # 0.01绝对误差
                return abs(pred_float - gt_float) < 0.01
        except ValueError:
            return gt == pred
    
    def _yesno_match(self, gt: str, pred: str) -> bool:
        """是否匹配检查"""
        gt_lower = gt.lower()
        pred_lower = pred.lower()
        
        if gt_lower in ['yes', 'y', 'true', '是', '对']:
            return pred_lower in ['yes', 'y', 'true', 'correct', 'yeah', 'yep', '是', '对', '正确']
        elif gt_lower in ['no', 'n', 'false', '否', '不']:
            return pred_lower in ['no', 'n', 'false', 'incorrect', 'nope', '否', '不', '错误']
        
        return gt_lower == pred_lower
    
    def _formula_match(self, gt: str, pred: str) -> bool:
        """公式匹配检查（简单版本）"""
        # 标准化空格和符号
        gt_norm = re.sub(r'\s+', ' ', gt.strip().lower())
        pred_norm = re.sub(r'\s+', ' ', pred.strip().lower())
        
        return gt_norm == pred_norm

# 测试函数
def test_enhanced_extractor():
    """测试增强版提取器"""
    extractor = EnhancedAnswerExtractor()
    
    # 测试案例
    test_cases = [
        # 数字类型
        ("strawberry中有3个字母r", "字母 'r' 在 'strawberry' 中出现了 3 次"),
        ("计算结果是14", "15 + 27 × 3 = 15 + 81 = 96"),
        ("面积是36平方米", "正方形面积 = 边长² = 6² = 36平方米"),
        
        # 选项类型
        ("答案是A", "选择A选项"),
        
        # 是否类型
        ("是的", "这是正确的"),
        ("不是", "这是错误的"),
    ]
    
    print("🧪 测试增强版答案提取器")
    print("=" * 50)
    
    for gt, pred in test_cases:
        is_match, gt_label, pred_label, answer_type = extractor.check_answers(gt, pred)
        
        print(f"\n真实答案: '{gt}'")
        print(f"预测答案: '{pred}'")
        print(f"提取结果: GT='{gt_label}', Pred='{pred_label}', Type='{answer_type}'")
        print(f"匹配结果: {'✅' if is_match else '❌'}")
        print("-" * 30)

if __name__ == "__main__":
    test_enhanced_extractor()