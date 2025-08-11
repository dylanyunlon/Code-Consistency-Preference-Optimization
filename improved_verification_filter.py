#!/usr/bin/env python3
"""
Improved Verification Filter - 过滤适合代码验证的问题
只处理自然科学和数学计算问题，避免代码分析类问题
"""

import re
import logging
from typing import List, Dict, Any, Tuple, Optional

logger = logging.getLogger(__name__)

class QuestionFilter:
    """问题过滤器 - 识别适合代码验证的问题"""
    
    def __init__(self):
        # 适合代码验证的问题类型关键词
        self.suitable_keywords = [
            # 数学计算
            '计算', '求解', '解方程', '数学', '算式', '运算', '结果是',
            'calculate', 'compute', 'solve', 'equation', 'math', 'result',
            
            # 统计分析
            '统计', '平均', '概率', '个数', '数量', '频率', '次数',
            'count', 'average', 'mean', 'probability', 'frequency', 'times',
            
            # 物理问题
            '物理', '速度', '距离', '时间', '加速度', '力', '质量', '能量',
            'physics', 'velocity', 'distance', 'acceleration', 'force', 'mass', 'energy',
            
            # 几何问题
            '面积', '体积', '周长', '半径', '直径', '角度', '几何',
            'area', 'volume', 'perimeter', 'radius', 'diameter', 'angle', 'geometry',
            
            # 字符串分析（如strawberry问题）
            '字母', '字符', '出现', '包含', '查找', '统计字符',
            'letter', 'character', 'occurrence', 'find', 'search', 'contains',
        ]
        
        # 不适合代码验证的问题类型关键词
        self.unsuitable_keywords = [
            # 代码分析
            'C#', 'Java', 'Python', 'JavaScript', 'code', '代码', '设计模式', 'design pattern',
            'class', 'method', 'function', 'algorithm', '算法实现', '编程',
            
            # 文本解释
            '解释', '说明', '分析', '描述', '比较', '评价', '讨论',
            'explain', 'describe', 'analyze', 'compare', 'discuss', 'evaluate',
            
            # 概念性问题
            '什么是', '如何', '为什么', '定义', '概念', '原理', '理论',
            'what is', 'how to', 'why', 'definition', 'concept', 'theory', 'principle',
            
            # 主观判断
            '认为', '觉得', '意见', '建议', '推荐', '评论', '观点',
            'opinion', 'suggest', 'recommend', 'comment', 'viewpoint', 'think',
        ]
        
        # 长度阈值
        self.max_prompt_length = 500  # 最大prompt长度
        self.max_token_estimate = 1000  # 估计最大token数
    
    def is_suitable_for_verification(self, prompt: str) -> Tuple[bool, str, Dict[str, Any]]:
        """
        判断问题是否适合代码验证
        
        Returns:
            Tuple[bool, str, Dict]: (是否适合, 原因, 详细信息)
        """
        details = {
            'prompt_length': len(prompt),
            'estimated_tokens': len(prompt.split()) * 1.3,  # 粗略估计token数
            'suitable_matches': [],
            'unsuitable_matches': [],
            'content_type': 'unknown'
        }
        
        # 1. 检查长度
        if len(prompt) > self.max_prompt_length:
            return False, f"Prompt太长 ({len(prompt)} > {self.max_prompt_length})", details
        
        if details['estimated_tokens'] > self.max_token_estimate:
            return False, f"估计token数过多 ({details['estimated_tokens']:.0f} > {self.max_token_estimate})", details
        
        # 2. 检查不适合的关键词
        prompt_lower = prompt.lower()
        for keyword in self.unsuitable_keywords:
            if keyword.lower() in prompt_lower:
                details['unsuitable_matches'].append(keyword)
        
        if details['unsuitable_matches']:
            details['content_type'] = 'code_analysis_or_conceptual'
            return False, f"包含不适合的关键词: {details['unsuitable_matches'][:3]}", details
        
        # 3. 检查适合的关键词
        for keyword in self.suitable_keywords:
            if keyword.lower() in prompt_lower:
                details['suitable_matches'].append(keyword)
        
        # 4. 特殊模式检测
        numerical_patterns = [
            r'\d+\s*[+\-*/]\s*\d+',  # 数学运算
            r'\d+\s*个',             # 计数
            r'\d+\s*次',             # 频次
            r'\d+\s*米',             # 单位
            r'\d+\s*平方米',         # 面积单位
            r'有几个',               # 计数问题
            r'多少',                 # 数量问题
        ]
        
        pattern_matches = []
        for pattern in numerical_patterns:
            if re.search(pattern, prompt):
                pattern_matches.append(pattern)
        
        details['pattern_matches'] = pattern_matches
        
        # 5. 综合判断
        if details['suitable_matches'] or pattern_matches:
            details['content_type'] = 'mathematical_or_scientific'
            return True, f"适合验证: 匹配关键词 {details['suitable_matches'][:3]} 或模式 {len(pattern_matches)}", details
        
        # 6. 默认情况：如果没有明显的不适合标志，且长度合理，可以尝试
        if len(prompt) < 200 and not details['unsuitable_matches']:
            details['content_type'] = 'potentially_suitable'
            return True, "长度适中且无明显不适合标志，可以尝试", details
        
        details['content_type'] = 'unclear'
        return False, "无法确定适合性，偏向保守", details

def filter_dataset_for_verification(
    prompts: List[str], 
    candidates_list: List[Tuple[str, ...]], 
    max_samples: Optional[int] = None,
    debug: bool = False
) -> Tuple[List[str], List[Tuple[str, ...]], Dict[str, Any]]:
    """
    过滤数据集，只保留适合代码验证的问题
    
    Args:
        prompts: 原始问题列表
        candidates_list: 原始候选回答列表
        max_samples: 最大保留样本数
        debug: 是否输出调试信息
    
    Returns:
        Tuple: (过滤后的问题, 过滤后的候选回答, 过滤统计信息)
    """
    filter_obj = QuestionFilter()
    
    filtered_prompts = []
    filtered_candidates = []
    filter_stats = {
        'total_original': len(prompts),
        'suitable_count': 0,
        'unsuitable_count': 0,
        'filter_reasons': {},
        'content_types': {},
        'examples': {
            'suitable': [],
            'unsuitable': []
        }
    }
    
    for i, (prompt, candidates) in enumerate(zip(prompts, candidates_list)):
        is_suitable, reason, details = filter_obj.is_suitable_for_verification(prompt)
        
        # 统计
        content_type = details['content_type']
        if content_type not in filter_stats['content_types']:
            filter_stats['content_types'][content_type] = 0
        filter_stats['content_types'][content_type] += 1
        
        if reason not in filter_stats['filter_reasons']:
            filter_stats['filter_reasons'][reason] = 0
        filter_stats['filter_reasons'][reason] += 1
        
        if is_suitable:
            filtered_prompts.append(prompt)
            filtered_candidates.append(candidates)
            filter_stats['suitable_count'] += 1
            
            # 保存适合的示例
            if len(filter_stats['examples']['suitable']) < 5:
                filter_stats['examples']['suitable'].append({
                    'index': i,
                    'prompt': prompt[:100] + "..." if len(prompt) > 100 else prompt,
                    'reason': reason,
                    'details': details
                })
            
            if debug:
                print(f"✅ 样本 {i}: 适合验证")
                print(f"   问题: {prompt[:50]}...")
                print(f"   原因: {reason}")
        else:
            filter_stats['unsuitable_count'] += 1
            
            # 保存不适合的示例
            if len(filter_stats['examples']['unsuitable']) < 5:
                filter_stats['examples']['unsuitable'].append({
                    'index': i,
                    'prompt': prompt[:100] + "..." if len(prompt) > 100 else prompt,
                    'reason': reason,
                    'details': details
                })
            
            if debug:
                print(f"❌ 样本 {i}: 不适合验证")
                print(f"   问题: {prompt[:50]}...")
                print(f"   原因: {reason}")
        
        # 达到最大样本数时停止
        if max_samples and filter_stats['suitable_count'] >= max_samples:
            break
    
    # 如果过滤后样本太少，警告
    if filter_stats['suitable_count'] < 10:
        logger.warning(f"过滤后只有 {filter_stats['suitable_count']} 个适合的样本，可能需要调整过滤条件")
    
    return filtered_prompts, filtered_candidates, filter_stats

def print_filter_report(filter_stats: Dict[str, Any]):
    """打印过滤报告"""
    print("\n" + "="*60)
    print("📊 数据集过滤报告")
    print("="*60)
    print(f"原始样本数: {filter_stats['total_original']}")
    print(f"适合验证: {filter_stats['suitable_count']}")
    print(f"不适合验证: {filter_stats['unsuitable_count']}")
    print(f"过滤率: {filter_stats['suitable_count']/filter_stats['total_original']*100:.1f}%")
    
    print(f"\n📋 内容类型分布:")
    for content_type, count in filter_stats['content_types'].items():
        percentage = count / filter_stats['total_original'] * 100
        print(f"  {content_type}: {count} ({percentage:.1f}%)")
    
    print(f"\n🔍 过滤原因统计:")
    for reason, count in sorted(filter_stats['filter_reasons'].items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {reason}: {count}")
    
    print(f"\n✅ 适合验证的示例:")
    for example in filter_stats['examples']['suitable'][:3]:
        print(f"  - [{example['index']}] {example['prompt']}")
        print(f"    原因: {example['reason']}")
    
    print(f"\n❌ 不适合验证的示例:")
    for example in filter_stats['examples']['unsuitable'][:3]:
        print(f"  - [{example['index']}] {example['prompt']}")
        print(f"    原因: {example['reason']}")
    
    print("="*60)

# 测试函数
def test_question_filter():
    """测试问题过滤器"""
    print("🧪 测试问题过滤器")
    print("="*50)
    
    test_prompts = [
        # 适合的问题
        "strawberry中有几个字母r？",
        "计算 2 + 3 × 4 的结果",
        "一个正方形边长是5米，面积是多少平方米？",
        "从1到100的所有偶数的和是多少？",
        "一个物体从10米高度自由落体，需要多长时间？",
        
        # 不适合的问题
        "请解释什么是面向对象编程？",
        "分析这段C#代码使用了哪些设计模式",
        "如何优化这个算法的时间复杂度？",
        "请比较Python和Java的优缺点",
        "can you act as a C# expert, with lots of experience in C# Domain Driven Design...",  # 太长的代码分析问题
    ]
    
    filter_obj = QuestionFilter()
    
    for i, prompt in enumerate(test_prompts):
        is_suitable, reason, details = filter_obj.is_suitable_for_verification(prompt)
        status = "✅ 适合" if is_suitable else "❌ 不适合"
        
        print(f"\n{i+1}. {status}")
        print(f"   问题: {prompt[:60]}...")
        print(f"   原因: {reason}")
        print(f"   长度: {details['prompt_length']} 字符")
        print(f"   内容类型: {details['content_type']}")
        
        if details['suitable_matches']:
            print(f"   匹配关键词: {details['suitable_matches'][:3]}")
        if details['unsuitable_matches']:
            print(f"   不适合关键词: {details['unsuitable_matches'][:3]}")

if __name__ == "__main__":
    test_question_filter()