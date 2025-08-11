#!/usr/bin/env python3
"""
UCLA Dataset Specialized Filter - 专门针对UCLA-AGI数据集的过滤器
识别适合代码验证的数学/科学问题，过滤掉LaTeX格式化等不适合的任务
"""

import re
import logging
from typing import List, Dict, Any, Tuple, Optional
import json

logger = logging.getLogger(__name__)

class UCLADatasetFilter:
    """UCLA数据集专用过滤器"""
    
    def __init__(self):
        # 数学/科学计算关键词（适合验证）
        self.math_science_keywords = [
            # 数学计算
            'calculate', 'compute', 'solve', 'find', 'determine', 'evaluate',
            'sum', 'product', 'difference', 'quotient', 'equation', 'formula',
            'arithmetic', 'algebra', 'geometry', 'trigonometry', 'calculus',
            
            # 数量统计
            'count', 'number', 'how many', 'total', 'average', 'mean', 'median',
            'frequency', 'percentage', 'ratio', 'proportion', 'probability',
            
            # 物理/科学
            'physics', 'chemistry', 'velocity', 'acceleration', 'force', 'energy',
            'mass', 'weight', 'distance', 'time', 'speed', 'temperature',
            'pressure', 'volume', 'density', 'concentration',
            
            # 几何/测量
            'area', 'volume', 'perimeter', 'circumference', 'radius', 'diameter',
            'length', 'width', 'height', 'angle', 'degree', 'meter', 'cm', 'inch',
            
            # 字符串分析
            'letter', 'character', 'word', 'occurrence', 'appears', 'contains',
            'string', 'text analysis', 'count letters', 'count words',
        ]
        
        # 不适合验证的任务类型（主要是格式化、解释性任务）
        self.unsuitable_keywords = [
            # LaTeX/文档格式化
            'latex', 'documentclass', 'usepackage', 'begin{document}', 'end{document}',
            'table', 'tabular', 'caption', 'label', 'section', 'subsection',
            'bibliography', 'citation', 'format', 'structure', 'template',
            
            # 代码格式化/重构
            'code structure', 'code format', 'refactor', 'optimize code',
            'programming', 'syntax', 'debug', 'compile', 'function definition',
            'class definition', 'variable declaration', 'import statement',
            
            # 文本生成/创作
            'write', 'create', 'generate', 'compose', 'draft', 'essay', 'article',
            'report', 'summary', 'description', 'explanation', 'analysis',
            'review', 'critique', 'discuss', 'compare', 'contrast',
            
            # 设计模式/架构
            'design pattern', 'architecture', 'framework', 'methodology',
            'best practices', 'guidelines', 'principles', 'standards',
            
            # 主观判断/建议
            'opinion', 'recommend', 'suggest', 'advice', 'preference',
            'better', 'worse', 'advantages', 'disadvantages', 'pros', 'cons',
            
            # 解释/教学
            'explain', 'describe', 'what is', 'how to', 'why', 'definition',
            'concept', 'theory', 'principle', 'introduction', 'overview',
        ]
        
        # 数学模式识别
        self.math_patterns = [
            r'\d+\s*[+\-*/×÷]\s*\d+',  # 基本运算
            r'\d+\s*=\s*\d+',          # 等式
            r'\d+\s*%',                # 百分比
            r'\d+\s*(cm|meter|inch|kg|gram|second|minute|hour)',  # 带单位
            r'x\s*=\s*\d+',            # 变量赋值
            r'\d+\^\d+',               # 指数
            r'sqrt\(\d+\)',            # 平方根
            r'\d+\s*factorial',        # 阶乘
            r'probability.*\d+',       # 概率计算
        ]
        
        # 长度和复杂度阈值
        self.max_prompt_length = 300      # 降低最大长度
        self.max_candidate_length = 1000  # 候选回答最大长度
        self.max_latex_ratio = 0.3        # LaTeX内容占比阈值
    
    def analyze_prompt_content(self, prompt: str) -> Dict[str, Any]:
        """分析prompt内容特征"""
        prompt_lower = prompt.lower()
        
        analysis = {
            'length': len(prompt),
            'word_count': len(prompt.split()),
            'has_math_keywords': False,
            'has_unsuitable_keywords': False,
            'math_keyword_matches': [],
            'unsuitable_keyword_matches': [],
            'math_pattern_matches': [],
            'latex_ratio': 0.0,
            'is_question': False,
            'content_type': 'unknown'
        }
        
        # 检查数学/科学关键词
        for keyword in self.math_science_keywords:
            if keyword.lower() in prompt_lower:
                analysis['math_keyword_matches'].append(keyword)
                analysis['has_math_keywords'] = True
        
        # 检查不适合的关键词
        for keyword in self.unsuitable_keywords:
            if keyword.lower() in prompt_lower:
                analysis['unsuitable_keyword_matches'].append(keyword)
                analysis['has_unsuitable_keywords'] = True
        
        # 检查数学模式
        for pattern in self.math_patterns:
            matches = re.findall(pattern, prompt, re.IGNORECASE)
            if matches:
                analysis['math_pattern_matches'].extend(matches)
        
        # 计算LaTeX内容占比
        latex_indicators = ['\\', '{', '}', 'begin{', 'end{', 'documentclass']
        latex_count = sum(prompt.count(indicator) for indicator in latex_indicators)
        analysis['latex_ratio'] = latex_count / max(len(prompt), 1)
        
        # 检查是否是问题形式
        question_indicators = ['?', 'how', 'what', 'calculate', 'find', 'determine']
        analysis['is_question'] = any(indicator in prompt_lower for indicator in question_indicators)
        
        # 确定内容类型
        if analysis['latex_ratio'] > self.max_latex_ratio:
            analysis['content_type'] = 'latex_formatting'
        elif analysis['has_unsuitable_keywords']:
            analysis['content_type'] = 'code_formatting_or_explanation'
        elif analysis['has_math_keywords'] or analysis['math_pattern_matches']:
            analysis['content_type'] = 'mathematical_or_scientific'
        elif analysis['is_question'] and analysis['length'] < self.max_prompt_length:
            analysis['content_type'] = 'potentially_suitable'
        else:
            analysis['content_type'] = 'unclear_or_unsuitable'
        
        return analysis
    
    def analyze_candidates(self, candidates: List[str]) -> Dict[str, Any]:
        """分析候选回答的特征"""
        if not candidates:
            return {'suitable': False, 'reason': 'No candidates provided'}
        
        analysis = {
            'candidate_count': len(candidates),
            'avg_length': 0,
            'max_length': 0,
            'latex_heavy_count': 0,
            'code_heavy_count': 0,
            'suitable_count': 0
        }
        
        total_length = 0
        for candidate in candidates:
            # 从候选回答中提取实际内容
            if isinstance(candidate, str):
                # 如果是JSON格式的字符串，尝试解析
                if candidate.strip().startswith('['):
                    try:
                        parsed = json.loads(candidate)
                        if isinstance(parsed, list) and len(parsed) > 0:
                            # 提取assistant的回答
                            for item in parsed:
                                if isinstance(item, dict) and item.get('role') == 'assistant':
                                    content = item.get('content', '')
                                    break
                            else:
                                content = candidate
                        else:
                            content = candidate
                    except:
                        content = candidate
                else:
                    content = candidate
            else:
                content = str(candidate)
            
            length = len(content)
            total_length += length
            analysis['max_length'] = max(analysis['max_length'], length)
            
            # 检查LaTeX内容占比
            latex_count = content.count('\\') + content.count('begin{') + content.count('documentclass')
            if latex_count > 10:  # LaTeX内容较多
                analysis['latex_heavy_count'] += 1
            
            # 检查代码内容
            code_indicators = ['def ', 'class ', 'import ', 'function', '#include', 'public class']
            if any(indicator in content for indicator in code_indicators):
                analysis['code_heavy_count'] += 1
            
            # 检查是否适合验证（长度合理且非格式化内容）
            if length < self.max_candidate_length and latex_count < 5:
                analysis['suitable_count'] += 1
        
        analysis['avg_length'] = total_length / len(candidates)
        
        return analysis
    
    def is_suitable_for_verification(self, prompt: str, candidates: List[str]) -> Tuple[bool, str, Dict[str, Any]]:
        """
        判断样本是否适合代码验证
        
        Returns:
            Tuple[bool, str, Dict]: (是否适合, 原因, 详细分析)
        """
        prompt_analysis = self.analyze_prompt_content(prompt)
        candidate_analysis = self.analyze_candidates(candidates)
        
        details = {
            'prompt_analysis': prompt_analysis,
            'candidate_analysis': candidate_analysis
        }
        
        # 1. 检查prompt长度
        if prompt_analysis['length'] > self.max_prompt_length:
            return False, f"Prompt太长 ({prompt_analysis['length']} > {self.max_prompt_length})", details
        
        # 2. 检查候选回答长度
        if candidate_analysis['avg_length'] > self.max_candidate_length:
            return False, f"候选回答太长 (平均{candidate_analysis['avg_length']:.0f} > {self.max_candidate_length})", details
        
        # 3. 检查LaTeX内容占比
        if prompt_analysis['latex_ratio'] > self.max_latex_ratio:
            return False, f"LaTeX内容过多 (占比{prompt_analysis['latex_ratio']:.2f} > {self.max_latex_ratio})", details
        
        # 4. 检查是否是LaTeX格式化任务
        if candidate_analysis['latex_heavy_count'] >= len(candidates) * 0.5:
            return False, "主要是LaTeX格式化任务", details
        
        # 5. 检查不适合的关键词
        if prompt_analysis['has_unsuitable_keywords']:
            return False, f"包含不适合关键词: {prompt_analysis['unsuitable_keyword_matches'][:3]}", details
        
        # 6. 检查是否包含数学/科学内容
        if prompt_analysis['has_math_keywords'] or prompt_analysis['math_pattern_matches']:
            return True, f"包含数学/科学内容: {prompt_analysis['math_keyword_matches'][:3]}", details
        
        # 7. 检查是否是简短的问题
        if (prompt_analysis['is_question'] and 
            prompt_analysis['length'] < 200 and 
            candidate_analysis['suitable_count'] > 0):
            return True, "简短问题且候选回答长度合理", details
        
        # 8. 默认不适合
        return False, f"内容类型不明确: {prompt_analysis['content_type']}", details

def filter_ucla_dataset(
    prompts: List[str],
    candidates_list: List[List[str]],
    max_samples: Optional[int] = None,
    debug: bool = False
) -> Tuple[List[str], List[List[str]], Dict[str, Any]]:
    """
    过滤UCLA数据集，只保留适合代码验证的样本
    """
    filter_obj = UCLADatasetFilter()
    
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
    
    print(f"🔍 开始过滤UCLA数据集，原始样本数: {len(prompts)}")
    
    for i, (prompt, candidates) in enumerate(zip(prompts, candidates_list)):
        is_suitable, reason, details = filter_obj.is_suitable_for_verification(prompt, candidates)
        
        # 统计
        content_type = details['prompt_analysis']['content_type']
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
            if len(filter_stats['examples']['suitable']) < 3:
                filter_stats['examples']['suitable'].append({
                    'index': i,
                    'prompt': prompt[:100] + "..." if len(prompt) > 100 else prompt,
                    'reason': reason,
                    'math_keywords': details['prompt_analysis']['math_keyword_matches'][:3]
                })
            
            if debug:
                print(f"✅ 样本 {i}: 适合验证")
                print(f"   问题: {prompt[:50]}...")
                print(f"   原因: {reason}")
        else:
            filter_stats['unsuitable_count'] += 1
            
            # 保存不适合的示例
            if len(filter_stats['examples']['unsuitable']) < 3:
                filter_stats['examples']['unsuitable'].append({
                    'index': i,
                    'prompt': prompt[:100] + "..." if len(prompt) > 100 else prompt,
                    'reason': reason,
                    'content_type': content_type
                })
            
            if debug:
                print(f"❌ 样本 {i}: 不适合验证")
                print(f"   问题: {prompt[:50]}...")
                print(f"   原因: {reason}")
        
        # 达到最大样本数时停止
        if max_samples and filter_stats['suitable_count'] >= max_samples:
            print(f"⚠️  已达到最大样本数限制: {max_samples}")
            break
        
        # 每100个样本显示一次进度
        if (i + 1) % 100 == 0:
            print(f"📊 过滤进度: {i+1}/{len(prompts)}, 适合: {filter_stats['suitable_count']}")
    
    return filtered_prompts, filtered_candidates, filter_stats

def print_ucla_filter_report(filter_stats: Dict[str, Any]):
    """打印UCLA数据集过滤报告"""
    print("\n" + "="*70)
    print("📊 UCLA数据集过滤报告")
    print("="*70)
    print(f"原始样本数: {filter_stats['total_original']}")
    print(f"适合验证: {filter_stats['suitable_count']}")
    print(f"不适合验证: {filter_stats['unsuitable_count']}")
    print(f"过滤率: {filter_stats['suitable_count']/filter_stats['total_original']*100:.1f}%")
    
    print(f"\n📋 内容类型分布:")
    for content_type, count in filter_stats['content_types'].items():
        percentage = count / filter_stats['total_original'] * 100
        print(f"  {content_type}: {count} ({percentage:.1f}%)")
    
    print(f"\n🔍 主要过滤原因:")
    sorted_reasons = sorted(filter_stats['filter_reasons'].items(), key=lambda x: x[1], reverse=True)
    for reason, count in sorted_reasons[:5]:
        print(f"  {reason}: {count}")
    
    print(f"\n✅ 适合验证的示例:")
    for example in filter_stats['examples']['suitable']:
        print(f"  - [{example['index']}] {example['prompt']}")
        print(f"    原因: {example['reason']}")
        if example.get('math_keywords'):
            print(f"    数学关键词: {example['math_keywords']}")
    
    print(f"\n❌ 不适合验证的示例:")
    for example in filter_stats['examples']['unsuitable']:
        print(f"  - [{example['index']}] {example['prompt']}")
        print(f"    原因: {example['reason']}")
        print(f"    类型: {example['content_type']}")
    
    print("="*70)

# 测试函数
def test_ucla_filter():
    """测试UCLA数据集过滤器"""
    print("🧪 测试UCLA数据集过滤器")
    print("="*50)
    
    # 模拟UCLA数据集样本
    test_samples = [
        # 适合的（数学计算）
        {
            'prompt': 'Calculate the sum of numbers from 1 to 100',
            'candidates': ['The sum is 5050', 'Sum = 100*101/2 = 5050', '5050']
        },
        {
            'prompt': 'How many times does the letter "a" appear in "banana"?',
            'candidates': ['3 times', 'The letter a appears 3 times', 'Count: 3']
        },
        
        # 不适合的（LaTeX格式化）
        {
            'prompt': 'Please provide the content structure of the following text using [Latex] data format...',
            'candidates': [
                '[{"role": "assistant", "content": "\\\\documentclass{article}\\\\usepackage{booktabs}..."}]',
                '[{"role": "assistant", "content": "\\\\begin{document}\\\\maketitle..."}]'
            ]
        },
        
        # 不适合的（代码解释）
        {
            'prompt': 'Explain the design patterns used in this C# code',
            'candidates': ['This code uses Factory pattern...', 'The patterns include Singleton...']
        }
    ]
    
    prompts = [sample['prompt'] for sample in test_samples]
    candidates_list = [sample['candidates'] for sample in test_samples]
    
    filtered_prompts, filtered_candidates, stats = filter_ucla_dataset(
        prompts, candidates_list, debug=True
    )
    
    print_ucla_filter_report(stats)

if __name__ == "__main__":
    test_ucla_filter()