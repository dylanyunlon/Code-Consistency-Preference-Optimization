#!/usr/bin/env python3
"""
Exact Format Test V2 - 基于最佳实践的精确格式测试脚本
参考了Stack Overflow和Python官方文档的答案提取最佳实践
"""

import asyncio
import logging
import os
import sys
import time
from typing import Optional, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExactFormatTesterV2:
    """精确格式测试器V2 - 集成最佳实践答案提取器"""
    
    def __init__(
        self,
        base_url: str = "https://8.134.217.190:17432",
        username: str = "newuser",
        password: str = "newPass123"
    ):
        self.base_url = base_url
        self.username = username
        self.password = password
        self.client = None
        self.extractor = None
        
    async def __aenter__(self):
        try:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from enhanced_client_example import EnhancedChatBotClient
            
            # 动态导入增强版提取器
            try:
                # 尝试导入V2版本（优先）
                from enhanced_answer_extractor_v2 import EnhancedAnswerExtractorV2
                self.extractor = EnhancedAnswerExtractorV2(debug=True)
                logger.info("使用增强版答案提取器V2")
            except ImportError:
                # 回退到原版本
                from enhanced_answer_extractor import EnhancedAnswerExtractor
                self.extractor = EnhancedAnswerExtractor()
                logger.info("使用原版增强答案提取器")
            
            self.client = EnhancedChatBotClient(self.base_url)
            await self.client.__aenter__()
            await self.client.login(self.username, self.password)
            logger.info("ExactFormatTesterV2 initialized successfully")
            return self
        except ImportError as e:
            raise ImportError(f"请确保相关文件在当前目录中: {e}")
        except Exception as e:
            logger.error(f"Failed to initialize: {e}")
            raise
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.client:
            await self.client.__aexit__(exc_type, exc_val, exc_tb)
    
    def build_exact_prompt(self, question: str) -> str:
        """构建与你日志中完全相同的prompt格式"""
        return f"""python版本为3.10.创建一个包含Python代码的脚本用于解决以下问题：{question}"""
    
    def extract_answers(self, ai_response: str, code_output: str) -> Tuple[Optional[str], Optional[str]]:
        """提取AI回答和代码输出中的答案"""
        
        # 从AI回答中提取答案
        if hasattr(self.extractor, 'extract_from_ai_response'):
            ai_answer = self.extractor.extract_from_ai_response(ai_response)
        else:
            # 回退方法
            ai_answer = self.extractor.extract_label(ai_response, 'digit')
        
        # 从代码输出中提取答案
        if hasattr(self.extractor, 'extract_from_code_output'):
            code_answer = self.extractor.extract_from_code_output(code_output)
        else:
            # 回退方法 - 使用简单的正则表达式
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
                # 提取所有数字，返回最后一个
                numbers = re.findall(r'\d+(?:\.\d+)?', code_output)
                code_answer = numbers[-1] if numbers else None
        
        return ai_answer, code_answer
    
    def compare_answers(self, ai_answer: str, code_answer: str) -> Tuple[bool, float, str]:
        """比较答案是否匹配"""
        if hasattr(self.extractor, 'compare_answers'):
            is_match, confidence = self.extractor.compare_answers(ai_answer, code_answer)
            reason = "增强版比较"
        else:
            # 回退比较方法
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
    
    async def test_single_question(
        self,
        question: str,
        expected_ai_answer: str,
        model: str = "claude-opus-4-20250514-all"
    ):
        """测试单个问题"""
        print(f"\n🧪 测试问题: {question}")
        print(f"期望AI回答: {expected_ai_answer}")
        print("-" * 60)
        
        # 步骤1: 构建精确的prompt
        prompt = self.build_exact_prompt(question)
        print(f"📝 发送的prompt: {prompt}")
        
        start_time = time.time()
        
        try:
            # 步骤2: 发送代码生成请求
            print("🔄 发送代码生成请求...")
            response = await self.client.send_code_request(
                prompt=prompt,
                auto_execute=False,
                model=model
            )
            
            if not response.get("success"):
                print(f"❌ 代码生成失败: {response.get('error')}")
                return False
            
            # 步骤3: 解析响应
            data = response["data"]
            ai_full_response = data.get("content", "")
            extracted_codes = data.get("metadata", {}).get("extracted_codes", [])
            
            print(f"🤖 AI完整回答:\n{ai_full_response[:200]}...")
            
            if not extracted_codes:
                print("❌ 没有提取到代码块") 
                return False
            
            # 步骤4: 获取代码块信息
            code_info = extracted_codes[0]
            code_id = code_info.get("id")
            
            if not code_id:
                print("❌ 代码块ID获取失败")
                return False
            
            print(f"✅ 获得代码块ID: {code_id}")
            
            # 步骤5: 执行代码块
            print(f"⚡ 执行代码块: /exec {code_id}")
            exec_result = await self.client.execute_code(code_id)
            
            execution_time = time.time() - start_time
            
            if not exec_result.get("success"):
                print(f"❌ 代码执行失败:")
                error_data = exec_result.get("data", {})
                print(f"   错误信息: {error_data}")
                return False
            
            # 步骤6: 分析执行结果
            result_data = exec_result.get("data", {})
            result_info = result_data.get("result", {})
            stdout = result_info.get("stdout", "")
            stderr = result_info.get("stderr", "")
            
            print(f"✅ 代码执行成功!")
            print(f"📤 完整执行输出:")
            print(stdout[:500] + "..." if len(stdout) > 500 else stdout)
            
            if stderr:
                print(f"⚠️  错误输出:")
                print(stderr)
            
            # 步骤7: 使用增强版提取器提取答案
            ai_extracted, code_extracted = self.extract_answers(expected_ai_answer, stdout)
            
            print(f"\n🔍 答案提取结果:")
            print(f"AI提取答案: '{ai_extracted}'")
            print(f"代码提取答案: '{code_extracted}'")
            
            # 如果有调试信息，显示详细过程
            if hasattr(self.extractor, 'debug_extraction_process'):
                debug_info = self.extractor.debug_extraction_process(stdout)
                print(f"\n🔧 调试信息:")
                print(f"  输入长度: {debug_info['input_length']}")
                print(f"  处理行数: {debug_info['lines_count']}")
                
                # 显示匹配的模式
                for category, matches in debug_info['matches_found'].items():
                    if matches:
                        print(f"  📋 {category}: {len(matches)} 个匹配")
                        for match in matches[:2]:  # 只显示前2个
                            print(f"    - 模式: {match['pattern'][:50]}...")
                            print(f"    - 匹配: {match['matches']}")
            
            # 步骤8: 验证一致性
            if ai_extracted and code_extracted:
                is_match, confidence, reason = self.compare_answers(ai_extracted, code_extracted)
                
                print(f"\n📊 答案验证:")
                print(f"比较方法: {reason}")
                print(f"置信度: {confidence:.3f}")
                
                if is_match:
                    print(f"✅ 验证通过!")
                    print(f"   匹配答案: {ai_extracted}")
                    verification_success = True
                else:
                    print(f"❌ 验证失败!")
                    print(f"   AI答案: {ai_extracted}")
                    print(f"   代码答案: {code_extracted}")
                    verification_success = False
            else:
                print(f"❌ 答案提取失败!")
                print(f"   AI提取结果: {ai_extracted}")
                print(f"   代码提取结果: {code_extracted}")
                verification_success = False
            
            print(f"⏱️  总执行时间: {execution_time:.2f}s")
            
            return verification_success
            
        except Exception as e:
            print(f"❌ 测试异常: {e}")
            logger.exception("测试异常")
            return False


async def main():
    """主测试函数"""
    print("🎯 精确格式测试V2 - 基于社区最佳实践")
    print("集成了Stack Overflow和Python官方文档的答案提取最佳实践")
    print("=" * 70)
    
    # 测试案例
    test_cases = [
        {
            "question": "strawberry 中有几个r",
            "ai_answer": "strawberry中有3个字母r"
        },
        {
            "question": "计算 2 + 3 × 4 的结果",
            "ai_answer": "2 + 3 × 4 = 2 + 12 = 14"
        },
        {
            "question": "一个正方形边长是5米，面积是多少平方米",
            "ai_answer": "正方形面积 = 5 × 5 = 25平方米"
        }
    ]
    
    try:
        async with ExactFormatTesterV2() as tester:
            print("✅ 测试器初始化成功")
            
            success_count = 0
            total_count = len(test_cases)
            
            for i, case in enumerate(test_cases, 1):
                print(f"\n{'='*20} 测试案例 {i}/{total_count} {'='*20}")
                
                success = await tester.test_single_question(
                    case["question"],
                    case["ai_answer"]
                )
                
                if success:
                    success_count += 1
                
                print(f"案例 {i} 结果: {'✅ 成功' if success else '❌ 失败'}")
            
            # 最终统计
            print(f"\n{'='*20} 最终统计 {'='*20}")
            print(f"总测试数: {total_count}")
            print(f"成功数: {success_count}")
            print(f"成功率: {success_count/total_count*100:.1f}%")
            
            if success_count == total_count:
                print("🎉 所有测试通过! 验证器工作正常!")
            else:
                print("⚠️  部分测试失败，建议检查具体错误信息")
                print("\n💡 调试建议:")
                print("1. 检查答案提取模式是否匹配你的输出格式")
                print("2. 验证代码执行结果的数据结构")
                print("3. 确认网络连接和服务器响应正常")
    
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        logger.error("测试失败", exc_info=True)
        print("\n🔧 故障排除:")
        print("1. 确保 enhanced_client_example.py 在当前目录")
        print("2. 如果有 enhanced_answer_extractor_v2.py，请确保它也在当前目录")
        print("3. 检查服务器地址和认证信息是否正确")
        print("4. 验证网络连接是否正常")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 用户中断测试")
    except Exception as e:
        print(f"❌ 程序异常: {e}")
        logger.exception("程序异常")