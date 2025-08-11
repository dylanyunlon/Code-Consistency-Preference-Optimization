#!/usr/bin/env python3
"""
Fixed Quick Start Script - 修复版快速测试脚本
解决导入和语法问题，直接在文件中包含所需的类
"""

import asyncio
import logging
import os
import sys
import re
import hashlib
import time
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VerificationStatus(Enum):
    SUCCESS = "success"
    EXECUTION_FAILED = "execution_failed"
    PARSE_FAILED = "parse_failed"
    NO_CODE_GENERATED = "no_code_generated"
    TIMEOUT = "timeout"
    ERROR = "error"

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

class AnswerExtractor:
    """答案提取器"""
    
    def __init__(self):
        self.patterns = [
            r'答案是[：:]?\s*([+-]?\d+(?:\.\d+)?)',
            r'结果是[：:]?\s*([+-]?\d+(?:\.\d+)?)',
            r'有\s*([+-]?\d+)\s*个',
            r'共\s*([+-]?\d+)\s*个',
            r'总共\s*([+-]?\d+)\s*个',
            r'([+-]?\d+)\s*个',
            r'等于\s*([+-]?\d+(?:\.\d+)?)',
            r'为\s*([+-]?\d+(?:\.\d+)?)',
            r'([+-]?\d+(?:\.\d+)?)\s*平方米',
            r'([+-]?\d+(?:\.\d+)?)\s*米',
            r'^([+-]?\d+(?:\.\d+)?)$',
        ]
    
    def extract_answer(self, text: str) -> Optional[str]:
        if not text:
            return None
        
        text = re.sub(r'\s+', ' ', text.strip())
        
        for pattern in self.patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                answer = matches[-1]
                try:
                    float(answer)
                    return answer
                except ValueError:
                    continue
        
        numbers = re.findall(r'[+-]?\d+(?:\.\d+)?', text)
        if numbers:
            return numbers[-1]
        
        return None

class CodeResultExtractor:
    """代码执行结果提取器"""
    
    def extract_result(self, stdout: str, stderr: str = "") -> Optional[str]:
        if not stdout:
            return None
        
        lines = [line.strip() for line in stdout.strip().split('\n') if line.strip()]
        if not lines:
            return None
        
        # 查找包含数字的行（从后往前找）
        for line in reversed(lines):
            if any(keyword in line.lower() for keyword in ['info', 'debug', 'warning', 'error']):
                continue
            
            numbers = re.findall(r'[+-]?\d+(?:\.\d+)?', line)
            if numbers:
                return numbers[-1]
        
        return None

class SimpleExecutionVerifier:
    """简化版执行验证器"""
    
    def __init__(
        self,
        base_url: str = "https://8.134.217.190:17432",
        username: str = "newuser",
        password: str = "newPass123"
    ):
        self.base_url = base_url
        self.username = username
        self.password = password
        
        self.answer_extractor = AnswerExtractor()
        self.code_extractor = CodeResultExtractor()
        self.client = None
        
    async def __aenter__(self):
        try:
            # 尝试导入enhanced_client_example
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from enhanced_client_example import EnhancedChatBotClient
            
            self.client = EnhancedChatBotClient(self.base_url)
            await self.client.__aenter__()
            await self.client.login(self.username, self.password)
            logger.info("SimpleExecutionVerifier initialized successfully")
            return self
        except ImportError as e:
            logger.error(f"Cannot import enhanced_client_example.py: {e}")
            raise ImportError("请确保 enhanced_client_example.py 在当前目录中")
        except Exception as e:
            logger.error(f"Failed to initialize verifier: {e}")
            raise
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.client:
            await self.client.__aexit__(exc_type, exc_val, exc_tb)
    
    def _generate_verification_id(self, question: str, ai_response: str) -> str:
        content = f"{question}||{ai_response}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    async def verify_response(
        self,
        question: str,
        ai_response: str,
        model: str = "claude-opus-4-20250514-all"
    ) -> VerificationResult:
        """验证AI回答"""
        verification_id = self._generate_verification_id(question, ai_response)
        start_time = time.time()
        
        try:
            # 步骤1: 构建代码生成提示
            code_prompt = self._build_code_prompt(question)
            logger.info(f"发送代码生成请求: {question[:50]}...")
            
            # 步骤2: 发送请求获取AI回答
            response = await self.client.send_code_request(
                prompt=code_prompt,
                auto_execute=False,
                model=model
            )
            
            if not response.get("success"):
                return self._create_error_result(
                    verification_id,
                    VerificationStatus.ERROR,
                    f"Code generation failed: {response.get('error')}",
                    time.time() - start_time,
                    ai_response
                )
            
            # 步骤3: 提取代码块ID
            data = response["data"]
            extracted_codes = data.get("metadata", {}).get("extracted_codes", [])
            
            if not extracted_codes:
                return self._create_error_result(
                    verification_id,
                    VerificationStatus.NO_CODE_GENERATED,
                    "No code blocks generated",
                    time.time() - start_time,
                    ai_response
                )
            
            # 步骤4: 获取代码块信息
            code_info = extracted_codes[0]
            code_id = code_info.get("id")
            code_content = code_info.get("content", "")
            
            if not code_id:
                return self._create_error_result(
                    verification_id,
                    VerificationStatus.ERROR,
                    "Code block not saved properly",
                    time.time() - start_time,
                    ai_response
                )
            
            logger.info(f"Found code block ID: {code_id}")
            
            # 步骤5: 执行代码块
            logger.info(f"Executing code block {code_id}...")
            exec_result = await self.client.execute_code(code_id)
            
            execution_time = time.time() - start_time
            
            # 步骤6: 处理执行结果
            if not exec_result.get("success"):
                return VerificationResult(
                    verified=False,
                    status=VerificationStatus.EXECUTION_FAILED,
                    ai_answer=None,
                    code_answer=None,
                    confidence=0.0,
                    execution_time=execution_time,
                    code_generated=code_content,
                    code_id=code_id,
                    stdout=exec_result.get("stdout", ""),
                    stderr=exec_result.get("stderr", ""),
                    error_message=f"Code execution failed: {exec_result.get('stderr', 'Unknown error')}",
                    verification_id=verification_id,
                    raw_ai_response=data.get("content", "")
                )
            
            # 步骤7: 提取和比较答案
            ai_answer = self.answer_extractor.extract_answer(ai_response)
            code_answer = self.code_extractor.extract_result(
                exec_result.get("stdout", ""),
                exec_result.get("stderr", "")
            )
            
            # 步骤8: 计算验证结果
            verified, confidence = self._compare_answers(ai_answer, code_answer)
            status = VerificationStatus.SUCCESS if verified else VerificationStatus.PARSE_FAILED
            
            result = VerificationResult(
                verified=verified,
                status=status,
                ai_answer=ai_answer,
                code_answer=code_answer,
                confidence=confidence,
                execution_time=execution_time,
                code_generated=code_content,
                code_id=code_id,
                stdout=exec_result.get("stdout", ""),
                stderr=exec_result.get("stderr", ""),
                error_message="",
                verification_id=verification_id,
                raw_ai_response=data.get("content", "")
            )
            
            logger.info(f"Verification completed: verified={verified}, confidence={confidence:.3f}")
            return result
            
        except Exception as e:
            logger.error(f"Verification failed: {e}")
            return self._create_error_result(
                verification_id,
                VerificationStatus.ERROR,
                str(e),
                time.time() - start_time,
                ai_response
            )
    
    def _build_code_prompt(self, question: str) -> str:
        """构建代码生成提示 - 使用与enhanced_client_example.py相同的格式"""
        return f"""python版本为3.10.创建一个包含Python代码的脚本用于解决以下问题：{question}"""
    
    def _compare_answers(self, ai_answer: Optional[str], code_answer: Optional[str]) -> Tuple[bool, float]:
        """比较AI答案和代码执行结果"""
        if ai_answer is None or code_answer is None:
            return False, 0.1
        
        try:
            ai_num = float(ai_answer)
            code_num = float(code_answer)
            
            if abs(ai_num - code_num) < 1e-10:
                return True, 1.0
            
            if code_num != 0:
                relative_error = abs(ai_num - code_num) / abs(code_num)
                if relative_error < 1e-6:
                    return True, 0.95
                elif relative_error < 1e-4:
                    return True, 0.8
                elif relative_error < 1e-2:
                    return True, 0.6
                else:
                    return False, 0.2
            else:
                if abs(ai_num) < 1e-6:
                    return True, 0.95
                else:
                    return False, 0.1
            
        except (ValueError, TypeError):
            if ai_answer.strip().lower() == code_answer.strip().lower():
                return True, 0.9
            else:
                return False, 0.1
    
    def _create_error_result(
        self,
        verification_id: str,
        status: VerificationStatus,
        error_message: str,
        execution_time: float,
        raw_ai_response: str
    ) -> VerificationResult:
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
            raw_ai_response=raw_ai_response
        )

async def test_basic_verification():
    """基础验证测试"""
    print("🚀 测试代码执行验证功能")
    print("=" * 50)
    print("工作流程: 发送问题 → 获取代码块 → 执行代码块 → 比较结果")
    print("=" * 50)
    
    # 测试案例
    test_cases = [
        {
            "question": "strawberry中有几个字母r？",
            "ai_answer": "strawberry中有3个字母r"
        },
        {
            "question": "计算 15 + 27 × 3 的结果",
            "ai_answer": "15 + 27 × 3 = 15 + 81 = 96"
        },
        {
            "question": "一个正方形的边长是6米，它的面积是多少平方米？",
            "ai_answer": "正方形面积 = 边长² = 6² = 36平方米"
        }
    ]
    
    try:
        async with SimpleExecutionVerifier() as verifier:
            print("✅ 验证器初始化成功")
            
            for i, case in enumerate(test_cases, 1):
                print(f"\n--- 测试案例 {i} ---")
                print(f"问题: {case['question']}")
                print(f"AI回答: {case['ai_answer']}")
                print("正在验证...")
                
                result = await verifier.verify_response(
                    case['question'], 
                    case['ai_answer']
                )
                
                print(f"验证结果: {'✅ 通过' if result.verified else '❌ 未通过'}")
                print(f"置信度: {result.confidence:.3f}")
                print(f"状态: {result.status.value}")
                print(f"AI提取答案: {result.ai_answer}")
                print(f"代码计算答案: {result.code_answer}")
                print(f"执行时间: {result.execution_time:.2f}s")
                
                if result.code_id:
                    print(f"代码块ID: {result.code_id}")
                
                if result.stdout:
                    print(f"执行输出: {result.stdout.strip()}")
                
                if result.error_message:
                    print(f"错误信息: {result.error_message}")
                
                print("-" * 50)
            
            print("\n🎉 基础验证测试完成!")
            
    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        print("请确保 enhanced_client_example.py 在当前目录中")
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        print("请检查服务器连接和配置")

async def test_single_step():
    """单步测试"""
    print("\n🔧 单步验证测试")
    print("=" * 30)
    
    question = "计算 2 + 3 × 4 的结果"
    ai_answer = "2 + 3 × 4 = 2 + 12 = 14"
    
    print(f"问题: {question}")
    print(f"AI回答: {ai_answer}")
    
    try:
        async with SimpleExecutionVerifier() as verifier:
            result = await verifier.verify_response(question, ai_answer)
            
            print(f"\n结果:")
            print(f"验证通过: {'✅' if result.verified else '❌'}")
            print(f"置信度: {result.confidence:.3f}")
            print(f"AI答案: {result.ai_answer}")
            print(f"代码答案: {result.code_answer}")
            
            if result.code_generated:
                print(f"\n生成的代码:")
                print(result.code_generated[:300] + "..." if len(result.code_generated) > 300 else result.code_generated)
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")

def test_extractors():
    """测试提取器"""
    print("\n🔍 测试答案提取功能")
    print("=" * 30)
    
    answer_extractor = AnswerExtractor()
    code_extractor = CodeResultExtractor()
    
    # 测试AI答案提取
    test_responses = [
        "答案是42",
        "有3个字母r",
        "面积等于36平方米",
        "strawberry中有3个r",
        "2 + 3 × 4 = 14"
    ]
    
    print("AI答案提取:")
    for response in test_responses:
        extracted = answer_extractor.extract_answer(response)
        print(f"'{response}' → {extracted}")
    
    # 测试代码输出提取
    test_outputs = [
        "结果: 42",
        "123",
        "2025-07-31 16:36:00,430 - INFO - 单词 'strawberry' 中字母 'r' 的个数为: 3"
    ]
    
    print("\n代码输出提取:")
    for output in test_outputs:
        extracted = code_extractor.extract_result(output)
        print(f"'{output}' → {extracted}")

async def interactive_mode():
    """交互模式"""
    print("\n💬 交互式测试")
    print("=" * 30)
    print("输入问题和AI回答进行验证，输入 'quit' 退出")
    
    try:
        async with SimpleExecutionVerifier() as verifier:
            while True:
                print("\n" + "-" * 30)
                question = input("问题: ").strip()
                if question.lower() == 'quit':
                    break
                if not question:
                    continue
                
                ai_answer = input("AI回答: ").strip()
                if not ai_answer:
                    continue
                
                print("验证中...")
                result = await verifier.verify_response(question, ai_answer)
                
                print(f"结果: {'✅' if result.verified else '❌'}")
                print(f"置信度: {result.confidence:.3f}")
                print(f"AI答案: {result.ai_answer} | 代码答案: {result.code_answer}")
                
    except Exception as e:
        print(f"❌ 交互模式失败: {e}")

async def main():
    """主函数"""
    print("🌟 代码执行验证系统 - 快速测试")
    print("=" * 40)
    
    while True:
        print("\n选择测试模式:")
        print("1. 基础验证测试")
        print("2. 单步测试")
        print("3. 提取器测试")
        print("4. 交互模式")
        print("5. 退出")
        
        choice = input("\n请选择 (1-5): ").strip()
        
        try:
            if choice == '1':
                await test_basic_verification()
            elif choice == '2':
                await test_single_step()
            elif choice == '3':
                test_extractors()
            elif choice == '4':
                await interactive_mode()
            elif choice == '5':
                print("👋 再见!")
                break
            else:
                print("❌ 无效选项")
        except Exception as e:
            print(f"❌ 执行出错: {e}")
            logger.exception("执行异常")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 用户中断")
    except Exception as e:
        print(f"❌ 程序异常: {e}")
        logger.exception("程序异常")