import sys
import os
# enhanced_client_example.py
import asyncio
import json
import ssl
import aiohttp
from typing import Optional, Dict, Any
from client import ChatBotAPI
from datetime import datetime

class EnhancedChatBotClient(ChatBotAPI):
    """增强的聊天机器人客户端 - SSL修复版"""
    
    def __init__(self, base_url: str = "https://httpsnet.top:17432"):
        super().__init__(base_url)
        self.v2_endpoint = "/api/chat/v2"
    
    async def __aenter__(self):
        """异步上下文管理器入口 - 创建SSL修复的会话"""
        # 创建宽松的SSL上下文 - 解决证书问题
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False  # 禁用主机名检查
        ssl_context.verify_mode = ssl.CERT_NONE  # 禁用证书验证
        
        print("⚠️  警告: 已禁用SSL证书验证（仅用于测试）")
        
        # 创建连接器
        connector = aiohttp.TCPConnector(
            ssl=ssl_context,
            limit=100,
            limit_per_host=10,
            keepalive_timeout=30,
            enable_cleanup_closed=True
        )
        
        # 创建超时配置
        timeout = aiohttp.ClientTimeout(
            total=60,
            connect=30
        )
        
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={'User-Agent': 'Enhanced-ChatBot-Client/1.0'}
        )
        
        return self
    
    async def send_code_request(
        self,
        prompt: str,
        language: Optional[str] = None,
        auto_execute: bool = False,
        setup_cron: bool = False,
        cron_expression: Optional[str] = None,
        conversation_id: Optional[str] = None,
        model: str = "claude-sonnet-4-20250514-all"  # 添加默认模型
    ) -> Dict[str, Any]:
        """发送代码生成请求"""
        # 构建消息内容
        message_content = prompt
        
        # 如果指定了cron表达式，添加到prompt中
        if setup_cron and cron_expression:
            message_content += f"\n\n请设计脚本以便通过cron表达式 {cron_expression} 定期运行"
        
        payload = {
            "content": message_content,  # 确保使用 content 字段
            "model": model,
            "extract_code": True,
            "auto_execute": auto_execute,
            "setup_cron": setup_cron,
            "conversation_id": conversation_id
        }
        
        if language:
            payload["code_language"] = language
        
        # 打印调试信息
        print(f"[DEBUG] Sending payload: {json.dumps(payload, indent=2)}")
        
        try:
            async with self.session.post(
                f"{self.base_url}{self.v2_endpoint}/message",
                json=payload,
                headers=self.get_headers()
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    print(f"[ERROR] Response status: {response.status}")
                    print(f"[ERROR] Response text: {error_text}")
                    raise Exception(f"Failed to send code request: {error_text}")
        except aiohttp.ClientConnectorError as e:
            raise Exception(f"连接错误: {e}")
        except asyncio.TimeoutError:
            raise Exception("请求超时")
        except Exception as e:
            raise Exception(f"请求失败: {e}")
    
    async def execute_code(
        self,
        code_id: str,
        parameters: Optional[Dict[str, str]] = None,
        timeout: int = 300
    ) -> Dict[str, Any]:
        """执行已保存的代码"""
        payload = {
            "code_id": code_id,
            "parameters": parameters or {},
            "timeout": timeout
        }
        
        async with self.session.post(
            f"{self.base_url}{self.v2_endpoint}/execute-code",
            json=payload,
            headers=self.get_headers()
        ) as response:
            if response.status == 200:
                return await response.json()
            else:
                error_text = await response.text()
                raise Exception(f"Failed to execute code: {error_text}")
    
    async def setup_cron(
        self,
        code_id: str,
        cron_expression: str,
        job_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """为代码设置定时任务"""
        payload = {
            "code_id": code_id,
            "cron_expression": cron_expression,
            "job_name": job_name
        }
        
        async with self.session.post(
            f"{self.base_url}{self.v2_endpoint}/setup-cron",
            json=payload,
            headers=self.get_headers()
        ) as response:
            if response.status == 200:
                return await response.json()
            else:
                error_text = await response.text()
                raise Exception(f"Failed to setup cron: {error_text}")
    
    async def get_code_templates(
        self,
        language: Optional[str] = None,
        task_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """获取代码模板"""
        params = {}
        if language:
            params["language"] = language
        if task_type:
            params["task_type"] = task_type
        
        async with self.session.get(
            f"{self.base_url}{self.v2_endpoint}/code-templates",
            params=params,
            headers=self.get_headers()
        ) as response:
            if response.status == 200:
                return await response.json()
            else:
                error_text = await response.text()
                raise Exception(f"Failed to get templates: {error_text}")


async def example_monitor_script():
    """示例：创建系统监控脚本"""
    async with EnhancedChatBotClient(base_url="https://httpsnet.top:17432") as client:
        # 登录
        await client.login("newuser", "newPass123")
        print("✅ 登录成功\n")
        
        # 1. 请求生成监控脚本
        print("📝 请求生成系统监控脚本...")
        
        # 构建完整的提示内容
        monitor_prompt = """创建一个Python脚本用于监控系统状态：
1. 监控CPU使用率（阈值80%）
2. 监控内存使用率（阈值90%）
3. 监控磁盘使用率（阈值85%）
4. 超过阈值时记录到error.log
5. 每次运行生成JSON格式的状态报告
6. 适合每5分钟运行一次"""

        
        response = await client.send_code_request(
            prompt=monitor_prompt,
            language="python",
            auto_execute=True,  # 自动执行测试
            setup_cron=True,    # 自动设置定时任务
            cron_expression="*/5 * * * *",  # 每5分钟
            model="claude-sonnet-4-20250514-all"  # 指定模型
        )
        
        if response["success"]:
            data = response["data"]
            print(f"✅ AI响应成功")
            print(f"📄 会话ID: {data['conversation_id']}")
            
            # 检查提取的代码
            if "metadata" in data and "extracted_codes" in data["metadata"]:
                codes = data["metadata"]["extracted_codes"]
                print(f"\n💾 提取到 {len(codes)} 个代码块:")
                
                for i, code in enumerate(codes):
                    print(f"\n代码块 {i+1}:")
                    print(f"  - 语言: {code['language']}")
                    print(f"  - 有效: {'✅' if code['valid'] else '❌'}")
                    print(f"  - 已保存: {'✅' if code['saved'] else '❌'}")
                    
                    if code.get('saved') and code.get('id'):
                        print(f"  - ID: {code['id']}")
                        
                        # 检查执行结果
                        if "executions" in data["metadata"]:
                                # 创建输出目录
                            output_dir = "output"
                            if not os.path.exists(output_dir):
                                os.makedirs(output_dir)
                                print(f"  - 创建输出目录: {output_dir}/")
                            
                            # 可选：按日期创建子目录
                            date_dir = datetime.now().strftime("%Y%m%d")
                            full_output_dir = os.path.join(output_dir, date_dir)

                            if not os.path.exists(full_output_dir):
                                os.makedirs(full_output_dir)
                            for exec_result in data["metadata"]["executions"]:
                                if exec_result["code_id"] == code["id"]:
                                    print(f"  - 执行结果: {'✅ 成功' if exec_result['success'] else '❌ 失败'}")
                                    if exec_result.get("output"):
                                        # 创建输出文件名，包含时间戳
                                        timestamp = datetime.now().strftime("%H%M%S")
                                        output_filename = f"execution_output_{code['id']}_{timestamp}.txt"
                                        output_filepath = os.path.join(full_output_dir, output_filename)
                                        
                                        # 保存完整输出
                                        with open(output_filepath, 'w', encoding='utf-8') as f:
                                            f.write(f"代码ID: {code['id']}\n")
                                            # f.write(code.get('content'))
                                            f.write(f"执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                                            f.write(f"执行状态: {'成功' if exec_result['success'] else '失败'}\n")
                                            f.write("-" * 50 + "\n")
                                            f.write("完整输出:\n")
                                            f.write(exec_result['output'])
                                        
                                        print(f"  - 完整输出已保存到: {output_filepath}")
                                        print(f"  - 输出预览:\n{exec_result['output'][:200]}...")
                        
                        # 检查定时任务
                        if "cron_jobs" in data["metadata"]:
                            for cron_job in data["metadata"]["cron_jobs"]:
                                if cron_job.get("success"):
                                    print(f"  - ⏰ 定时任务: {cron_job['job_info']['job_name']}")
                                    print(f"  - 下次运行: {cron_job.get('next_run', 'N/A')}")


async def example_backup_script():
    """示例：创建备份脚本"""
    async with EnhancedChatBotClient() as client:
        # 登录
        await client.login("newuser", "newPass123")
        print("✅ 登录成功\n")
        
        # 获取备份模板
        print("📋 获取备份脚本模板...")
        templates = await client.get_code_templates(language="bash", task_type="backup")
        
        if templates.get("success"):
            template_content = templates.get("templates", {}).get("bash", {}).get("backup", "")
            print(f"模板内容:\n{template_content}\n")
        
        # 基于模板生成脚本
        response = await client.send_code_request(
            prompt=template_content + "\n\n备份目录：/var/www/html，备份到：/backup/web/",
            language="bash",
            auto_execute=False,  # 不自动执行（备份脚本需要谨慎）
            setup_cron=True,
            cron_expression="0 2 * * *",  # 每天凌晨2点
            model="claude-sonnet-4-20250514-all"
        )
        
        if response["success"]:
            print("✅ 备份脚本生成成功")
            
            # 获取生成的代码ID
            codes = response["data"]["metadata"].get("extracted_codes", [])
            if codes and codes[0].get("saved"):
                code_id = codes[0]["id"]
                print(f"💾 代码ID: {code_id}")
                
                # 手动执行测试（带参数）
                print("\n🚀 执行备份脚本测试...")
                exec_response = await client.execute_code(
                    code_id=code_id,
                    parameters={
                        "BACKUP_SOURCE": "/tmp/test_source",
                        "BACKUP_DEST": "/tmp/test_backup"
                    }
                )
                
                if exec_response["success"]:
                    result = exec_response["data"]["result"]
                    print(f"执行状态: {'✅ 成功' if result['success'] else '❌ 失败'}")
                    print(f"执行报告:\n{exec_response['data']['report']}")


async def example_interactive_code_chat():
    """示例：交互式代码聊天"""
    async with EnhancedChatBotClient(base_url="https://8.134.217.190:17432") as client:
        # 登录
        await client.login("newuser", "newPass123")
        print("🤖 增强代码聊天模式")
        print("="*50)
        print("命令: /templates - 查看模板, /exec <code_id> - 执行代码")
        print("     /cron <code_id> <expression> - 设置定时任务")
        print("     /exit - 退出")
        print("="*50 + "\n")
        
        conversation_id = None
        saved_codes = []  # 保存代码ID列表
        
        while True:
            try:
                user_input = input("\n👤 你: ").strip()
                
                if not user_input:
                    continue
                
                # 处理命令
                if user_input.lower() == '/exit':
                    print("👋 再见！")
                    break
                
                elif user_input.lower() == '/templates':
                    try:
                        templates = await client.get_code_templates()
                        print("\n📋 可用模板:")
                        for lang, tasks in templates.get("templates", {}).items():
                            print(f"\n{lang.upper()}:")
                            for task in tasks.keys():
                                print(f"  - {task}")
                    except Exception as e:
                        print(f"❌ 获取模板失败: {e}")
                    continue
                
                elif user_input.startswith('/exec '):
                    code_id = user_input[6:].strip()
                    print(f"\n🚀 执行代码 {code_id}...")
                    try:
                        result = await client.execute_code(code_id)
                        if result["success"]:
                            print(result["data"]["report"])
                    except Exception as e:
                        print(f"❌ 执行失败: {e}")
                    continue
                
                elif user_input.startswith('/cron '):
                    parts = user_input[6:].split(maxsplit=1)
                    if len(parts) == 2:
                        code_id, cron_expr = parts
                        print(f"\n⏰ 设置定时任务...")
                        try:
                            result = await client.setup_cron(code_id, cron_expr)
                            if result["success"]:
                                print(f"✅ 定时任务设置成功")
                        except Exception as e:
                            print(f"❌ 设置失败: {e}")
                    continue
                
                # 检测是否是代码请求
                is_code_request = any(
                    keyword in user_input.lower() 
                    for keyword in ['脚本', '代码', 'script', 'code', '编写', 'write']
                )
                
                # 发送消息
                response = await client.send_code_request(
                    prompt=user_input,
                    auto_execute=is_code_request,  # 代码请求自动执行测试
                    conversation_id=conversation_id,
                    model="claude-sonnet-4-20250514-all"  # 使用你原来的模型
                )
                
                if response["success"]:
                    data = response["data"]
                    conversation_id = data["conversation_id"]
                    
                    # 显示AI回复
                    print(f"\n🤖 AI: {data['content'][:500]}...")
                    
                    # 如果有提取的代码
                    if data.get("metadata", {}).get("extracted_codes"):
                        codes = data["metadata"]["extracted_codes"]
                        print(f"\n💾 提取到 {len(codes)} 个代码块")
                        
                        for code in codes:
                            if code.get("saved") and code.get("id"):
                                saved_codes.append(code["id"])
                                print(f"  - [{code['language']}] ID: {code['id']}")
                    
                    # 显示建议
                    if data.get("follow_up_questions"):
                        print("\n💡 建议操作:")
                        for i, q in enumerate(data["follow_up_questions"][:3]):
                            print(f"  {i+1}. {q}")
                
            except KeyboardInterrupt:
                print("\n\n👋 检测到中断，退出...")
                break
            except Exception as e:
                print(f"\n❌ 错误: {e}")


# 主函数
async def main():
    """运行示例"""
    print("🚀 ChatBot API 增强功能示例 (SSL修复版)")
    print("="*50)
    print("1. 创建监控脚本（自动执行+定时任务）")
    print("2. 创建备份脚本（手动执行）")
    print("3. 交互式代码聊天")
    print("="*50)
    
    choice = input("\n请选择示例 (1-3): ").strip()
    
    if choice == "1":
        await example_monitor_script()
    elif choice == "2":
        await example_backup_script()
    elif choice == "3":
        await example_interactive_code_chat()
    else:
        print("无效选择")


if __name__ == "__main__":
    asyncio.run(main())