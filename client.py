
import asyncio
import json
import aiohttp
from typing import Optional, Dict, Any, Callable, List, AsyncGenerator
import websockets
from urllib.parse import urlencode
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class ChatBotAPI:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.token: Optional[str] = None
        self.refresh_token: Optional[str] = None
        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def __aenter__(self):
        # 配置超时和连接器
        timeout = aiohttp.ClientTimeout(total=60, connect=10)
        connector = aiohttp.TCPConnector(
            force_close=True,
            enable_cleanup_closed=True
        )
        self.session = aiohttp.ClientSession(
            timeout=timeout,
            connector=connector
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
            # 等待一下确保连接正确关闭
            await asyncio.sleep(0.1)
        if self.ws:
            await self.ws.close()
            
    def get_headers(self) -> Dict[str, str]:
        """获取请求头"""
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers
    
    def get_sse_headers(self) -> Dict[str, str]:
        """获取SSE请求头"""
        headers = {
            "Accept": "text/event-stream",
            "Cache-Control": "no-cache"
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers
    
    async def refresh_access_token(self) -> bool:
        """刷新访问令牌"""
        if not self.refresh_token:
            return False
            
        try:
            async with self.session.post(
                f"{self.base_url}/api/auth/refresh",
                json={"refresh_token": self.refresh_token}
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    self.token = data["access_token"]
                    return True
                return False
        except Exception as e:
            logger.error(f"Token refresh failed: {e}")
            return False
    
    # 1. 认证相关
    async def register(self, username: str, email: str, password: str) -> Dict[str, Any]:
        """用户注册"""
        async with self.session.post(
            f"{self.base_url}/api/auth/register",
            json={"username": username, "email": email, "password": password}
        ) as response:
            if response.status == 200:
                data = await response.json()
                self.token = data["access_token"]
                self.refresh_token = data["refresh_token"]
                return data
            else:
                error_text = await response.text()
                raise Exception(f"Registration failed: {error_text}")
    
    async def login(self, username: str, password: str) -> Dict[str, Any]:
        """用户登录"""
        data = aiohttp.FormData()
        data.add_field("username", username)
        data.add_field("password", password)
        
        async with self.session.post(
            f"{self.base_url}/api/auth/login",
            data=data
        ) as response:
            if response.status == 200:
                result = await response.json()
                self.token = result["access_token"]
                self.refresh_token = result["refresh_token"]
                return result
            else:
                error_text = await response.text()
                raise Exception(f"Login failed: {error_text}")
    
    # 2. 聊天相关
    async def send_message(
        self, 
        content: str, 
        conversation_id: Optional[str] = None,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        attachments: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """发送消息"""
        payload = {"content": content}
        if conversation_id:
            payload["conversation_id"] = conversation_id
        if model:
            payload["model"] = model
        if system_prompt:
            payload["system_prompt"] = system_prompt
        if attachments:
            payload["attachments"] = attachments
        
        headers = self.get_headers()
        
        try:
            async with self.session.post(
                f"{self.base_url}/api/chat/message",
                json=payload,
                headers=headers
            ) as response:
                # 检查认证
                if response.status == 401 and self.refresh_token:
                    # 尝试刷新令牌
                    refreshed = await self.refresh_access_token()
                    if refreshed:
                        # 更新headers并重试
                        headers["Authorization"] = f"Bearer {self.token}"
                        async with self.session.post(
                            f"{self.base_url}/api/chat/message",
                            json=payload,
                            headers=headers
                        ) as retry_response:
                            if retry_response.status == 200:
                                return await retry_response.json()
                            else:
                                error_text = await retry_response.text()
                                raise Exception(f"Failed to send message after retry: {error_text}")
                
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    raise Exception(f"Failed to send message: {response.status} - {error_text}")
                    
        except aiohttp.ClientError as e:
            logger.error(f"Client error: {e}")
            raise Exception(f"Network error: {str(e)}")
    
    async def stream_chat(
        self,
        message: str,
        conversation_id: Optional[str] = None,
        model: Optional[str] = None,
        on_message: Optional[Callable] = None,
        on_error: Optional[Callable] = None,
        on_complete: Optional[Callable] = None
    ):
        """流式聊天 (SSE)"""
        params = {"message": message}
        if model:
            params["model"] = model
        if conversation_id:
            params["conversation_id"] = conversation_id
        
        headers = self.get_sse_headers()
        
        try:
            async with self.session.get(
                f"{self.base_url}/api/chat/stream",
                params=params,
                headers=headers
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    if on_error:
                        on_error(f"Stream failed: {response.status} - {error_text}")
                    return
                
                # 处理SSE流
                buffer = ""
                event_type = None
                event_data = ""
                
                async for chunk in response.content:
                    if chunk:
                        buffer += chunk.decode('utf-8')
                        lines = buffer.split('\n')
                        buffer = lines[-1]
                        
                        for line in lines[:-1]:
                            line = line.strip()
                            
                            if line.startswith("event: "):
                                event_type = line[7:]
                            elif line.startswith("data: "):
                                event_data = line[6:]
                            elif line == "" and event_data:
                                # 空行表示事件结束
                                try:
                                    data = json.loads(event_data)
                                    
                                    if event_type == "message" and on_message:
                                        on_message(data)
                                    elif event_type == "error" and on_error:
                                        on_error(data.get("error", "Unknown error"))
                                    elif event_type == "done" and on_complete:
                                        on_complete(data)
                                        
                                except json.JSONDecodeError as e:
                                    logger.warning(f"Failed to parse SSE data: {event_data}, error: {e}")
                                
                                # 重置
                                event_type = None
                                event_data = ""
                                
        except asyncio.TimeoutError:
            if on_error:
                on_error("Stream timeout")
        except Exception as e:
            if on_error:
                on_error(str(e))
    
    async def stream_chat_generator(
        self,
        message: str,
        conversation_id: Optional[str] = None,
        model: Optional[str] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """流式聊天生成器"""
        params = {"message": message}
        if model:
            params["model"] = model
        if conversation_id:
            params["conversation_id"] = conversation_id
        
        headers = self.get_sse_headers()
        
        async with self.session.get(
            f"{self.base_url}/api/chat/stream",
            params=params,
            headers=headers
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                yield {"type": "error", "error": f"Stream failed: {response.status} - {error_text}"}
                return
            
            # 处理SSE流
            buffer = ""
            event_type = None
            event_data = ""
            
            async for chunk in response.content:
                if chunk:
                    buffer += chunk.decode('utf-8')
                    lines = buffer.split('\n')
                    buffer = lines[-1]
                    
                    for line in lines[:-1]:
                        line = line.strip()
                        
                        if line.startswith("event: "):
                            event_type = line[7:]
                        elif line.startswith("data: "):
                            event_data = line[6:]
                        elif line == "" and event_data:
                            try:
                                data = json.loads(event_data)
                                data["event_type"] = event_type
                                yield data
                            except json.JSONDecodeError:
                                pass
                            
                            event_type = None
                            event_data = ""
    
    # 3. 会话管理
    async def get_conversations(
        self,
        limit: int = 10,
        offset: int = 0,
        search: Optional[str] = None,
        model: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """获取会话列表"""
        params = {"limit": limit, "offset": offset}
        if search:
            params["search"] = search
        if model:
            params["model"] = model
        
        async with self.session.get(
            f"{self.base_url}/api/conversations",
            params=params,
            headers=self.get_headers()
        ) as response:
            if response.status == 200:
                return await response.json()
            else:
                error_text = await response.text()
                raise Exception(f"Failed to get conversations: {error_text}")
    
    async def get_conversation(self, conversation_id: str) -> Dict[str, Any]:
        """获取会话详情"""
        async with self.session.get(
            f"{self.base_url}/api/conversations/{conversation_id}",
            headers=self.get_headers()
        ) as response:
            if response.status == 200:
                return await response.json()
            else:
                error_text = await response.text()
                raise Exception(f"Failed to get conversation: {error_text}")
    
    async def get_conversation_messages(
        self,
        conversation_id: str,
        limit: int = 50,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """获取会话消息历史"""
        params = {"limit": limit, "offset": offset}
        
        async with self.session.get(
            f"{self.base_url}/api/conversations/{conversation_id}/messages",
            params=params,
            headers=self.get_headers()
        ) as response:
            if response.status == 200:
                return await response.json()
            else:
                error_text = await response.text()
                raise Exception(f"Failed to get messages: {error_text}")
    
    async def update_conversation_title(
        self,
        conversation_id: str,
        title: str
    ) -> Dict[str, Any]:
        """更新会话标题"""
        async with self.session.post(
            f"{self.base_url}/api/conversations/{conversation_id}/title",
            json={"title": title},
            headers=self.get_headers()
        ) as response:
            if response.status == 200:
                return await response.json()
            else:
                error_text = await response.text()
                raise Exception(f"Failed to update title: {error_text}")
    
    async def delete_conversation(self, conversation_id: str) -> Dict[str, Any]:
        """删除会话"""
        async with self.session.delete(
            f"{self.base_url}/api/conversations/{conversation_id}",
            headers=self.get_headers()
        ) as response:
            if response.status == 200:
                return await response.json()
            else:
                error_text = await response.text()
                raise Exception(f"Failed to delete conversation: {error_text}")
    
    async def export_conversation(
        self,
        conversation_id: str,
        format: str = "json"
    ) -> Dict[str, Any]:
        """导出会话"""
        params = {"format": format}
        
        async with self.session.post(
            f"{self.base_url}/api/conversations/{conversation_id}/export",
            params=params,
            headers=self.get_headers()
        ) as response:
            if response.status == 200:
                return await response.json()
            else:
                error_text = await response.text()
                raise Exception(f"Failed to export conversation: {error_text}")
    
    # 4. WebSocket相关
    async def connect_websocket(
        self,
        on_message: Optional[Callable] = None,
        on_error: Optional[Callable] = None,
        on_connect: Optional[Callable] = None,
        on_disconnect: Optional[Callable] = None
    ):
        """连接WebSocket"""
        ws_url = self.base_url.replace("http", "ws") + "/api/ws"
        if self.token:
            ws_url += f"?token={self.token}"
        
        try:
            self.ws = await websockets.connect(ws_url)
            
            if on_connect:
                await on_connect()
            
            # 监听消息
            async for message in self.ws:
                try:
                    data = json.loads(message)
                    if on_message:
                        on_message(data)
                except json.JSONDecodeError:
                    if on_error:
                        on_error(f"Invalid message format: {message}")
                except Exception as e:
                    if on_error:
                        on_error(str(e))
                        
        except websockets.exceptions.ConnectionClosed:
            if on_disconnect:
                on_disconnect()
        except Exception as e:
            if on_error:
                on_error(str(e))
        finally:
            if self.ws:
                await self.ws.close()
                self.ws = None
    
    async def send_websocket_message(
        self,
        action: str,
        data: Dict[str, Any]
    ):
        """发送WebSocket消息"""
        if not self.ws:
            raise Exception("WebSocket not connected")
        
        message = {"action": action, **data}
        await self.ws.send(json.dumps(message))
    
    # 5. 文件相关
    async def upload_file(
        self,
        file_path: str,
        conversation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """上传文件"""
        from pathlib import Path
        
        file_path = Path(file_path)
        if not file_path.exists():
            raise Exception(f"File not found: {file_path}")
        
        data = aiohttp.FormData()
        data.add_field("file",
                      open(file_path, 'rb'),
                      filename=file_path.name)
        if conversation_id:
            data.add_field("conversation_id", conversation_id)
        
        headers = {}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        
        async with self.session.post(
            f"{self.base_url}/api/files/upload",
            data=data,
            headers=headers
        ) as response:
            if response.status == 200:
                return await response.json()
            else:
                error_text = await response.text()
                raise Exception(f"Upload failed: {error_text}")
    
    # 6. 模型相关
    async def get_available_models(self) -> List[Dict[str, Any]]:
        """获取可用模型列表"""
        async with self.session.get(
            f"{self.base_url}/api/models/available",
            headers=self.get_headers()
        ) as response:
            if response.status == 200:
                return await response.json()
            else:
                error_text = await response.text()
                raise Exception(f"Failed to get models: {error_text}")
    
    # 7. 用户相关
    async def get_profile(self) -> Dict[str, Any]:
        """获取用户信息"""
        async with self.session.get(
            f"{self.base_url}/api/users/profile",
            headers=self.get_headers()
        ) as response:
            if response.status == 200:
                return await response.json()
            else:
                error_text = await response.text()
                raise Exception(f"Failed to get profile: {error_text}")
    
    async def update_preferences(self, preferences: Dict[str, Any]) -> Dict[str, Any]:
        """更新用户偏好"""
        async with self.session.put(
            f"{self.base_url}/api/users/preferences",
            json=preferences,
            headers=self.get_headers()
        ) as response:
            if response.status == 200:
                return await response.json()
            else:
                error_text = await response.text()
                raise Exception(f"Failed to update preferences: {error_text}")

# 8. 代码管理相关
    async def extract_code(
        self,
        ai_response: str,
        conversation_id: str,
        auto_save: bool = True
    ) -> Dict[str, Any]:
        """从AI响应中提取代码"""
        async with self.session.post(
            f"{self.base_url}/api/code/extract",
            json={
                "ai_response": ai_response,
                "conversation_id": conversation_id,
                "auto_save": auto_save
            },
            headers=self.get_headers()
        ) as response:
            if response.status == 200:
                return await response.json()
            else:
                error_text = await response.text()
                raise Exception(f"Failed to extract code: {error_text}")
    
    async def execute_code(
        self,
        code_id: str,
        env_vars: Optional[Dict[str, str]] = None,
        timeout: int = 30000
    ) -> Dict[str, Any]:
        """执行保存的代码"""
        payload = {"timeout": timeout}
        if env_vars:
            payload["env_vars"] = env_vars
        
        async with self.session.post(
            f"{self.base_url}/api/code/execute/{code_id}",
            json=payload,
            headers=self.get_headers()
        ) as response:
            if response.status == 200:
                return await response.json()
            else:
                error_text = await response.text()
                raise Exception(f"Failed to execute code: {error_text}")
    
    async def create_cron_job(
        self,
        code_id: str,
        cron_expression: str,
        job_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """为代码创建定时任务"""
        async with self.session.post(
            f"{self.base_url}/api/code/cron/{code_id}",
            json={
                "cron_expression": cron_expression,
                "job_name": job_name
            },
            headers=self.get_headers()
        ) as response:
            if response.status == 200:
                return await response.json()
            else:
                error_text = await response.text()
                raise Exception(f"Failed to create cron job: {error_text}")
    
    async def list_codes(
        self,
        language: Optional[str] = None,
        limit: int = 20,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """获取代码列表"""
        params = {"limit": limit, "offset": offset}
        if language:
            params["language"] = language
        
        async with self.session.get(
            f"{self.base_url}/api/code/list",
            params=params,
            headers=self.get_headers()
        ) as response:
            if response.status == 200:
                return await response.json()
            else:
                error_text = await response.text()
                raise Exception(f"Failed to get codes: {error_text}")

# 使用示例
async def main():
    """完整功能示例"""
    async with ChatBotAPI() as chatbot:
        # 1. 登录
        try:
            print("1. 登录测试")
            await chatbot.login("newuser", "newPass123")
            print("✅ 登录成功\n")
        except Exception as e:
            print(f"❌ 登录失败: {e}")
            return
        
        # 2. 发送普通消息
        try:
            print("2. 发送普通消息")
            response = await chatbot.send_message(
                "你好！今天天气怎么样？",
                model="o3-gz"
            )
            print(f"✅ 消息发送成功")
            print(f"📍 会话ID: {response['conversation_id']}")
            print(f"🤖 AI回复: {response['content'][:100]}...")
            if response.get('follow_up_questions'):
                print(f"❓ 推荐问题: {response['follow_up_questions'][:2]}")
            print()
            
            conversation_id = response['conversation_id']
        except Exception as e:
            print(f"❌ 发送失败: {e}\n")
            return
        
        # 3. 流式聊天
        try:
            print("3. 流式消息测试")
            print("🤖 AI回复: ", end='', flush=True)
            
            message_count = 0
            full_response = ""
            
            async for data in chatbot.stream_chat_generator(
                "给我讲一个关于未来城市的小故事，大约100字",
                conversation_id=conversation_id,
                model="o3-gz"
            ):
                if data.get("event_type") == "message":
                    content = data.get("content", "")
                    if content:
                        print(content, end='', flush=True)
                        full_response += content
                        message_count += 1
                elif data.get("event_type") == "done":
                    print(f"\n✅ 流式响应完成 (共 {message_count} 个数据块)\n")
                    break
                elif data.get("event_type") == "error":
                    print(f"\n❌ 错误: {data.get('error')}\n")
                    break
                    
        except Exception as e:
            print(f"\n❌ 流式聊天失败: {e}\n")
        
        # 4. 获取会话列表
        try:
            print("4. 获取会话列表")
            conversations = await chatbot.get_conversations(limit=5)
            print(f"✅ 找到 {len(conversations)} 个会话:")
            for i, conv in enumerate(conversations[:3], 1):
                created_at = conv.get('created_at', 'Unknown')
                if isinstance(created_at, str) and 'T' in created_at:
                    created_at = created_at.split('T')[0]
                print(f"   {i}. {conv.get('title', 'Untitled')[:30]}... (创建于: {created_at})")
            print()
        except Exception as e:
            print(f"❌ 获取会话列表失败: {e}\n")
        
        # 5. 获取会话历史
        if conversation_id:
            try:
                print("5. 获取会话历史")
                messages = await chatbot.get_conversation_messages(
                    conversation_id,
                    limit=10
                )
                print(f"✅ 当前会话有 {len(messages)} 条消息:")
                for msg in messages[-4:]:  # 显示最后4条
                    role = "👤 用户" if msg['role'] == "user" else "🤖 AI"
                    content = msg['content'][:50] + "..." if len(msg['content']) > 50 else msg['content']
                    print(f"   {role}: {content}")
            except Exception as e:
                print(f"❌ 获取会话历史失败: {e}")
        
        print("\n" + "="*50)
        print("✅ 测试完成！")


async def interactive_chat():
    """交互式聊天示例"""
    print("🤖 ChatBot 交互式聊天")
    print("="*50)
    print("输入 '/exit' 退出, '/new' 开始新会话, '/history' 查看历史")
    print("="*50 + "\n")
    
    async with ChatBotAPI() as chatbot:
        # 登录
        username = input("👤 用户名 (默认: newuser): ").strip() or "newuser"
        password = input("🔑 密码 (默认: newPass123): ").strip() or "newPass123"
        
        try:
            await chatbot.login(username, password)
            print("\n✅ 登录成功！开始聊天...\n")
        except Exception as e:
            print(f"❌ 登录失败: {e}")
            return
        
        conversation_id = None
        
        while True:
            try:
                # 获取用户输入
                user_input = input("\n👤 你: ").strip()
                
                if not user_input:
                    continue
                
                # 处理命令
                if user_input.lower() == '/exit':
                    print("👋 再见！")
                    break
                elif user_input.lower() == '/new':
                    conversation_id = None
                    print("🆕 开始新的会话")
                    continue
                elif user_input.lower() == '/history':
                    if conversation_id:
                        messages = await chatbot.get_conversation_messages(
                            conversation_id,
                            limit=10
                        )
                        print(f"\n📜 会话历史 ({len(messages)} 条消息):")
                        for msg in messages:
                            role = "👤" if msg['role'] == "user" else "🤖"
                            print(f"{role} {msg['content'][:100]}...")
                    else:
                        print("❌ 当前没有活动会话")
                    continue
                
                # 发送消息（使用流式）
                print("\n🤖 AI: ", end='', flush=True)
                
                response_received = False
                async for data in chatbot.stream_chat_generator(
                    user_input,
                    conversation_id=conversation_id,
                    model="o3-gz"
                ):
                    if data.get("event_type") == "message":
                        content = data.get("content", "")
                        if content:
                            print(content, end='', flush=True)
                            response_received = True
                            
                        # 获取会话ID
                        if data.get("metadata", {}).get("conversation_id"):
                            conversation_id = data["metadata"]["conversation_id"]
                            
                    elif data.get("event_type") == "done":
                        if not response_received:
                            print("(没有收到响应)")
                        break
                    elif data.get("event_type") == "error":
                        print(f"\n❌ 错误: {data.get('error')}")
                        break
                
                print()  # 换行
                
            except KeyboardInterrupt:
                print("\n\n👋 检测到中断，退出聊天...")
                break
            except Exception as e:
                print(f"\n❌ 发生错误: {e}")


async def code_management_example():
    """代码管理功能示例"""
    async with ChatBotAPI() as chatbot:
        # 1. 登录
        await chatbot.login("newuser", "newPass123")
        print("✅ 登录成功")
        
        # 2. 请求AI生成代码
        print("\n📝 请求AI生成系统监控脚本...")
        response = await chatbot.send_message(
            """
            Please write a Python script that:
            1. Monitors system CPU and memory usage
            2. Logs the data to a file
            3. Sends an alert if CPU usage exceeds 80% or memory usage exceeds 90%
            4. Runs every 5 minutes when scheduled
            """,
            model="o3-gz"
        )
        
        print(f"🤖 AI响应已收到")
        
        # 3. 检查是否有提取的代码
        if response.get("metadata", {}).get("extracted_codes"):
            codes = response["metadata"]["extracted_codes"]
            print(f"\n💾 发现 {len(codes)} 个代码块")
            
            for i, code in enumerate(codes):
                if code.get("saved") and code.get("id"):
                    print(f"\n📄 代码 {i+1}:")
                    print(f"   - 语言: {code['language']}")
                    print(f"   - ID: {code['id']}")
                    print(f"   - 描述: {code.get('description', 'N/A')}")
                    
                    # 4. 执行代码测试
                    print(f"\n🚀 执行代码...")
                    try:
                        exec_result = await chatbot.execute_code(code['id'])
                        print(f"   - 执行{'成功' if exec_result['success'] else '失败'}")
                        print(f"   - 退出码: {exec_result['exit_code']}")
                        print(f"   - 执行时间: {exec_result['execution_time']:.2f}秒")
                        
                        if exec_result['stdout']:
                            print(f"   - 输出:\n{exec_result['stdout'][:200]}...")
                        
                        # 5. 创建定时任务
                        if exec_result['success']:
                            print(f"\n⏰ 创建定时任务 (每5分钟运行)...")
                            cron_result = await chatbot.create_cron_job(
                                code_id=code['id'],
                                cron_expression="*/5 * * * *",
                                job_name=f"system_monitor_{i+1}"
                            )
                            
                            if cron_result['success']:
                                print(f"   ✅ 定时任务创建成功")
                                print(f"   - 任务名: {cron_result['job_name']}")
                                print(f"   - 下次运行: {cron_result['next_run']}")
                            else:
                                print(f"   ❌ 定时任务创建失败: {cron_result.get('error')}")
                                
                    except Exception as e:
                        print(f"   ❌ 执行失败: {e}")
        
        # 6. 列出所有保存的代码
        print("\n📋 获取代码列表...")
        codes = await chatbot.list_codes(limit=5)
        print(f"找到 {len(codes)} 个保存的代码:")
        for code in codes:
            print(f"  - {code['language']}: {code.get('description', 'No description')[:50]}...")
            print(f"    创建于: {code['created_at']}, 执行次数: {code['execution_count']}")


# 添加交互式代码管理
async def interactive_code_chat():
    """支持代码管理的交互式聊天"""
    print("🤖 ChatBot 交互式聊天 (支持代码管理)")
    print("="*50)
    print("命令: /exit 退出, /new 新会话, /history 查看历史")
    print("     /codes 查看代码, /exec <code_id> 执行代码")
    print("     /cron <code_id> <expression> 创建定时任务")
    print("="*50 + "\n")
    
    async with ChatBotAPI() as chatbot:
        # ... 登录逻辑 ...
        
        conversation_id = None
        last_code_ids = []  # 记录最近的代码ID
        
        while True:
            try:
                user_input = input("\n👤 你: ").strip()
                
                if not user_input:
                    continue
                
                # 处理命令
                if user_input.lower() == '/exit':
                    print("👋 再见！")
                    break
                    
                elif user_input.lower() == '/codes':
                    # 显示代码列表
                    codes = await chatbot.list_codes(limit=10)
                    if codes:
                        print("\n📋 保存的代码:")
                        for i, code in enumerate(codes):
                            print(f"{i+1}. [{code['language']}] {code.get('description', 'No description')[:50]}...")
                            print(f"   ID: {code['id']}, 执行次数: {code['execution_count']}")
                            last_code_ids.append(code['id'])
                    else:
                        print("没有保存的代码")
                    continue
                    
                elif user_input.startswith('/exec '):
                    # 执行代码
                    code_id = user_input[6:].strip()
                    print(f"\n🚀 执行代码 {code_id}...")
                    try:
                        result = await chatbot.execute_code(code_id)
                        print(f"执行{'成功' if result['success'] else '失败'} (退出码: {result['exit_code']})")
                        if result['stdout']:
                            print(f"输出:\n{result['stdout']}")
                        if result['stderr']:
                            print(f"错误:\n{result['stderr']}")
                    except Exception as e:
                        print(f"❌ 执行失败: {e}")
                    continue
                    
                elif user_input.startswith('/cron '):
                    # 创建定时任务
                    parts = user_input[6:].strip().split(maxsplit=1)
                    if len(parts) == 2:
                        code_id, cron_expr = parts
                        print(f"\n⏰ 创建定时任务...")
                        try:
                            result = await chatbot.create_cron_job(code_id, cron_expr)
                            if result['success']:
                                print(f"✅ 定时任务创建成功: {result['job_name']}")
                            else:
                                print(f"❌ 创建失败: {result.get('error')}")
                        except Exception as e:
                            print(f"❌ 错误: {e}")
                    else:
                        print("用法: /cron <code_id> <cron_expression>")
                    continue
                
                # 发送普通消息...
                # (保持原有的聊天逻辑)
                
            except KeyboardInterrupt:
                print("\n\n👋 检测到中断，退出聊天...")
                break
            except Exception as e:
                print(f"\n❌ 发生错误: {e}")

if __name__ == "__main__":
    # 运行完整功能测试
    # asyncio.run(main())
    
    # 或者运行交互式聊天
    asyncio.run(interactive_chat())
    # asyncio.run(code_management_example())