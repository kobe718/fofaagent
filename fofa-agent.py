import sys
import os
import base64
import uuid
import datetime
import yaml
import json
from typing import Optional, Dict, Any
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv
import litellm
import requests

try:
    from langchain_openai import ChatOpenAI
except ImportError:
    ChatOpenAI = None

# 确保中文显示正常
yaml.Dumper.ignore_aliases = lambda *args: True

# 模块导出标记
__all__ = [
    'fofa_search', 
    'FofaAPI', 
    'create_react_agent_for_fofa',
    'format_results_to_yaml'
]

def parse_json_response(content: str, search_request: str) -> Dict[str, Any]:
    """
    解析JSON响应，处理可能的格式问题并提取结果，支持复杂的错误恢复策略
    
    Args:
        content: 原始响应内容
        search_request: 原始搜索请求
        
    Returns:
        解析后的字典结果
    """
    try:
        if not content or not isinstance(content, str):
            return {"success": False, "error": "响应内容为空或格式不正确"}
            
        # 清理内容，尝试找到有效的JSON部分
        if not content.startswith('{'):
            start_pos = content.find('{')
            if start_pos != -1:
                content = content[start_pos:]
        
        if not content.endswith('}'):
            # 尝试找到JSON的结束位置
            end_pos = content.rfind('}')
            if end_pos != -1:
                content = content[:end_pos+1]
        
        # 规范化JSON格式
        content = content.replace("'", '"')
        content = content.replace('True', 'true').replace('False', 'false').replace('None', 'null')
        
        # 尝试直接解析
        return json.loads(content)
    except json.JSONDecodeError as json_error:
        print(f"JSON解析错误: {str(json_error)}")
        # 尝试使用正则表达式修复常见的JSON格式问题
        try:
            import re
            # 移除行首的空格和制表符
            content = re.sub(r'^\s+', '', content, flags=re.MULTILINE)
            # 修复可能的不规范的JSON格式
            # 1. 确保键值对中的键用双引号包裹
            content = re.sub(r'(\w+):', r'"\1":', content)
            # 2. 确保字符串值用双引号包裹，但保留已经是数字/布尔值/null的值
            content = re.sub(r':\s*([^"\d\[\]{}:,\s][^:\[\]{}:,]*?)([\],})])', r':"\1"\2', content)
            # 3. 移除多余的逗号
            content = re.sub(r',\s*([}\]])', r'\1', content)
            # 尝试再次解析
            return json.loads(content)
        except Exception as repair_error:
            # 如果修复失败，尝试提取关键信息
            try:
                # 提取关键信息
                total_match = re.search(r'total:?\s*(\d+)', content)
                total = int(total_match.group(1)) if total_match else 0
                results = []
                # 返回基本的搜索结果结构
                return {"success": True, "query": search_request, "total": total, "results": results, "size": len(results), "fields": "host,ip,port,title,protocol,banner,app"}
            except:
                # 最后的备选方案，返回空结果
                return {"success": False, "error": f"无法解析JSON响应: {str(json_error)}", "query": search_request}
    except Exception as e:
        print(f"解析过程中发生错误: {str(e)}")
        return {"success": False, "error": f"解析错误: {str(e)}", "query": search_request}

# 加载.env文件中的环境变量
load_dotenv()

# 从.env读取MAX_RESULT配置，默认200，最大不超过1000
MAX_RESULT = int(os.environ.get("FOFA_MAX_RESULT", "200"))
MAX_RESULT = min(MAX_RESULT, 1000)

def generate_scroll_id() -> str:
    """生成一个8位的随机字符串作为scrollid"""
    # 使用uuid生成随机字符串并截取前8位
    return uuid.uuid4().hex[:8]

class FofaAPI:
    """Fofa API客户端，用于封装与Fofa API的交互"""
    def __init__(self, email: Optional[str] = None, api_key: Optional[str] = None):
        # 优先使用传入的参数，如果没有则从环境变量中获取
        self.email = email or os.environ.get("FOFA_EMAIL")
        self.api_key = api_key or os.environ.get("FOFA_API_KEY")
        self.base_url = "https://fofa.info/api/v1/search/all"
        
        # 验证必要的认证信息
        if not self.email or not self.api_key:
            raise ValueError("未配置Fofa API认证信息，请设置FOFA_EMAIL和FOFA_API_KEY环境变量")
            
    def search(self, query_str: str, size: int = 10, fields: str = "host,ip,port,title", scroll: Optional[str] = None) -> Dict[str, Any]:
        # 限制单次搜索结果不超过MAX_RESULT个
        if size > MAX_RESULT:
            size = MAX_RESULT
        # 限制单次返回结果数量为20个
        page_size = min(size, 20)
        """
        执行Fofa搜索
        
        Args:
            query_str: 查询字符串，遵循Fofa搜索语法
            size: 返回结果的数量，默认为10
            fields: 返回的字段，默认为"host,ip,port,title"
            scroll: scroll_id，用于分页查询
        
        Returns:
            包含搜索结果的字典
        """
        # 对查询字符串进行Base64编码
        qbase64 = base64.b64encode(query_str.encode('utf-8')).decode('utf-8')
        
        # 构建请求参数
        params = {
            'email': self.email,
            'key': self.api_key,
            'qbase64': qbase64,
            'size': size,
            'fields': fields
        }
        
        # 如果提供了scroll_id，则添加到参数中
        if scroll:
            params['scroll'] = scroll
        
        try:
            # 发送请求，禁用SSL验证以解决证书问题
            response = requests.get(self.base_url, params=params, verify=True)
            
            # 检查响应状态
            if response.status_code == 200:
                data = response.json()
                
                # 检查API返回的状态
                if data.get('error') == False:
                    return data
                else:
                    raise Exception(f"Fofa API错误: {data.get('errmsg')}")
            else:
                raise Exception(f"HTTP请求错误: 状态码 {response.status_code}")
        except Exception as e:
            raise Exception(f"搜索过程中发生错误: {str(e)}")

@tool
def fofa_search(
    query: str,
    size: int = 10,
    fields: str = "host,ip,port,title,protocol,banner,product",
    scroll: Optional[str] = None
) -> dict:
    """
    使用Fofa API搜索网络资产
    
    Args:
        query: 搜索查询条件，遵循Fofa搜索语法，例如：domain="example.com"、title="登录"
        size: 最大结果数量，默认10
        fields: 返回的字段，默认包含host,ip,port,title,protocol,banner,product
        
    Returns:
        包含搜索结果的字典
    """
    try:
        # 创建FofaAPI实例
        fofa = FofaAPI()
        
        # 执行搜索
        result = fofa.search(query_str=query, size=size, fields=fields, scroll=scroll)
        
        # 返回原始结果数据，让外部处理格式化
        return {
            "success": True,
            "query": query,
            "total": result.get('total', 0),
            "size": result.get('size', 0),
            "results": result.get('results', []),
            "fields": fields,
            "scroll_id": result.get('scroll_id')
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

# React Agent的提示词，专注于资产搜索任务
REACT_AGENT_PROMPT = """你是一个专业的网络资产搜索助手，专注于使用Fofa API进行网络资产查询。

## 工作流程
你必须严格按照以下模式工作：

1. **推理（Reasoning）**: 
   - 分析用户的网络资产搜索请求，理解需要查找的目标
   - 思考如何将自然语言请求转换为正确的Fofa搜索语法
   - 确定合适的搜索参数（如结果数量）

2. **行动（Action）**: 
   - 唯一可用的工具是fofa_search，必须使用它来执行搜索
   - 确保搜索语法符合Fofa要求
   - 不需要指定fields参数，使用默认值即可

3. **反思与重试（Reflection & Retry）**: 
   - 评估搜索结果是否成功：
     * 如返回的结果中success为false，**不输出任何内容**，立即反思并优化搜索策略
     * 如返回有效结果（success为true），继续处理
   - 持续优化：如搜索结果不满足用户需求或匹配数量过少，调整搜索条件重新搜索
   - 停止条件：当搜索结果满足用户需求（如达到指定结果数量）或连续3次搜索仍无法获得有效结果时，停止搜索
   - 最终输出：仅在成功搜索到有效结果后，**原封不动**地返回fofa_search的结果（不要添加任何额外解释或格式化）

## 可用工具

### fofa_search
使用Fofa API搜索网络资产信息。必须提供搜索查询条件，遵循Fofa搜索语法。
- 默认搜索200个结果，单次最多能返回2000个结果
- 默认搜索host,ip,port,title,protocol,banner,product字段

## 搜索语法示例
以下是一些常见的Fofa搜索语法示例，你可以根据用户需求进行调整和组合：
- 搜索特定域名: domain="example.com"
- 搜索特定IP地址: ip="1.1.1.1"
- 搜索特定端口: port="8080"
- 搜索特定标题: title="登录页面"
- 组合搜索: domain="example.com" && port="443"
- 搜索特定服务: service="nginx"
- 搜索特定国家: country="CN"

## 工作原则

1. **专注资产搜索**: 只处理与网络资产搜索相关的任务，拒绝处理其他无关任务
2. **精确转换**: 准确将自然语言转换为Fofa搜索语法
3. **结果清晰**: 以结构化方式呈现搜索结果
4. **用户至上**: 确保搜索结果满足用户需求
5. **结果管理**: 一次整理所有的结果并输出，不要添加任何影响后续json解析的内容

## 重要规则
1. **静默重试**: 搜索失败或结果无效时，**必须完全静默处理**，不向用户输出任何中间过程信息
2. **结果导向**: 只在成功完成搜索任务并获得有效结果后，才输出结构化的最终结果
3. **策略调整**: 如遇到搜索语法错误，尝试简化或调整搜索表达式；**绝对不要**尝试扩大搜索范围或使用不同关键词
4. **内容聚焦**: 最终输出必须只包含与用户请求直接相关的有效搜索结果，避免任何无关信息

请开始处理用户的网络资产搜索请求。"""

def _coalesce(*values, default=None):
    """获取第一个非None且非空的值"""
    for v in values:
        if v is not None and v != "":
            return v
    return default

def create_openai_model():
    """创建OpenAI模型实例"""
    if ChatOpenAI is None:
        raise ImportError("未安装 langchain-openai，请先安装: pip install langchain-openai")

    model_name = _coalesce(
        os.environ.get("OPENAI_MODEL"),
        default="gpt-4o-mini",
    )
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("缺少 OPENAI_API_KEY，请设置环境变量")

    base_url = os.environ.get("OPENAI_BASE_URL")
    temperature = float(_coalesce(os.environ.get("OPENAI_TEMPERATURE"), 0.7))

    params = {
        "model": model_name,
        "api_key": api_key,
        "temperature": temperature,
    }
    if base_url:
        params["base_url"] = base_url

    return ChatOpenAI(**params)

def create_react_agent_for_fofa():
    """创建配置好Fofa搜索工具的React Agent"""
    # 创建模型
    model = create_openai_model()
    
    # 配置litellm
    litellm.drop_params = True
    
    # 工具列表
    tools = [fofa_search]
    
    # 创建React Agent
    agent = create_react_agent(
        model=model,
        tools=tools,
        prompt=REACT_AGENT_PROMPT
    )
    
    return agent

def ensure_tmp_directory():
    """确保.tmp目录存在"""
    tmp_dir = os.path.join(os.getcwd(), '.tmp')
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    return tmp_dir

def load_search_result_from_file(scroll_id: str) -> dict:
    """从.tmp目录加载之前保存的搜索结果"""
    tmp_dir = ensure_tmp_directory()
    file_path = os.path.join(tmp_dir, f"{scroll_id}.json")
    
    if not os.path.exists(file_path):
        return None
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"错误: 无法解析scrollid为'{scroll_id}'的搜索结果文件")
        return None
    except Exception as e:
        print(f"错误: 读取scrollid为'{scroll_id}'的搜索结果时出错: {str(e)}")
        return None


def save_search_result_to_file(scroll_id, search_data):
    """将搜索结果保存到.tmp目录下，使用scrollid作为文件名"""
    tmp_dir = ensure_tmp_directory()
    file_path = os.path.join(tmp_dir, f"{scroll_id}.json")
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(search_data, f, ensure_ascii=False, indent=2)
    return file_path

def format_results_to_yaml(search_results, user_query, cumulative_assets=None, current_page=1, total_pages=1):
    """
    将搜索结果格式化为YAML格式
    
    Args:
        search_results: fofa_search返回的结果字典
        user_query: 用户的原始搜索请求
        cumulative_assets: 累计的资产列表
        current_page: 当前页码
        total_pages: 总页数
        
    Returns:
        YAML格式的结果字符串
    """
    try:
        if not search_results or not search_results.get('success'):
            return ""
        
        # 获取当前时间戳
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # 获取搜索条件
        search_query = search_results.get('query', '')
        
        # 获取匹配总数
        total = search_results.get('total', 0)
        if total == 0 and search_results.get('size', 0) > 0:
            total = search_results.get('size', 0)
        
        # 获取资产列表
        assets = cumulative_assets or []
        if not assets:
            # 如果没有累计资产，则从搜索结果中获取
            fields = search_results.get('fields', 'host,ip,port,title,protocol,banner,product').split(',')
            results = search_results.get('results', [])
            
            # 计算当前页资产的起始序号
            start_idx = 1
            
            for idx, item in enumerate(results, start_idx):
                asset = {
                    'id': idx,
                    'host': '',
                    'ip': '',
                    'port': '',
                    'protocol': '',
                    'banner': '',
                    'title': '',
                    'product': ''
                }
                
                # 填充资产信息
                for i, field in enumerate(fields):
                    if i < len(item) and item[i]:
                        if field.lower() in asset:
                            asset[field.lower()] = item[i]
                
                assets.append(asset)
        
        # 获取scroll_id
        scroll_id = search_results.get('scroll_id', '')
        
        # 确保current_page和total_pages是整数
        current_page = int(current_page)
        total_pages = int(total_pages)
        
        # 构建YAML数据结构
        yaml_data = {
            'timestamp': timestamp,
            'query': user_query,
            'search': search_query,
            'total': total,
            'current_page': current_page,
            'total_pages': total_pages,
            'assets': assets,
            'scrollid': scroll_id
        }
        
        # 转换为YAML格式，处理可能的中文和特殊字符
        yaml_output = yaml.dump(
            yaml_data,
            allow_unicode=True,
            default_flow_style=False,
            sort_keys=False,
            indent=2
        )
        return yaml_output
    except Exception as e:
        # 如果格式化过程出错，返回错误信息的YAML格式
        error_data = {
            'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'query': user_query,
            'error': f'YAML格式化错误: {str(type(e))} - {str(e)}'
        }
        return yaml.dump(error_data, allow_unicode=True)


def main():
    """主程序入口，接受搜索请求或scrollid参数"""
    # 检查参数
    if len(sys.argv) != 2:
        print("用法: python fofa-agent.py '查找域名example.com的所有资产' 或 python fofa-agent.py 'scrollid: xxxxxxx'")
        print(f"说明: 程序支持自动分页获取最多{MAX_RESULT}个资产，每次返回20条")
        sys.exit(1)
    
    search_request = sys.argv[1]
    
    # 检查是否是scrollid格式的请求
    if search_request.lower().startswith('scrollid: '):
        # 提取scrollid
        scroll_id = search_request[10:].strip()
        
        # 检查scrollid格式（8位字母数字）
        if not (len(scroll_id) == 8 and all(c.isalnum() for c in scroll_id)):
            print("错误: scrollid必须是8位字母和数字的组合")
            sys.exit(1)
        
        # 尝试加载之前保存的搜索结果
        saved_data = load_search_result_from_file(scroll_id)
        if not saved_data:
            print(f"错误: 找不到scrollid为'{scroll_id}'的搜索结果")
            sys.exit(1)
        
        try:
            # 获取所有保存的结果和相关信息
            all_results = saved_data.get('results', [])
            total_assets = max(saved_data.get('total', 0), len(all_results))  # 确保total_assets不为0
            search_query = saved_data.get('query', '')
            next_page_index = saved_data.get('next_page_index', 0)
            total_fetched = len(all_results)
            
            # 计算总页数
            total_pages = (len(all_results) + 19) // 20  # 向上取整
            
            # 初始化new_next_page_index变量
            new_next_page_index = next_page_index
            
            # 初始化current_page_results
            current_page_results = []
            
            # 获取当前页要显示的结果（最多20条）
            if len(all_results) > 0:
                # 从保存的所有结果中获取当前页结果
                start_index = (next_page_index // 20) * 20
                current_page_results = all_results[start_index:start_index+20]
                
                # 更新下一页索引为下一页的起始位置
                new_next_page_index = start_index + 20
                saved_data['next_page_index'] = new_next_page_index
                save_search_result_to_file(scroll_id, saved_data)
            else:
                new_next_page_index = next_page_index
            
            # 准备字段列表
            fields = "host,ip,port,title,protocol,banner,product".split(',')
            
            # 将current_page_results转换为正确的资产列表格式
            formatted_assets = []
            start_idx = (next_page_index // 20) * 20 + 1  # 计算当前页资产的起始序号
            
            for idx, item in enumerate(current_page_results, start_idx):
                asset = {
                    'id': idx,
                    'host': '',
                    'ip': '',
                    'port': '',
                    'protocol': '',
                    'banner': '',
                    'title': '',
                    'product': ''
                }
                
                # 填充资产信息
                for i, field in enumerate(fields):
                    if i < len(item) and item[i]:
                        if field.lower() in asset:
                            asset[field.lower()] = item[i]
                
                formatted_assets.append(asset)
            
            # 格式化结果
            result_dict = {
                "success": True,
                "query": search_query,
                "total": total_assets,
                "size": len(current_page_results),
                "results": current_page_results,
                "fields": "host,ip,port,title,protocol,banner,product",
                "scroll_id": scroll_id if new_next_page_index < len(all_results) else None
            }
            
            # 计算当前页码
            if next_page_index < len(all_results):
                current_page = (next_page_index // 20) + 1
            elif len(all_results) > 0:
                current_page = total_pages
            else:
                current_page = 1
            
            yaml_output = format_results_to_yaml(
                result_dict, 
                search_query,
                cumulative_assets=formatted_assets,
                current_page=current_page,
                total_pages=total_pages
            )
            print(yaml_output)
            
            # 如果没有更多结果，提示用户
            if next_page_index >= len(all_results):
                print("提示: 已获取全部结果")
            else:
                # 计算剩余结果数
                remaining_results = len(all_results) - (next_page_index + 20)
                if remaining_results > 0:
                    print(f"提示: 还剩{remaining_results}条结果，继续使用相同的scrollid查看下一页")
                else:
                    print("提示: 下一页是最后一页结果")
            return
        except Exception as e:
            error_data = {
                'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'query': search_request,
                'error': str(e)
            }
            print(yaml.dump(error_data, allow_unicode=True))
            sys.exit(1)
    
    # 非scrollid请求，使用agent搜索处理所有搜索请求
    try:
        # 创建agent
        agent = create_react_agent_for_fofa()
        
        # 生成会话ID
        session_id = f"fofa-agent-{uuid.uuid4().hex[:8]}"
        # 配置递归限制和线程ID
        config = {
            "configurable": {"thread_id": session_id},
            "recursion_limit": 50  # 增加递归限制到50
        }
        
        # 添加系统提示，确保agent了解新功能
        system_message = "你是fofa网络资产搜索助手，负责把用户的搜索需求转换成fofa的搜索语句，提交给fofa_search工具执行，fofa_search工具会返回一个字典结果，你需要原封不动地返回这个结果，不要添加任何额外解释或格式化。"
        
        # 收集工具调用结果
        search_results = None
        cumulative_assets = []
        total_fetched = 0
        current_page = 1
        
        # 执行agent获取第一页结果
        for chunk in agent.stream(
            {"messages": [("system", system_message), ("user", search_request)]},
            config=config
        ):
            if "tools" in chunk:
                for tool_call in chunk["tools"]["messages"]:
                    try:
                        content = tool_call.content
                        if content and isinstance(content, str):
                            # 使用新的parse_json_response函数简化JSON解析
                            search_results = parse_json_response(content, search_request)
                            break  # 获取到结果后立即跳出循环
                    except Exception as e:
                        print(f"解析结果错误: {str(e)}")
                        search_results = {"success": False, "error": f"解析结果错误: {str(e)}"}
        
        # 如果搜索成功，尝试自动分页获取更多结果（最多200个）
        if search_results and search_results.get('success'):
            # 计算需要的总页数
            total = search_results.get('total', 0)
            # 确定实际要获取的结果总数（不超过MAX_RESULT）
            actual_max_results = min(total, MAX_RESULT)
            # 计算总页数
            total_pages = (actual_max_results + 19) // 20  # 向上取整
            
            # 如果total为0但有结果，重新计算
            if total == 0 and len(search_results.get('results', [])) > 0:
                actual_max_results = min(len(search_results.get('results', [])), MAX_RESULT)
                total_pages = (actual_max_results + 19) // 20
            
            # 如果需要分页
            if actual_max_results > 20 and current_page < total_pages:
                try:
                    fofa = FofaAPI()
                    current_scroll_id = search_results.get('scroll_id')
                    
                    # 保存第一页的结果
                    first_page_results = search_results.get('results', [])
                    cumulative_assets.extend(first_page_results)
                    total_fetched = len(cumulative_assets)
                    current_page = 2
                    
                    # 循环获取剩余页面
                    while current_scroll_id and total_fetched < actual_max_results and current_page <= total_pages:
                        # 获取下一页结果
                        next_page_data = fofa.search(
                            query_str=search_results.get('query', ''),
                            size=20,  # 每页20条
                            fields=search_results.get('fields', 'host,ip,port,title,protocol,banner,product'),
                            scroll=current_scroll_id
                        )
                        
                        # 添加新结果
                        next_results = next_page_data.get('results', [])
                        cumulative_assets.extend(next_results)
                        total_fetched = len(cumulative_assets)
                        
                        # 更新scroll_id
                        current_scroll_id = next_page_data.get('scroll_id')
                        current_page += 1
                        
                        # 如果没有更多结果或已达到最大数量，停止
                        if not next_results or total_fetched >= actual_max_results:
                            break
                    
                    # 更新搜索结果
                    search_results['results'] = cumulative_assets
                    search_results['size'] = len(cumulative_assets)
                except Exception as e:
                    print(f"自动分页过程中发生错误: {str(e)}")
                    # 继续使用已获取的结果
        
        # 格式化并输出YAML结果
        if search_results and search_results.get('success'):
            # 计算实际的页数
            total_assets = len(search_results.get('results', []))
            actual_total_pages = (total_assets + 19) // 20
            
            # 生成随机的scrollid
            scroll_id = generate_scroll_id()
            
            # 保存所有搜索结果到文件
            saved_data = {
                'query': search_results.get('query', ''),
                'total': search_results.get('total', 0),
                'scroll_id': scroll_id,  # 使用我们生成的随机ID
                'next_page_index': 0,  # 下一页的起始索引，从0开始
                'results': search_results.get('results', []),  # 保存所有结果
                'total_fetched': total_assets
            }
            file_path = save_search_result_to_file(scroll_id, saved_data)
            
            # 更新搜索结果中的scroll_id为我们生成的随机ID
            search_results['scroll_id'] = scroll_id
            
            yaml_output = format_results_to_yaml(
                search_results, 
                search_request,
                cumulative_assets=search_results.get('results', [])[:20],  # 只显示前20条
                current_page=1,
                total_pages=actual_total_pages
            )
            if yaml_output:
                print(yaml_output)
                print(f"提示: 搜索结果已全部保存到 {file_path}")
                print(f"提示: 使用 'python fofa-agent.py scrollid: {scroll_id}' 继续查看下一页")
        else:
            # 输出错误信息的YAML格式
            error_data = {
                'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'query': search_request,
                'error': search_results.get('error', '搜索失败') if search_results else '未获取到搜索结果'
            }
            print(yaml.dump(error_data, allow_unicode=True))
                     
    except Exception as e:
        # 输出异常的YAML格式
        error_data = {
            'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'query': search_request,
            'error': str(e)
        }
        print(yaml.dump(error_data, allow_unicode=True))
        sys.exit(1)

if __name__ == "__main__":
    main()