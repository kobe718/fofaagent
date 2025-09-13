import sys
import os
import datetime
import yaml
import uuid
import json
import logging
from typing import Optional, Dict, Any
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv
from langsmith import traceable

# 导入Fofa客户端功能
from fofaclient import (
    FofaAPI,
    load_search_result_from_file,
    save_search_result_to_file,
    MAX_RESULT
)


try:
    from langchain_openai import ChatOpenAI
except ImportError:
    ChatOpenAI = None

# 确保中文显示正常
yaml.Dumper.ignore_aliases = lambda *args: True

# 配置日志系统
def setup_logger():
    logger = logging.getLogger('fofa-agent')
    
    # 检查环境变量DEBUG，设置日志级别
    if os.environ.get('DEBUG', 'false').lower() == 'true':
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    
    # 确保logger没有重复的处理器
    if not logger.handlers:
        # 创建控制台处理器
        console_handler = logging.StreamHandler()
        
        # 设置日志格式
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        
        # 添加处理器到logger
        logger.addHandler(console_handler)
    
    return logger

# 初始化logger
logger = setup_logger()

# 模块导出标记
__all__ = [
    'fofa_agentic_search'
]

def generate_scroll_id() -> str:
    """生成一个8位的随机字符串作为scrollid"""
    # 使用uuid生成随机字符串并截取前8位
    return uuid.uuid4().hex[:8]

def parse_json_response(content: str, search_request: str, test_error: bool = False) -> Dict[str, Any]:
    """
    解析JSON响应，处理可能的格式问题并提取结果，支持复杂的错误恢复策略
    
    Args:
        content: 原始响应内容
        search_request: 原始搜索请求
        test_error: 是否启用测试模式（故意制造JSON解析错误）
        
    Returns:
        解析后的字典结果
    """
    try:
        # 测试模式：故意生成格式错误的JSON
        if test_error and content and isinstance(content, str):
            logger.debug("[测试模式] 启用JSON解析错误测试")
            # 制造一个故意的JSON格式错误 - 移除逗号
            if '}, {' in content:
                content = content.replace('}, {', '}{')
            logger.debug(f"[测试模式] 已修改JSON内容，尝试触发解析错误")
        
        if not content or not isinstance(content, str):
            logger.warning(f"[解析警告] 响应内容为空或格式不正确，类型: {type(content)}")
            return {"success": False, "error": "响应内容为空或格式不正确"}
            
        # 记录原始内容长度和前100个字符，便于调试
        content_preview = content[:100] + ('...' if len(content) > 100 else '')
        logger.debug(f"[解析信息] 开始解析响应 (长度: {len(content)}, 前100字符: {content_preview})")
            
        # 清理内容，尝试找到有效的JSON部分
        # 处理开头格式问题
        if not content.startswith('{'):
            start_pos = content.find('{')
            if start_pos != -1:
                logger.debug(f"[解析调整] 非JSON开头，从位置{start_pos}开始截取")
                content = content[start_pos:]
        
        # 处理结尾格式问题
        # 移除末尾可能的右括号')'，这是Fofa API常见的格式问题
        if content.endswith('})'):
            logger.debug("[解析调整] 检测到JSON末尾包含多余的')'，已移除")
            content = content[:-1]  # 移除末尾的')'
        elif not content.endswith('}'):
            # 尝试找到JSON的结束位置
            end_pos = content.rfind('}')
            if end_pos != -1:
                logger.debug(f"[解析调整] 非JSON结尾，截取到位置{end_pos+1}")
                content = content[:end_pos+1]
        
        # 规范化JSON格式
        original_length = len(content)
        content = content.replace("'", '"')
        content = content.replace('True', 'true').replace('False', 'false').replace('None', 'null')
        logger.debug(f"[解析调整] 规范化JSON格式 (长度从{original_length}变为{len(content)})")
        
        # 尝试直接解析
        result = json.loads(content)
        logger.debug("[解析成功] 成功解析JSON响应")
        return result
    except json.JSONDecodeError as json_error:
        # 获取错误位置的上下文
        error_pos = json_error.pos
        context_start = max(0, error_pos - 20)
        context_end = min(len(content), error_pos + 20)
        error_context = content[context_start:context_end]
        
        logger.error(f"[解析错误] JSON解析失败: {str(json_error)}")
        logger.debug(f"[错误位置] 位置: {error_pos}, 上下文: '{error_context}'")
        logger.debug(f"[原始请求] {search_request}")
        
        # 尝试使用正则表达式修复常见的JSON格式问题
        try:
            logger.debug("[解析尝试] 尝试使用正则表达式修复JSON格式")
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
            
            # 记录修复后的内容预览
            fixed_preview = content[:100] + ('...' if len(content) > 100 else '')
            logger.debug(f"[解析尝试] 修复后内容 (前100字符): {fixed_preview}")
            
            # 尝试再次解析
            result = json.loads(content)
            logger.debug("[解析成功] 修复后JSON解析成功")
            return result
        except Exception as repair_error:
            logger.error(f"[解析失败] JSON修复失败: {str(repair_error)}")
            # 如果修复失败，尝试提取关键信息
            try:
                logger.debug("[解析尝试] 尝试提取关键信息")
                # 提取关键信息
                import re
                total_match = re.search(r'total:?\s*(\d+)', content)
                total = int(total_match.group(1)) if total_match else 0
                results = []
                # 返回基本的搜索结果结构
                logger.debug(f"[解析尝试] 成功提取部分信息，total={total}")
                return {"success": True, "query": search_request, "total": total, "results": results, "size": len(results), "fields": "host,ip,port,title,protocol,banner,app"}
            except:
                logger.error("[解析失败] 无法提取任何有效信息")
                # 最后的备选方案，返回空结果
                return {"success": False, "error": f"无法解析JSON响应: {str(json_error)}", "query": search_request}
    except Exception as e:
        logger.error(f"[解析异常] 解析过程中发生未知错误: {str(type(e))} - {str(e)}")
        return {"success": False, "error": f"解析错误: {str(e)}", "query": search_request}

# 加载.env文件中的环境变量
load_dotenv()

LANGSMITH_API_KEY = os.environ.get("LANGSMITH_API_KEY")

if LANGSMITH_API_KEY:
    try:
        # 设置全局Langsmith跟踪配置
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_ENDPOINT"] = os.environ.get("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
        os.environ["LANGCHAIN_PROJECT"] = os.environ.get("LANGCHAIN_PROJECT", "fofa-agent")
        
        logger.info("Langsmith跟踪已启用")
    except Exception as e:
        logger.error(f"配置Langsmith跟踪时出错: {str(e)}")
        # 即使配置失败，也继续执行程序



@tool
@traceable
def fofa_search(
    query: str,
    scrollid: str,
    size: int = 10,
    fields: str = "host,ip,port,title,protocol,banner,product,country_name,region,city",
    ) -> dict:
    """
    使用Fofa API搜索网络资产
    
    Args:
        query: 搜索查询条件，遵循Fofa搜索语法，例如：domain="example.com"、title="登录"
        scrollid: 搜索id，8位随机字母数字
        size: 最大结果数量，默认10
        fields: 返回的字段，默认包含host,ip,port,title,protocol,banner,product
        
    Returns:
        包含搜索结果的字典（成功时返回除完整results外的数据和3个样例）
    """
    try:
        # 创建FofaAPI实例
        fofa = FofaAPI()
        
        # 执行搜索
        result = fofa.search(query_str=query, size=size, fields=fields)
        
        # 获取完整结果数据
        full_results = result.get('results', [])
        total = result.get('total', 0)
        
        # 生成scrollid并保存原始结果到文件
        scroll_id = scrollid
        saved_data = {
            'query': query,
            'total': total,
            'scroll_id': scroll_id,
            'results': full_results,
            'fields': fields
        }
        save_search_result_to_file(scroll_id, saved_data)
        
        # 返回除完整results外的数据和3个样例给模型，让模型只判断搜索是否正确
        return {
            "success": True,
            "query": query,
            "total": total,
            "size": len(full_results),
            "results_sample": full_results[:3],  # 只返回前3个结果作为样例
            "fields": fields,
            "scroll_id": scroll_id  # 返回生成的scrollid，以便后续可以从文件中读取完整结果
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

# React Agent的提示词，专注于资产搜索任务
REACT_AGENT_PROMPT = """你是一个专业的网络资产搜索助手，专注于使用Fofa API进行网络资产查询。你具备智能搜索策略优化能力，可以通过多次尝试不同的查询组合找到最佳解决方案。

## 工作流程
你必须严格按照以下模式工作：

1. **推理（Reasoning）**: 
   - 分析用户的网络资产搜索请求，理解需要查找的目标
   - 思考如何将自然语言请求转换为正确的Fofa搜索语法
   - 确定合适的搜索参数（如结果数量）

2. **行动与优化（Action & Optimization）**: 
   - 唯一可用的工具是fofa_search，必须使用它来执行搜索
   - 首先使用最精确的搜索条件尝试
   - 记录每次搜索的结果质量（包括相关性、数量、精确性等）
   - 基于结果分析，智能调整搜索策略：
     * 如果结果过多（>2000条），尝试增加筛选条件或使用更精确的关键词
     * 如果结果过少（<10条），尝试减少限制条件或使用更通用的关键词
     * 如果结果不相关，尝试替换同义词或调整搜索字段
     * 在探索最优搜索条件的过程中，可以把size设置的小一些，减少api的消耗，关注结果中的size总量，找到总量最多的查询组合

3. **策略比较与选择（Comparison & Selection）**: 
   - 在多次尝试后，根据以下标准选择最佳查询组合：
     * 相关性：结果是否与用户需求高度匹配
     * 数量：既不过多也不过少（理想范围10-2000条）
     * 精确度：是否准确反映用户指定的搜索条件
   - 对于复杂查询，可以尝试多种组合方式并比较结果
   - 停止条件：当找到满意的查询组合或已尝试5种不同策略时，停止搜索

4. **最终输出（Final Output）**: 
   - 仅在找到最佳查询组合并获得有效结果后，**原封不动**地返回该查询的fofa_search结果
   - 我会记录每次工具调用的结果，并以最后一次工具调用的结果作为最终输出，务必确保最后以最优的搜索条件进行一次搜索

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
5. **结果管理**: 对最终的所使用的搜索条件进行总结和说明

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

    model_name = os.environ.get("OPENAI_MODEL")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("缺少 OPENAI_API_KEY，请设置环境变量")

    base_url = os.environ.get("OPENAI_BASE_URL")

    params = {
        "model": model_name,
        "api_key": api_key,
        "temperature": 0.7,
    }
    if base_url:
        params["base_url"] = base_url

    return ChatOpenAI(**params)

@traceable
def create_react_agent_for_fofa():
    """创建配置好Fofa搜索工具的React Agent"""
    # 创建模型
    model = create_openai_model()
    
    # 工具列表
    tools = [fofa_search]
    
    # 创建React Agent
    agent = create_react_agent(
        model=model,
        tools=tools,
        prompt=REACT_AGENT_PROMPT
    )
    
    return agent



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
        if not search_results:
            return ""
        
        # 获取当前时间戳
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # 获取搜索条件
        search_query = search_results.get('query', '')
        
        # 获取匹配总数
        total = search_results.get('total', 0)
        if total == 0 and search_results.get('size', 0) > 0:
            total = search_results.get('size', 0)
        
        # 确保current_page和total_pages是整数
        current_page = int(current_page)
        total_pages = int(total_pages)
        
        # 获取scroll_id - 这个会从main函数中传递过来
        scroll_id = search_results.get('scroll_id', '')
        
        # 构建Python字典，然后使用yaml.dump()进行格式化
        yaml_data = {
            "timestamp": timestamp,
            "query": user_query,
            "search": search_query,
            "total": total,
            "current_page": current_page,
            "total_pages": total_pages
        }
        
        # 添加assets部分
        if cumulative_assets and isinstance(cumulative_assets, list) and len(cumulative_assets) > 0:
            # 如果提供了cumulative_assets参数，优先使用它来格式化结果
            yaml_data["assets"] = cumulative_assets
        else:
            # 从main函数传递的search_results中获取results数据
            results = search_results.get('results', [])
            if results:
                # 处理assets数据
                i=0
                formatted_assets = []
                for asset in results:
                    i += 1
                    formatted_asset = {
                        "id": i,
                        "host": asset[0],
                        "ip": asset[1],
                        "port": asset[2],
                        "title": asset[3],
                        "protocol": asset[4],
                        "banner": asset[5],
                        "product": asset[6],
                        "location": f"{asset[7]},{asset[8]},{asset[9]}"
                    }
                    formatted_assets.append(formatted_asset)
                yaml_data["assets"] = formatted_assets
            else:
                yaml_data["assets"] = []
        
        # 添加scrollid字段
        if scroll_id:
            yaml_data["scrollid"] = scroll_id
        
        # 使用yaml.dump()将Python字典转换为YAML格式
        # default_flow_style=False 确保使用块样式而不是流样式
        # allow_unicode=True 确保中文正常显示
        yaml_output = yaml.dump(
            yaml_data,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False  # 保持键的顺序
        )
        
        return yaml_output
    except Exception as e:
        # 如果格式化过程出错，返回错误信息的YAML格式
        error_data = {
            "timestamp": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "query": user_query,
            "error": f"YAML格式化错误: {str(type(e))} - {str(e)}"
        }
        return yaml.dump(error_data, allow_unicode=True, default_flow_style=False)


@traceable
def fofa_agentic_search(search_request, test_json=False):
    """处理FOFA搜索请求或滚动ID请求的核心功能函数
    
    参数:
        search_request (str): 自然语言表述的搜索请求，或格式为"scrollid: xxxxxxx"的请求，用于对历史搜索结果进行分页获取
        test_json (bool): 是否启用JSON解析错误测试模式
    
    功能:
        - 处理搜索请求，使用agent执行搜索
        - 处理滚动ID请求，分页加载并显示保存的搜索结果
        - 支持自动分页获取更多结果（最多1000个）
        - 格式化并输出YAML格式的结果
        - 支持JSON解析错误测试模式，用于调试
    
    返回:
        None: 结果直接打印到标准输出
    """
    # 检查是否是scrollid格式的请求
    if search_request.lower().startswith('scrollid: '):
        # 提取scrollid
        scroll_id = search_request[10:].strip()
        
        # 检查scrollid格式（8位字母数字）
        if not (len(scroll_id) == 8 and all(c.isalnum() for c in scroll_id)):
            error_data = {
                'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'query': search_request,
                'error': "scrollid必须是8位字母和数字的组合"
            }
            logger.info(yaml.dump(error_data, allow_unicode=True))
            return
        
        # 尝试加载之前保存的搜索结果
        saved_data = load_search_result_from_file(scroll_id)
        if not saved_data:
            error_data = {
                'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'query': search_request,
                'error': f"找不到scrollid为'{scroll_id}'的搜索结果"
            }
            logger.info(yaml.dump(error_data, allow_unicode=True))
            return
        
        try:
            # 获取所有保存的结果和相关信息
            all_results = saved_data.get('results', [])
            total_assets = max(saved_data.get('total', 0), len(all_results))  # 确保total_assets不为0
            search_query = saved_data.get('query', '')
            next_page_index = saved_data.get('next_page_index', 0)
            total_fetched = len(all_results)
            
            # 修复next_page_index过大的问题
            if next_page_index >= len(all_results):
                # 如果next_page_index超过了结果总数，重新设置为0（显示第一页）
                next_page_index = 0
                saved_data['next_page_index'] = next_page_index
                save_search_result_to_file(scroll_id, saved_data)
            
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
            fields = "host,ip,port,title,protocol,banner,product,country_name,region,city".split(',')
            
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
                    'product': '',
                    'location': ''
                }
                
                # 填充资产信息
                for i, field in enumerate(fields):
                    if i < len(item) and item[i]:
                        if field.lower() in asset:
                            asset[field.lower()] = item[i]
                
                formatted_assets.append(asset)
            
            # 格式化结果
            # 总是包含scroll_id，不管是否有下一页
            result_dict = {
                "success": True,
                "query": search_query,
                "total": total_assets,
                "size": len(current_page_results),
                "results": current_page_results,
                "fields": "host,ip,port,title,protocol,banner,product,country_name,region,city",
                "scroll_id": scroll_id  # 始终传递scroll_id
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
            return
    
    # 非scrollid请求，使用agent搜索处理所有搜索请求
    try:
        # 创建agent
        agent = create_react_agent_for_fofa()
        
        # 生成会话ID
        session_id = f"fofa-agent-{uuid.uuid4().hex[:8]}"
        # 配置递归限制和线程ID
        config = {
            "configurable": {"thread_id": session_id},
            "recursion_limit": 30  # 增加递归限制到30
        }
        
        # 添加系统提示，确保agent了解新功能
        # 生成一个scroll_id用于本次搜索
        new_scroll_id=generate_scroll_id()
        system_message = f"你是fofa网络资产搜索助手，负责把用户的搜索需求转换成fofa的搜索语句，提交给fofa_search工具执行，fofa_search工具会返回一个字典结果，你需要原封不动地返回这个结果，不要添加任何额外解释或格式化。本次搜索使用的scroll_id为{new_scroll_id}"
        
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
                            # 使用新的parse_json_response函数简化JSON解析，传递test_error参数
                            search_results = parse_json_response(content, search_request, test_error=test_json)
                            break  # 获取到结果后立即跳出循环
                    except Exception as e:
                        logger.error(f"解析结果错误: {str(e)}")
                        search_results = {"success": False, "error": f"解析结果错误: {str(e)}"}
        
        # 如果搜索成功，从保存的文件中读取完整结果
        if search_results and search_results.get('success') and search_results.get('scroll_id'):
            scroll_id = search_results.get('scroll_id')
            # 从文件中加载完整结果
            saved_results = load_search_result_from_file(scroll_id)
            
            if saved_results:
                # 更新搜索结果为完整的保存结果
                search_results['results'] = saved_results.get('results', [])
                search_results['size'] = len(saved_results.get('results', []))
                search_results['total'] = saved_results.get('total', 0)
            else:
                logger.warning(f"警告: 无法从文件加载完整结果，使用原始结果")
        
        # 格式化并输出YAML结果
        if search_results and search_results.get('success'):
            # 计算实际的页数
            total_assets = len(search_results.get('results', []))
            actual_total_pages = (total_assets + 19) // 20
            
            # 确保使用从fofa_search获取的scroll_id，不再生成新的scrollid
            scroll_id = search_results.get('scroll_id')
            
            # 更新保存的搜索结果，添加分页信息
            saved_data = {
                'query': search_results.get('query', ''),
                'total': search_results.get('total', 0),
                'scroll_id': scroll_id,  # 使用已有的scroll_id
                'next_page_index': 0,  # 下一页的起始索引，从0开始
                'results': search_results.get('results', []),  # 保存所有结果
                'total_fetched': total_assets
            }
            file_path = save_search_result_to_file(scroll_id, saved_data)
            
            # 确保搜索结果中的scroll_id保持不变
            search_results['scroll_id'] = scroll_id
            
            # 格式化前20条结果为正确的字典列表
            top_20_results = search_results.get('results', [])[:20]
            formatted_top_20 = []
            for idx, asset in enumerate(top_20_results, 1):
                formatted_asset = {
                    "id": idx,
                    "host": asset[0],
                    "ip": asset[1],
                    "port": asset[2],
                    "title": asset[3],
                    "protocol": asset[4],
                    "banner": asset[5],
                    "product": asset[6],
                    "location": f"{asset[7]},{asset[8]},{asset[9]}"
                }
                formatted_top_20.append(formatted_asset)
                
            yaml_output = format_results_to_yaml(
                search_results, 
                search_request,
                cumulative_assets=formatted_top_20,  # 传递格式化后的前20条结果
                current_page=1,
                total_pages=actual_total_pages
            )
            if yaml_output:
                logger.info(yaml_output)
                logger.info(f"提示: 搜索结果已全部保存到 {file_path}")
                logger.info(f"提示: 使用 'python fofa_agent.py \'scrollid: {scroll_id}\' 继续查看下一页")
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

def main():
    """主程序入口，解析命令行参数并调用核心功能"""
    
    # 检查参数
    if len(sys.argv) < 2:
        logger.info("用法: python fofa-agent.py '查找域名example.com的所有资产' 或 python fofa-agent.py 'scrollid: xxxxxxx'")
        logger.info(f"说明: 程序支持自动分页获取最多{MAX_RESULT}个资产，每次返回20条")
        sys.exit(1)
    
    # 提取搜索查询和测试模式标志
    search_request = sys.argv[1]
    
    # 调用核心功能函数，传递测试模式标志
    if test_json:
        fofa_agentic_search(search_request, test_json=test_json)
    else:
        fofa_agentic_search(search_request)

if __name__ == "__main__":
    main()