import os
import base64
import uuid
import json
from typing import Optional, Dict, Any
import requests
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 从.env读取MAX_RESULT配置，默认200，最大不超过2000
MAX_RESULT = int(os.environ.get("FOFA_MAX_RESULT", "200"))
MAX_RESULT = min(MAX_RESULT, 2000)

def check_tmp_directory():
    """确保.tmp目录存在"""
    tmp_dir = os.path.join(os.getcwd(), '.tmp')
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    return tmp_dir


def load_search_result_from_file(scroll_id: str) -> dict:
    """从.tmp目录加载之前保存的搜索结果"""
    tmp_dir = check_tmp_directory()
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
    tmp_dir = check_tmp_directory()
    file_path = os.path.join(tmp_dir, f"{scroll_id}.json")
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(search_data, f, ensure_ascii=False, indent=2)
    return file_path


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
            
    def search(self, query_str: str, size: int = 20, fields: str = "host,ip,port,title", scroll: Optional[str] = None) -> Dict[str, Any]:
        # 限制单次搜索结果不超过MAX_RESULT个
        if size > MAX_RESULT:
            size = MAX_RESULT
        # 限制单次返回结果数量为20个
        page_size = min(size, 20)
        """
        执行Fofa搜索
        
        Args:
            query_str: 查询字符串，遵循Fofa搜索语法
            size: 返回结果的数量，默认为20
            fields: 返回的字段，默认为"host,ip,port,title"
            scroll: scroll_id，用于分页查询
        
        Returns:
            包含搜索结果的字典
        """
        # 对查询字符串进行Base64编码
        qbase64 = base64.b64encode(query_str.encode('utf-8'))
        
        # 构建请求参数
        params = {
            'email': self.email,
            'key': self.api_key,
            'qbase64': qbase64,
            'size': size,
            'fields': fields
        }
        
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


# 模块导出标记
__all__ = [
    'FofaAPI',
    'generate_scroll_id',
    'check_tmp_directory',
    'load_search_result_from_file',
    'save_search_result_to_file',
    'MAX_RESULT'
]