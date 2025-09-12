# Fofa搜索Agent

这是一个基于React框架构建的智能Agent，集成了Fofa网络资产搜索功能，可以帮助用户快速查询和分析网络资产信息。

## 功能特性

- 🧠 **智能推理**：基于大语言模型实现的推理-反思-行动循环
- 🔍 **Fofa搜索**：集成Fofa API，支持网络资产搜索
- 🌐 **Tavily搜索**：获取互联网上的相关信息
- 🖥️ **SSH操作**：远程连接Linux主机执行命令

## 环境要求

- Python 3.8+ 
- 安装相关依赖：`pip install -r requirements.txt`

## 配置说明

1. 复制`.env`文件中的示例配置，替换为您自己的API密钥

```
# OpenAI配置
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini

# Fofa API配置
FOFA_EMAIL=your_email@example.com
FOFA_API_KEY=your_fofa_api_key_here

# Tavily API配置
TAVILY_API_KEY=your_tavily_api_key_here
```

2. 关键配置说明：
   - `OPENAI_API_KEY`：OpenAI API密钥
   - `FOFA_EMAIL`和`FOFA_API_KEY`：Fofa账号和API密钥
   - `TAVILY_API_KEY`：Tavily搜索引擎API密钥

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

1. 启动Agent

```bash
python react_agent.py
```

2. 使用示例：

   - **Fofa搜索示例**：
     ```
     请搜索domain="example.com"的网络资产，返回前5条结果
     ```
     ```
     请帮我查找title包含"登录"的网站，返回前10条
     ```

   - **组合查询示例**：
     ```
     请搜索country="CN" && app="nginx"的服务器，返回IP和端口信息
     ```

## Fofa搜索语法

Fofa支持丰富的搜索语法，常用的语法包括：

- `domain="example.com"`：搜索指定域名的资产
- `ip="1.1.1.1"`：搜索指定IP的资产
- `title="登录"`：搜索标题包含特定关键词的资产
- `app="nginx"`：搜索特定应用的资产
- `port="80"`：搜索特定端口的资产
- `country="CN"`：搜索特定国家的资产
- 逻辑运算符：`&&`（与）、`||`（或）、`!`（非）

## 注意事项

1. Fofa API有调用次数限制，请合理使用
2. 确保.env文件中的API密钥正确配置
3. 对于大批量数据查询，建议调整size参数
4. 使用前请确保安装了所有必要的依赖包

## 常见问题

**Q: 运行时提示"未配置Fofa API认证信息"？**
A: 请检查.env文件中的FOFA_EMAIL和FOFA_API_KEY是否正确配置。

**Q: 搜索结果为空？**
A: 请检查搜索语法是否正确，或者尝试调整搜索条件。

**Q: 如何获取更多返回字段？**
A: 可以在搜索时指定fields参数，例如：`fields="host,ip,port,title,domain,protocol"`

## 更新日志

- v1.0：初始版本，集成Fofa搜索、Tavily搜索和SSH功能