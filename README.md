# Fofa智能搜索Agent

这是一个基于LangGraph框架构建的智能Agent，集成了Fofa网络资产搜索功能，可以帮助用户快速查询和分析网络资产信息。

## 功能特性

- 🧠 **智能推理**：基于大语言模型实现的推理-反思-行动循环
- 🔍 **Fofa搜索**：集成Fofa API，支持网络资产搜索
- 📃 **结果分页**：支持使用scrollid滚动分页查看大量结果
- 💾 **结果保存**：自动将搜索结果保存到文件系统
- 📊 **结构化输出**：使用YAML格式输出结构化搜索结果

## 环境要求

- Python 3.8+ 
- 安装相关依赖：`pip install -r requirements.txt`

## 安装依赖

```bash
pip install -r requirements.txt
```

## 配置说明

1. 复制`.env.example`文件为`.env`，替换为您自己的API密钥

```
# OpenAI配置
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini

# Fofa API配置
FOFA_EMAIL=your_email@example.com
FOFA_API_KEY=your_fofa_api_key_here

# 调试模式
DEBUG=false
```

2. 关键配置说明：
   - `OPENAI_API_KEY`：OpenAI API密钥
   - `FOFA_EMAIL`和`FOFA_API_KEY`：Fofa账号和API密钥
   - `DEBUG`：设置为true启用详细日志输出

## 使用方法

1. 基本搜索

```bash
python fofaagent.py '查找域名example.com的所有资产'
```

2. 查看下一页结果

当搜索结果超过20条时，可以使用返回的scrollid查看下一页：

```bash
python fofaagent.py 'scrollid: xxxxxxxx'
```

3. 使用示例：

   - **Fofa搜索示例**：
     ```bash
     python fofaagent.py '请搜索domain="example.com"的网络资产'
     python fofaagent.py '请帮我查找title包含"登录"的网站'
     ```

   - **组合查询示例**：
     ```bash
     python fofaagent.py '请搜索country="CN" && app="nginx"的服务器'
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

## 项目结构

```
fofa-agent/
├── fofaagent.py       # 主要的Agent实现文件
├── fofaclient.py      # Fofa API的客户端实现
├── .env.example       # 环境变量配置示例
├── .gitignore         # Git忽略文件
├── README.md          # 项目说明文档
└── requirements.txt   # 项目依赖
```

## 注意事项

1. Fofa API有调用次数限制，请合理使用
2. 确保.env文件中的API密钥正确配置
3. 搜索结果默认保存到本地文件系统
4. 使用前请确保安装了所有必要的依赖包
5. 程序默认支持自动分页获取最多5000个资产，每次返回20条

## 常见问题

**Q: 运行时提示"未配置Fofa API认证信息"？**
A: 请检查.env文件中的FOFA_EMAIL和FOFA_API_KEY是否正确配置。

**Q: 搜索结果为空？**
A: 请检查搜索语法是否正确，或者尝试调整搜索条件。

**Q: 如何获取更多日志信息？**
A: 将.env文件中的DEBUG设置为true，可以启用详细日志输出。

## 更新日志

- v1.0：初始版本，集成Fofa搜索、结果分页和保存功能