import yaml
import subprocess
import sys

# 运行fofa-agent.py并捕获输出
try:
    result = subprocess.run(
        ['python', 'fofa-agent.py', 'scrollid: 5ikh3axg'],
        capture_output=True,
        text=True,
        check=True
    )
    output = result.stdout
    
    # 打印捕获的输出以供参考
    print("捕获的YAML输出:")
    print("=" * 50)
    print(output)
    print("=" * 50)
    
    # 尝试解析YAML
except subprocess.CalledProcessError as e:
    print(f"❌ 运行fofa-agent.py失败: {e}")
    print(f"错误输出: {e.stderr}")
    sys.exit(1)

# 尝试解析YAML
try:
    parsed_data = yaml.safe_load(output)
    print("✅ YAML格式验证通过！")
    print("解析后的数据结构:")
    print(f"- 类型: {type(parsed_data)}")
    print(f"- 包含的键: {list(parsed_data.keys())}")
    if 'assets' in parsed_data:
        print(f"- assets数量: {len(parsed_data['assets'])}")
except yaml.YAMLError as e:
    print(f"❌ YAML格式错误: {e}")
    sys.exit(1)

# 验证所有必要字段是否存在
required_fields = ['timestamp', 'query', 'search', 'total', 'current_page', 'total_pages', 'assets']
missing_fields = [field for field in required_fields if field not in parsed_data]

if missing_fields:
    print(f"❌ 缺少必要字段: {missing_fields}")
    sys.exit(1)
else:
    print("✅ 所有必要字段都已存在")
    
    # 验证assets的格式
    if isinstance(parsed_data['assets'], list):
        print("✅ assets是有效的列表")
        for i, asset in enumerate(parsed_data['assets'], 1):
            if isinstance(asset, dict):
                print(f"  ✅ 资产项 {i} 是有效的字典")
                # 检查资产项是否包含必要字段
                asset_required_fields = ['id', 'host', 'ip', 'port', 'title', 'protocol', 'banner', 'product']
                asset_missing = [field for field in asset_required_fields if field not in asset]
                if asset_missing:
                    print(f"    ❌ 资产项 {i} 缺少字段: {asset_missing}")
            else:
                print(f"  ❌ 资产项 {i} 不是字典: {type(asset)}")
    else:
        print(f"❌ assets不是列表: {type(parsed_data['assets'])}")
        sys.exit(1)

sys.exit(0)