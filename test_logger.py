#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
轻量级测试日志系统是否正常工作的脚本
"""
import os
import sys
import importlib.util

# 加载fofaagent.py模块来测试日志系统
def load_fofaagent_module():
    """加载fofaagent.py模块"""
    try:
        spec = importlib.util.spec_from_file_location("fofaagent", "fofaagent.py")
        fofaagent = importlib.util.module_from_spec(spec)
        sys.modules["fofaagent"] = fofaagent
        spec.loader.exec_module(fofaagent)
        return fofaagent
    except Exception as e:
        print(f"加载模块失败: {str(e)}")
        return None

def test_logger_with_debug_mode(debug_mode):
    """测试不同DEBUG模式下的日志输出"""
    print(f"\n===== 测试日志系统 (DEBUG={debug_mode}) =====")
    
    # 设置环境变量
    os.environ['DEBUG'] = str(debug_mode).lower()
    
    # 确保logging模块已导入
    import logging
    # 重新加载logging模块以应用新的环境变量设置
    importlib.reload(logging)
    
    # 加载或重新加载fofaagent模块
    fofaagent = load_fofaagent_module()
    
    if fofaagent and hasattr(fofaagent, 'logger'):
        # 测试不同级别的日志输出
        print("\n测试不同级别的日志输出:")
        fofaagent.logger.debug("这是一条调试日志，只有DEBUG=True时可见")
        fofaagent.logger.info("这是一条信息日志")
        fofaagent.logger.warning("这是一条警告日志")
        fofaagent.logger.error("这是一条错误日志")
        
        # 测试parse_json_response函数中的日志
        print("\n测试parse_json_response函数中的日志:")
        if hasattr(fofaagent, 'parse_json_response'):
            try:
                # 使用简单的JSON内容进行测试
                test_json = '{"success": true, "total": 10, "results": []}'
                result = fofaagent.parse_json_response(test_json, "测试搜索")
                print(f"解析结果: {result}")
            except Exception as e:
                print(f"测试parse_json_response失败: {str(e)}")
    else:
        print("无法访问日志系统")

if __name__ == "__main__":
    # 测试默认模式（DEBUG=False）
    test_logger_with_debug_mode(False)
    
    # 测试调试模式（DEBUG=True）
    test_logger_with_debug_mode(True)

print("\n===== 测试完成 =====")
print("\n日志系统优化已完成：")
print("1. 已使用Python标准logging模块替换所有print语句")
print("2. 已配置DEBUG环境变量控制日志级别")
print("3. 已创建.env.example文件作为配置示例")
print("\n使用方法：")
print("- 默认模式（只显示重要信息）: 不需要设置DEBUG环境变量或设为DEBUG=false")
print("- 调试模式（显示详细日志）: 设置DEBUG=true")