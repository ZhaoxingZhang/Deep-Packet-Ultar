#!/usr/bin/env python3
"""
论文组合脚本
将 thesis/chapters/ 目录中的各章节文件组合到 thesis/main.md 中
使用方法: python3 assemble_thesis.py
"""

import re
from pathlib import Path


def assemble_thesis(main_file, chapters_dir):
    main_path = Path(main_file)
    chapters_path = Path(chapters_dir)
    
    print(f"读取主文件: {main_path}")
    with open(main_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # 匹配 [插入xxx.md全文] 或 [插入xxx]
    pattern = r"\[插入([^\]]+?)(?:\.md)?(?:全文|)\]"
    
    insertions = re.findall(pattern, content)
    print(f"\n找到 {len(insertions)} 个需要插入的内容:")
    for item in insertions:
        print(f"  - {item}")
    
    modified_content = content
    for item in insertions:
        # 构建占位符 - 匹配原始格式
        placeholder = rf"\[插入{re.escape(item)}(?:\.md)?(?:全文|)\]"
        
        if "TOC" in item or "目录" in item or "章节目录" in item:
            print(f"\n[跳过] 目录占位符: {item}")
            continue
        
        # 如果 item 不含 .md，则添加
        chapter_name = item if item.endswith(".md") else f"{item}.md"
        chapter_file = chapters_path / chapter_name
        
        if not chapter_file.exists():
            print(f"\n[警告] 章节文件不存在: {chapter_file}")
            continue
        
        print(f"\n读取章节: {chapter_file}")
        with open(chapter_file, "r", encoding="utf-8") as f:
            chapter_content = f.read()
        
        # 跳过第一行（通常是重复的章节标题）
        lines = chapter_content.split("\n")
        if len(lines) > 1 and lines[0].startswith("#"):
            print(f"[跳过] 移除重复标题: {lines[0][:50]}...")
            chapter_content = "\n".join(lines[1:])
        
        # 替换占位符
        modified_content = re.sub(placeholder, chapter_content, modified_content)
        print(f"[完成] 已插入章节: {item}")
    
    print(f"\n保存更新后的论文: {main_path}")
    with open(main_path, "w", encoding="utf-8") as f:
        f.write(modified_content)
    
    print("\n论文组合完成!")
    print(f"更新后的文件: {main_path}")


if __name__ == "__main__":
    # 设置路径
    script_dir = Path(__file__).parent
    main_file = script_dir / "main.md"
    chapters_dir = script_dir / "chapters"
    
    # 检查文件是否存在
    if not main_file.exists():
        print(f"错误: 主文件不存在: {main_file}")
        exit(1)
    
    if not chapters_dir.exists():
        print(f"错误: 章节目录不存在: {chapters_dir}")
        exit(1)
    
    # 执行组合
    assemble_thesis(str(main_file), str(chapters_dir))
