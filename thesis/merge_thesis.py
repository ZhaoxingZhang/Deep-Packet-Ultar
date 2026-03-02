#!/usr/bin/env python3
"""Merge all thesis chapters into a single document."""

import os

# List of files to merge in order
files_to_merge = [
    "abstract.md",
    "chapter1_introduction.md",
    "chapter2_related_work.md",
    "chapter3_gee_architecture.md",
    "chapter4_experiments.md",
    "chapter5_theoretical_analysis.md",
    "chapter6_conclusion.md",
    "references.md",
    "acknowledgments.md"
]

# Base directory
base_dir = "/Users/weisman/Documents/repo_root/graduate/Deep-Packet-Ultar/thesis/chapters"
output_file = "/Users/weisman/Documents/repo_root/graduate/Deep-Packet-Ultar/thesis/main.md"

# Header
header = """# 面向开放集识别与增量学习的加密流量分类方法研究
# Research on Encrypted Traffic Classification Methods for Open-Set Recognition and Incremental Learning

---

**学科专业**: 网络与信息安全
**作者**: [作者姓名]
**指导教师**: [导师姓名]
**学校**: [学校名称]
**完成时间**: 2026年X月

---

"""

# Merge files
with open(output_file, 'w', encoding='utf-8') as out:
    # Write header
    out.write(header)

    # Merge each file
    for i, filename in enumerate(files_to_merge):
        filepath = os.path.join(base_dir, filename)

        if not os.path.exists(filepath):
            print(f"Warning: {filename} not found, skipping...")
            continue

        print(f"Merging {filename}...")

        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            out.write(content)

        # Add separator between chapters (except for the last one)
        if i < len(files_to_merge) - 1:
            out.write("\n\n---\n\n")

print(f"\nThesis successfully merged to: {output_file}")
print(f"Total files merged: {len([f for f in files_to_merge if os.path.exists(os.path.join(base_dir, f))])}")
