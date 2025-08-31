# Cursor Python 配置指南

## 问题描述
Cursor无法跳转到Python函数的定义，即使安装了JetBrains IDE keymap也没有效果。

## 解决方案

### 1. 已完成的配置

我已经为您的项目创建了以下配置文件：

#### `.vscode/settings.json`
- 设置了正确的Python解释器路径：`/usr/local/Caskroom/miniconda/base/envs/deep_packet/bin/python`
- 配置了Python语言服务器为Pylsp
- 启用了自动导入补全和路径搜索
- 设置了代码格式化和linting

#### `.vscode/launch.json`
- 配置了Python调试器
- 设置了正确的环境变量

#### `pyrightconfig.json`
- 配置了Pyright类型检查器
- 设置了项目路径和排除规则

### 2. 已安装的包

在`deep_packet` conda环境中安装了：
- `python-lsp-server[all]` - Python语言服务器
- `black` - 代码格式化工具
- `flake8` - 代码检查工具

### 3. 使用步骤

1. **重启Cursor**：关闭并重新打开Cursor
2. **选择Python解释器**：
   - 按 `Cmd+Shift+P` 打开命令面板
   - 输入 "Python: Select Interpreter"
   - 选择 `/usr/local/Caskroom/miniconda/base/envs/deep_packet/bin/python`

3. **验证配置**：
   - 打开任意Python文件
   - 将光标放在函数名上
   - 按 `Cmd+Click` 或 `F12` 应该能够跳转到定义

### 4. 快捷键

- `Cmd+Click` 或 `F12` - 跳转到定义
- `Cmd+Shift+O` - 跳转到符号
- `Cmd+T` - 快速打开文件
- `Cmd+Shift+F` - 全局搜索

### 5. 故障排除

如果仍然无法跳转：

1. **检查语言服务器状态**：
   - 打开命令面板 (`Cmd+Shift+P`)
   - 输入 "Python: Restart Language Server"

2. **检查输出面板**：
   - 查看是否有错误信息
   - 检查Python扩展的输出

3. **重新安装扩展**：
   - 卸载并重新安装Python扩展

4. **清除缓存**：
   - 删除 `.vscode` 文件夹中的缓存文件
   - 重启Cursor

### 6. 环境要求

- Python 3.10.18 (在deep_packet环境中)
- Cursor最新版本
- Python扩展已安装

## 注意事项

- 确保始终在`deep_packet` conda环境中工作
- 如果修改了conda环境，需要重新配置解释器路径
- 某些第三方库可能无法提供完整的类型信息 