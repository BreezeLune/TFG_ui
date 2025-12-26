# LLM API 配置说明

## 概述

本项目支持通过配置文件管理多个LLM API，包括 OpenAI、Zhipu AI 和 DeepSeek。配置优先级为：**环境变量 > 配置文件 > 默认值**。

## 配置文件位置

- **配置文件**：`backend/api_config.json`（需要手动创建）
- **示例文件**：`backend/api_config.example.json`（已提供，可参考）

## 快速开始

### 1. 创建配置文件

```bash
# 复制示例文件
cp backend/api_config.example.json backend/api_config.json
```

### 2. 编辑配置文件

打开 `backend/api_config.json`，填入你的API密钥：

```json
{
  "openai": {
    "api_key": "sk-your-openai-api-key",
    "base_url": "https://api.openai.com/v1",
    "model": "gpt-3.5-turbo",
    "enabled": true
  },
  "zhipu": {
    "api_key": "your-zhipu-api-key",
    "base_url": "https://open.bigmodel.cn/api/paas/v4/",
    "model": "glm-4",
    "enabled": false
  },
  "deepseek": {
    "api_key": "sk-your-deepseek-api-key",
    "base_url": "https://api.deepseek.com",
    "model": "deepseek-chat",
    "enabled": true
  }
}
```

### 3. 配置说明

每个API配置包含以下字段：

- **api_key**：API密钥（必填）
- **base_url**：API基础URL（必填）
- **model**：使用的模型名称（必填）
- **enabled**：是否启用该API（true/false）


## 环境变量配置（可选）

你也可以通过环境变量设置API密钥，环境变量的优先级高于配置文件：

```bash
# 设置环境变量
export OPENAI_API_KEY="sk-your-openai-key"
export ZHIPU_API_KEY="your-zhipu-key"
export DEEPSEEK_API_KEY="sk-your-deepseek-key"
```

## API选择逻辑

1. 前端通过 `api_choice` 参数选择API（"openai"、"zhipu"、"deepseek"）
2. 后端检查该API是否启用（`enabled: true`）
3. 如果未启用，自动切换到其他已启用的API
4. 检查API密钥是否有效（不能是默认占位符）
5. 如果密钥无效，返回错误提示

## 获取API密钥

### DeepSeek API
1. 访问 [DeepSeek官网](https://www.deepseek.com/)
2. 注册/登录账号
3. 进入控制台，创建API密钥
4. 复制密钥到配置文件

### OpenAI API
1. 访问 [OpenAI官网](https://platform.openai.com/)
2. 注册/登录账号
3. 进入API Keys页面，创建新密钥
4. 复制密钥到配置文件

### Zhipu AI
1. 访问 [智谱AI开放平台](https://open.bigmodel.cn/)
2. 注册/登录账号
3. 获取API密钥
4. 复制密钥到配置文件

## 安全提示

⚠️ **重要**：
- `api_config.json` 已添加到 `.gitignore`，不会被提交到Git仓库
- 不要将包含真实API密钥的配置文件分享给他人
- 如果密钥泄露，请立即在对应平台重新生成密钥

## 故障排查

### 问题1：提示"API密钥未配置"
- 检查 `api_config.json` 文件是否存在
- 检查API密钥是否正确填写（不是占位符）
- 检查 `enabled` 字段是否为 `true`

### 问题2：API调用失败
- 检查网络连接
- 检查API密钥是否有效
- 检查账户余额是否充足
- 查看后端日志获取详细错误信息

### 问题3：配置文件读取失败
- 检查JSON格式是否正确（可以使用在线JSON验证工具）
- 检查文件编码是否为UTF-8
- 检查文件权限

## 动态更新配置

配置文件支持热更新，修改 `api_config.json` 后，下次调用LLM时会自动重新加载配置，无需重启服务。

