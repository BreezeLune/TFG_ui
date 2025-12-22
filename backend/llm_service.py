from openai import OpenAI
import os

# === API 配置中心 ===
# 你可以在这里添加更多模型支持
API_CONFIG = {
    "OpenAI API": {
        "api_key": os.getenv("OPENAI_API_KEY", "sk-xxxxxxxx"), # 替换你的 OpenAI Key
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-3.5-turbo"
    },
    "Zhipu AI": {
        "api_key": os.getenv("ZHIPU_API_KEY", "31af4e1567ad48f49b6d7b914b4145fb.MDVLvMiePGYLRJ7M"),
        "base_url": "https://open.bigmodel.cn/api/paas/v4/",
        "model": "glm-4"
    },
    "DeepSeek": {
        "api_key": os.getenv("DEEPSEEK_API_KEY", "sk-xxxxxxxx"),
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-chat"
    }
}

def query_llm(text, api_choice="Zhipu AI"):
    """
    统一的 LLM 调用接口
    """
    print(f"[LLM Service] 正在调用: {api_choice}")
    
    # 获取配置，如果前端传来的名字匹配不到，默认使用 Zhipu AI
    config = API_CONFIG.get(api_choice)
    if not config:
        print(f"[LLM Service] 未找到 {api_choice} 配置，回退到 Zhipu AI")
        config = API_CONFIG["Zhipu AI"]

    try:
        # 使用 OpenAI SDK 统一调用 (智谱、DeepSeek 等现在都兼容此格式)
        client = OpenAI(
            api_key=config['api_key'],
            base_url=config['base_url']
        )

        response = client.chat.completions.create(
            model=config['model'],
            messages=[
                {"role": "system", "content": "你是一个数字人助手，请用简短、口语化的中文回答用户，字数控制在50字以内。"},
                {"role": "user", "content": text}
            ],
            temperature=0.7,
            max_tokens=150
        )
        
        reply = response.choices[0].message.content
        print(f"[LLM Service] 回复: {reply}")
        return reply

    except Exception as e:
        print(f"[LLM Service] 调用失败: {e}")
        return "抱歉，我现在连接大脑有点问题，请稍后再试。"