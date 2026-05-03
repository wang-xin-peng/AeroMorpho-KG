"""测试DeepSeek API连接"""
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("NVAPI_KEY")
base_url = "https://integrate.api.nvidia.com/v1"
model = "deepseek-ai/deepseek-v4-pro"

print(f"测试连接: {model}")
print(f"API URL: {base_url}")

client = OpenAI(
    api_key=api_key,
    base_url=base_url,
)

response = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "user", "content": "你好，请回复OK"}
    ],
    max_tokens=50,
    temperature=0,
)

print(f"\n回复: {response.choices[0].message.content}")
print("\n✓ API连接成功!")