from openai import OpenAI
import httpx
from time import sleep
import requests
import json
from requests.auth import HTTPBasicAuth

api_key3 = "sk-Z66xyXtGpj3tKgMT0aBbA98fAfDd4a6aA01d24309cDd5240"
api_key4 = "sk-8hVp1lEGD6uSZ82K772487Cf31A7429bA0575231BbEf392a"

def Client(api_key):
    client = OpenAI(
        base_url="http://47.76.75.25:9000/v1", 
        api_key= api_key,
        # api_key="sk-8O3INsAElO4chajY65DaCa6eA39e481993C521Bd40DaB4A1",
        # api_key="sk-h5wiTRKqt9SBmy1z37E55f4eB1B044718e5cC5E712A36b2b",
        # api_key = "sk-Z66xyXtGpj3tKgMT0aBbA98fAfDd4a6aA01d24309cDd5240",
        # api_key = "sk-3Wa8cKScU2Oylt8G081c82De3e624362Bf5455A898249bC5",
        # api_key = "sk-js9FqvkfeXKMWLV2B787522fAcA14d1fB64aB8B22b2300C2",
        http_client=httpx.Client(
            base_url="http://47.76.75.25:9000/v1",
            follow_redirects=True,
        ),
    )
    return client

def chatgpt_api(instruction=None, data='你好！', temperature=1.0, max_tokens=4096,
                MODEL = "gpt-4o-mini"):
    messages = []
    if instruction:
        messages.append({
            "role": "system",
            "content": instruction
        })
    
    # 如果 data 是字符串，直接添加到 messages 中
    if isinstance(data, str):
        messages.append({
            "role": "user",
            "content": data
        })
        
    # 如果 data 是列表，应该是表示多轮对话历史的二元组列表
    elif isinstance(data, list):
        # 最后一项是用户的 query，不是元组
        for i, (query, response) in enumerate(data):
            if i == len(data) - 1:
                messages.append({
                    "role": "user",
                    "content": query
                })
            else:
                messages.append({
                    "role": "user",
                    "content": query
                })
                messages.append({
                    "role": "assistant",
                    "content": response
                })
    else:
        raise ValueError('data 参数类型错误！')
    
    # print(messages) 
    for _ in range(50):
        try:
            if MODEL == "gpt-3.5-turbo":
                client = Client(api_key = api_key3)
                completion = client.chat.completions.create(
                    model = MODEL,    ##### 根据自己需要更换 #####
                    temperature = temperature,
                    messages = messages,
                    max_tokens = max_tokens
                )
                response = completion.choices[0].message.content
            else:
                url = "http://15.204.101.64:4000/v1/chat/completions"
                headers = {
                    'Content-Type': 'application/json',
                    'Authorization': 'Bearer sk-yetWJJoCPSzdRI3U489061EfBaCa4792858d74Ec4d87B4C5'
                }
                data = {
                    "model": MODEL,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "stream": False}
                response = requests.post(url, json=data, headers=headers).json()
                response = response["choices"][0]["message"]["content"]
                break
            
            break
        except:
            print('retrying...')
            sleep(1)

    # 验证 response 必须是字符串
    try:
        assert isinstance(response, str)
    except:
        print('boom')
    
    # print(response)
    return response

if __name__ == '__main__':
    print(chatgpt_api(data='你好！'))
