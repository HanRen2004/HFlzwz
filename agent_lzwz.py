import streamlit as st
import requests
import uuid
import dashscope
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.tools import tool
from langchain_community.llms.tongyi import Tongyi
import os
from langchain.agents import initialize_agent, AgentType, Tool
from langchain_core.prompts import PromptTemplate
from openai import OpenAI
from langchain_openai import ChatOpenAI
from flask import Flask, request, jsonify
from streamlit.components.v1 import html
import base64
import threading
import time

# 页面配置
st.set_page_config(page_title="励志文章推荐", page_icon="📖")

# 自定义 CSS 样式
custom_css = """
<style>
    /* 页面背景和全局样式 */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        background-size: 400% 400%;
        animation: gradientBG 15s ease infinite;
    }
    
    @keyframes gradientBG {
        0% { background-position: 0% 50% }
        50% { background-position: 100% 50% }
        100% { background-position: 0% 50% }
    }
    
    /* 标题样式 */
    h1 {
        font-size: 3rem;
        font-weight: 700;
        color: #FF6B6B;
        text-align: center;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 10px;
        background: linear-gradient(90deg, #FF6B6B, #FFD166, #06D6A0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: rainbow 6s ease infinite;
    }

    @keyframes rainbow { 
        0%{background-position:0% 50%}
        50%{background-position:100% 50%}
        100%{background-position:0% 50%}
    }

    h2 {
        font-size: 1.7rem;
        color: #4A90E2;
        text-align: center;
        margin-bottom: 30px;
        font-family: 'Arial', sans-serif;
        font-style: italic;
        letter-spacing: 1px;
    }

    /* 装饰元素 */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: url('data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSI4MDAiIGhlaWdodD0iNjAwIiB2aWV3Qm94PSIwIDAgODAwIDYwMCI+CiAgPGRlZnM+CiAgICA8bGluZWFyR3JhZGllbnQgaWQ9ImdyYWQxIiB4MT0iMCUiIHkxPSIwJSIgeDI9IjEwMCUiIHkyPSIxMDAlIj4KICAgICAgPHN0b3Agb2Zmc2V0PSIwJSIgc3R5bGU9InN0b3AtY29sb3I6I0ZGNkI2QjtzdG9wLW9wYWNpdHk6MC42IiAvPgogICAgICA8c3RvcCBvZmZzZXQ9IjUwJSIgc3R5bGU9InN0b3AtY29sb3I6I0ZGRDk2NjtzdG9wLW9wYWNpdHk6MC40IiAvPgogICAgICA8c3RvcCBvZmZzZXQ9IjEwMCUiIHN0eWxlPSJzdG9wLWNvbG9yOiMwNkQ2QTA7c3RvcC1vcGFjaXR5OjAuNSIgLz4KICAgIDwvbGluZWFyR3JhZGllbnQ+CiAgICA8ZmlsdGVyIGlkPSJibHVyIiB4PSItNTAlIiB5PSItNTAlIiB3aWR0aD0iMjAwJSIgaGVpZ2h0PSIyMDAlIj4KICAgICAgPGZlR2F1c3NpYW5CbHVyIGluPSJTb3VyY2VHcmFwaGljIiBzdGREZXZpYXRpb249IjMwIiAvPgogICAgPC9maWx0ZXI+CiAgPC9kZWZzPgogIDxjaXJjbGUgY3g9IjIwMCIgY3k9IjE1MCIgcj0iODAiIGZpbGw9IiNGRkQ5NjYiIG9wYWNpdHk9IjAuNCIgZmlsdGVyPSJ1cmwoI2JsdXIpIj4KICAgIDxhbmltYXRlIGF0dHJpYnV0ZU5hbWU9InIiIHZhbHVlcz0iODA7OTU7ODAiIGR1cj0iOHMiIHJlcGVhdENvdW50PSJpbmRlZmluaXRlIiAvPgogICAgPGFuaW1hdGUgYXR0cmlidXRlTmFtZT0ib3BhY2l0eSIgdmFsdWVzPSIwLjQ7MC42OzAuNCIgZHVyPSI4cyIgcmVwZWF0Q291bnQ9ImluZGVmaW5pdGUiIC8+CiAgPC9jaXJjbGU+CiAgPGNpcmNsZSBjeD0iNjAwIiBjeT0iNDAwIiByPSI5MCIgZmlsbD0iI0ZGNkI2QiIgb3BhY2l0eT0iMC4zIiBmaWx0ZXI9InVybCgjYmx1cikiPgogICAgPGFuaW1hdGUgYXR0cmlidXRlTmFtZT0iciIgdmFsdWVzPSI5MDsxMTA7OTAiIGR1cj0iMTBzIiByZXBlYXRDb3VudD0iaW5kZWZpbml0ZSIgLz4KICAgIDxhbmltYXRlIGF0dHJpYnV0ZU5hbWU9Im9wYWNpdHkiIHZhbHVlcz0iMC4zOzAuNTswLjMiIGR1cj0iMTBzIiByZXBlYXRDb3VudD0iaW5kZWZpbml0ZSIgLz4KICA8L2NpcmNsZT4KICA8Y2lyY2xlIGN4PSIxMDAiIGN5PSI1MDAiIHI9IjcwIiBmaWxsPSIjMDZENkEwIiBvcGFjaXR5PSIwLjMiIGZpbHRlcj0idXJsKCNibHVyKSI+CiAgICA8YW5pbWF0ZSBhdHRyaWJ1dGVOYW1lPSJyIiB2YWx1ZXM9IjcwOzg1OzcwIiBkdXI9IjlzIiByZXBlYXRDb3VudD0iaW5kZWZpbml0ZSIgLz4KICAgIDxhbmltYXRlIGF0dHJpYnV0ZU5hbWU9Im9wYWNpdHkiIHZhbHVlcz0iMC4zOzAuNTswLjMiIGR1cj0iOXMiIHJlcGVhdENvdW50PSJpbmRlZmluaXRlIiAvPgogIDwvY2lyY2xlPgogIDxjaXJjbGUgY3g9IjcwMCIgY3k9IjEwMCIgcj0iNjAiIGZpbGw9IiM0QTkwRTIiIG9wYWNpdHk9IjAuMyIgZmlsdGVyPSJ1cmwoI2JsdXIpIj4KICAgIDxhbmltYXRlIGF0dHJpYnV0ZU5hbWU9InIiIHZhbHVlcz0iNjA7NzU7NjAiIGR1cj0iN3MiIHJlcGVhdENvdW50PSJpbmRlZmluaXRlIiAvPgogICAgPGFuaW1hdGUgYXR0cmlidXRlTmFtZT0ib3BhY2l0eSIgdmFsdWVzPSIwLjM7MC41OzAuMyIgZHVyPSI3cyIgcmVwZWF0Q291bnQ9ImluZGVmaW5pdGUiIC8+CiAgPC9jaXJjbGU+Cjwvc3ZnPg==');
        opacity: 0.5;
        z-index: -1;
        pointer-events: none;
        animation: floatBackground 60s ease-in-out infinite alternate;
    }
    
    @keyframes floatBackground {
        0% { background-position: 0% 0%; }
        100% { background-position: 100% 100%; }
    }

    /* 结果区块样式 */
    .result-block {
        border: none;
        padding: 20px;
        margin-bottom: 25px;
        border-radius: 12px;
        background: rgba(255, 255, 255, 0.8);
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.15);
        backdrop-filter: blur(4px);
        border: 1px solid rgba(255, 255, 255, 0.18);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }

    .result-block:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 36px rgba(31, 38, 135, 0.25);
    }

    .result-block h3 {
        font-size: 1.4rem;
        color: #FF6B6B;
        margin-bottom: 15px;
        border-bottom: 2px solid #FFD166;
        padding-bottom: 8px;
        font-weight: 600;
    }

    .result-block p {
        font-size: 1.1rem;
        color: #444;
        margin-bottom: 8px;
        line-height: 1.6;
    }

    .result-block a {
        display: inline-block;
        margin-top: 15px;
        padding: 10px 20px;
        background: linear-gradient(90deg, #FF6B6B, #FFD166);
        color: white;
        text-decoration: none;
        border-radius: 30px;
        font-weight: 600;
        letter-spacing: 0.5px;
        box-shadow: 0 4px 15px rgba(255, 107, 107, 0.4);
        transition: all 0.3s ease;
    }

    .result-block a:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(255, 107, 107, 0.6);
        background: linear-gradient(90deg, #FFD166, #FF6B6B);
    }
    
    /* 分隔线样式 */
    hr {
        border: 0;
        height: 3px;
        background-image: linear-gradient(to right, rgba(0, 0, 0, 0), rgba(255, 107, 107, 0.75), rgba(0, 0, 0, 0));
        margin: 30px 0;
    }
    
    /* 自定义滚动条 */
    ::-webkit-scrollbar {
        width: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #FF6B6B, #FFD166);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, #FFD166, #FF6B6B);
    }
    
    /* 代码块样式 */
    .code-content {
        background: rgba(240, 240, 245, 0.8);
        padding: 15px;
        border-radius: 8px;
        font-family: monospace;
        white-space: pre-wrap;
        margin: 15px 0;
        border-left: 4px solid #4A90E2;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
    }
    
    .inline-code {
        background: rgba(240, 240, 245, 0.8);
        padding: 2px 5px;
        border-radius: 4px;
        font-family: monospace;
        font-size: 0.9em;
        color: #FF6B6B;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# 页面标题和小标题
st.title("励志文章推荐")
st.markdown('<h2 style="font-family: Arial;">相信困难是存在的，但只是暂时的</h2>', unsafe_allow_html=True)

# 添加欢迎横幅
welcome_banner = """
<div style="background: linear-gradient(90deg, #FFD166, #06D6A0); padding: 15px; border-radius: 10px; margin-bottom: 25px; 
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1); animation: pulse 2s infinite;">
    <h3 style="color: white; text-align: center; margin: 0; text-shadow: 1px 1px 2px rgba(0,0,0,0.2);">✨ 欢迎来到励志文章推荐 ✨</h3>
    <p style="color: white; text-align: center; margin: 10px 0 0 0; font-size: 1.1rem;">每一次阅读，都是一次心灵的成长之旅</p>
</div>

<style>
@keyframes pulse {
    0% { box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1); }
    50% { box-shadow: 0 8px 25px rgba(0, 0, 0, 0.2); }
    100% { box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1); }
}
</style>
"""
st.markdown(welcome_banner, unsafe_allow_html=True)

# 添加装饰性分隔线
st.markdown('<hr>', unsafe_allow_html=True)

# 设置 API 密钥
os.environ["DASHSCOPE_API_KEY"] = os.getenv("DASHSCOPE_API_KEY", "sk-38a6f574d6c6483eae5c32998a16822a")
os.environ["DASHSCOPE_API_BASE"] = os.getenv("DASHSCOPE_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1")



# 创建网络搜索工具
@tool
def bocha_websearch_tool(query: str, count: int = 20) -> str:
    """
    使用Bocha Web Search API 网页搜索
    """
    url = 'https://api.bochaai.com/v1/web-search'
    headers = {
        'Authorization': f'Bearer sk-6012a020f72d4c26ae5ad415300c94f9',
        'Content-Type': 'application/json'
    }
    data = {
        "query": query,
        "freshness": "noLimit",
        "summary": True,
        "count": count
    }

    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        try:
            json_response = response.json()
            if json_response["code"] == 200 and json_response.get("data"):
                webpages = json_response["data"]["webPages"]["value"]
                if not webpages:
                    return "未找到相关结果."
                formatted_results = ""
                for idx, page in enumerate(webpages, start=1):
                    formatted_results += (
                        f"引用：{idx}\n"
                        f"标题：{page['name']}\n"
                        f"URL: {page['url']}\n"
                        f"摘要：{page['summary']}\n"
                        f"网站名称：{page['siteName']}\n"
                        f"网站图标：{page['siteIcon']}\n"
                        f"发布时间：{page['dateLastCrawled']}\n\n"
                    )
                return formatted_results.strip()
            else:
                return f"搜索失败，原因：{json_response.get('message', '未知错误')}"
        except Exception as e:
            return f"处理搜索结果失败，原因是：{str(e)}\n原始响应：{response.text}"
    else:
        return f"搜索API请求失败，状态码：{response.status_code}, 错误信息：{response.text}"


memory = ConversationBufferMemory(memory_key="chat_history",return_messages=True)

llm = ChatOpenAI(
    model="qwen-max",
    temperature=0.8,
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

bocha_tool = Tool(
    name="Bocha Web Search",
    func=bocha_websearch_tool,
    description="使用Bocha Web Search API进行搜索互联网网页，输入应为搜索查询字符串，输出将返回搜索结果的详细信息。包括网页标题、网页URL",
)



#搜索工具提示词
agent_prompt = """
知乎盐选免费专栏 高赞励志文章
豆瓣日记公开文章 心理成长故事
简书公开集 心理自愈爆款
今日头条心理健康频道 无需登录文章
HTTP状态码200 直接访问文章
无Cookie验证 心理文章 直链
无Referer限制 励志故事 HTTPS链接
含「免费试读」标签 心理文章
标注「公共可见」属性 成长故事
显示「全站热门」标识 治疗文章

读取在bocha_tool返回结果中可用的网址链接，并返回
"""


agent = initialize_agent(
    tools=[bocha_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    memory=memory,
    verbose=True,
    agent_kwargs={"agent_prompt": agent_prompt, 'memory': memory}
)




#大语言模型提示词
prompt_template_with_search_results = """
{previous_conversation}

最新的搜索结果如下：
{search_results}

请根据以上信息推荐近期可直接访问的励志文章,要求：
1. 必须满足以下条件：
   - 无需登录/关注/收藏即可完整阅读
   - 直连链接（非首页跳转）
   - 内容含可操作心理技巧或权威背书
2. 输出格式严格遵循：
   - 文章标题（含热度数据，如：24小时阅读增量）
   - 发布平台及免登录标识（例：知乎·盐选免费）
   - 作者资质（需含专业认证链接）
   - 国内直连链接（HTTPS协议，已验证可访问，并且应该在bocha_tool的搜索返回结果）
   - 核心方法论标签（如：正念练习/认知重构）
3. 排除以下类型：
   - 需要微信扫码登录的内容
   - 显示"试读结束"的片段
   - 强制跳转到App下载的链接
"""

final_prompt = PromptTemplate(
    input_variables=["previous_conversation", "search_results"],
    template=prompt_template_with_search_results
)



llm_chat = ChatOpenAI(
    model="qwen-max-latest",
    temperature=0.8,
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

chain = LLMChain(llm=llm_chat, prompt=final_prompt)





#用户提问（功能相关）
user_question = "我想得到一些励志文章,请给我可用的网络链接"


response = agent.run(user_question)

# 准备输入给 Final Prompt 的数据
inputs = {
    "previous_conversation": "\n".join([str(message) for message in memory.load_memory_variables({})["chat_history"]]),
    "search_results": response
}

# 获取原始响应
final_response_raw = chain.run(inputs)

# 处理响应中可能存在的代码块，防止显示code框
# 将markdown代码块转换为普通文本格式
import re
final_response = re.sub(r'```(.*?)```', r'<div class="code-content">\1</div>', final_response_raw, flags=re.DOTALL)
# 移除单行代码标记
final_response = re.sub(r'`(.*?)`', r'<span class="inline-code">\1</span>', final_response)

# 添加音乐波形装饰
music_wave = """
<div style="margin: 30px 0; text-align: center;">
    <svg xmlns="http://www.w3.org/2000/svg" width="100%" height="80" viewBox="0 0 800 80">
        <defs>
            <linearGradient id="waveGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" style="stop-color:#FF6B6B" />
                <stop offset="50%" style="stop-color:#FFD166" />
                <stop offset="100%" style="stop-color:#06D6A0" />
            </linearGradient>
        </defs>
        <g class="wave-group">
            <path d="M0,40 Q20,10 40,40 T80,40 T120,40 T160,40 T200,40 T240,40 T280,40 T320,40 T360,40 T400,40" 
                  stroke="url(#waveGrad)" stroke-width="3" fill="none">
                <animate attributeName="d" 
                       values="M0,40 Q20,10 40,40 T80,40 T120,40 T160,40 T200,40 T240,40 T280,40 T320,40 T360,40 T400,40;
                               M0,40 Q20,40 40,70 T80,40 T120,70 T160,40 T200,70 T240,40 T280,70 T320,40 T360,70 T400,40;
                               M0,40 Q20,10 40,40 T80,40 T120,40 T160,40 T200,40 T240,40 T280,40 T320,40 T360,40 T400,40" 
                       dur="5s" 
                       repeatCount="indefinite" />
            </path>
            <path d="M400,40 Q420,10 440,40 T480,40 T520,40 T560,40 T600,40 T640,40 T680,40 T720,40 T760,40 T800,40" 
                  stroke="url(#waveGrad)" stroke-width="3" fill="none">
                <animate attributeName="d" 
                       values="M400,40 Q420,10 440,40 T480,40 T520,40 T560,40 T600,40 T640,40 T680,40 T720,40 T760,40 T800,40;
                               M400,40 Q420,40 440,70 T480,40 T520,70 T560,40 T600,70 T640,40 T680,70 T720,40 T760,70 T800,40;
                               M400,40 Q420,10 440,40 T480,40 T520,40 T560,40 T600,40 T640,40 T680,40 T720,40 T760,40 T800,40" 
                       dur="5s" 
                       repeatCount="indefinite" />
            </path>
        </g>
    </svg>
</div>
"""
st.markdown(music_wave, unsafe_allow_html=True)

# 添加阳光气泡装饰
sunshine_bubbles = """
<div style="position: relative; margin: 20px 0;">
    <div class="sunshine-bubble" style="position: absolute; top: -20px; left: 10%; width: 60px; height: 60px; 
                                       background: radial-gradient(circle at 30% 30%, #FFD166, #FF6B6B); 
                                       border-radius: 50%; opacity: 0.7; animation: float 8s ease-in-out infinite;"></div>
    <div class="sunshine-bubble" style="position: absolute; top: 10px; right: 15%; width: 40px; height: 40px; 
                                       background: radial-gradient(circle at 30% 30%, #06D6A0, #4A90E2); 
                                       border-radius: 50%; opacity: 0.5; animation: float 6s ease-in-out infinite 1s;"></div>
    <div class="sunshine-bubble" style="position: absolute; bottom: -15px; left: 30%; width: 50px; height: 50px; 
                                       background: radial-gradient(circle at 30% 30%, #4A90E2, #FF6B6B); 
                                       border-radius: 50%; opacity: 0.6; animation: float 7s ease-in-out infinite 0.5s;"></div>
</div>

<style>
@keyframes float {
    0% { transform: translateY(0) rotate(0deg); }
    50% { transform: translateY(-20px) rotate(10deg); }
    100% { transform: translateY(0) rotate(0deg); }
}
</style>
"""

# 美化结果展示
result_container = f"""
    {final_response}

"""
st.markdown(result_container, unsafe_allow_html=True)

# 添加底部装饰
footer = """
<div style="margin-top: 40px; text-align: center; padding: 20px; color: #888; font-size: 0.9rem;">
    <div style="margin-bottom: 10px;">
        <span style="display: inline-block; width: 8px; height: 8px; background: #FF6B6B; border-radius: 50%; margin: 0 5px;"></span>
        <span style="display: inline-block; width: 8px; height: 8px; background: #FFD166; border-radius: 50%; margin: 0 5px;"></span>
        <span style="display: inline-block; width: 8px; height: 8px; background: #06D6A0; border-radius: 50%; margin: 0 5px;"></span>
    </div>
    <p>每一次阅读，都是一次心灵的成长之旅 ✨</p>
</div>
"""
st.markdown(footer, unsafe_allow_html=True)
