import streamlit as st
from openai import OpenAI
import numpy as np
from quotes import quotes

client = OpenAI()

# 页面设置
st.set_page_config(page_title="人类回声", layout="centered")

# UI
st.markdown("<h1 style='text-align: center;'>ECHO</h1>", unsafe_allow_html=True)

st.write("")

user_input = st.text_input("", placeholder="妳在想什麼...")

# ========= 基础函数 =========
def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# ========= 缓存语料向量 =========
@st.cache_resource
def compute_db_embeddings():
    return [get_embedding(q["text"]) for q in quotes]

db_embeddings = compute_db_embeddings()

# ========= 主逻辑 =========
if user_input:

    with st.spinner("……"):

        # 1️⃣ 用户 embedding
        user_emb = get_embedding(user_input)

        # 2️⃣ 语义相似度排序（加权）
        scores = []
        for i, emb in enumerate(db_embeddings):
            score = cosine(user_emb, emb) * quotes[i].get("weight", 1)
            scores.append((score, i))

        scores.sort(reverse=True)
        top_indices = [idx for _, idx in scores[:12]]  # 稍微多一点候选

        # 3️⃣ 🧠 LLM 判断用户“思想类型”（关键升级）
        type_prompt = f"""
用户输入：
{user_input}

这个表达更接近哪种状态？

- struggle（挣扎 / 怀疑 / 困惑）
- acceptance（接受 / 坚持 / 继续生活）
- rebellion（反抗 / 意志 / 对抗世界）
- statement（冷静陈述 / 观察）

只返回一个英文单词。
"""

        type_response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": type_prompt}]
        )

        user_type = type_response.choices[0].message.content.strip().lower()

        # 4️⃣ 按 type 过滤（核心优化）
        filtered = [quotes[i] for i in top_indices if quotes[i]["type"] == user_type]

        # fallback（防止筛太少）
        if len(filtered) < 3:
            filtered = [quotes[i] for i in top_indices]

        candidates = filtered[:8]

        # 5️⃣ 构建最终选择 prompt
        prompt = f"""
用户输入：
{user_input}

该表达的思想类型是：{user_type}

下面是一些人类曾经表达过的句子：
"""

        for i, c in enumerate(candidates):
            prompt += f"{i+1}. {c['text']} ——{c['author']}《{c['source']}》\n"

        prompt += """
请选出“思想最接近”的一句，而不是字面最相似的一句。

判断标准：
- 优先匹配情绪（绝望 / 坚持 / 反抗）
- 优先匹配立场（接受 vs 对抗）
- 不要因为关键词相似而选错

只返回编号。
"""

        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}]
        )

        # 6️⃣ 安全解析
        try:
            choice = int(response.choices[0].message.content.strip()) - 1
            best_quote = candidates[choice]
        except:
            best_quote = candidates[0]

    # ========= 输出 =========
    st.write("")
    st.write("")

    st.markdown(
        f"<div style='font-size: 24px; text-align: center;'>{best_quote['text']}</div>",
        unsafe_allow_html=True
    )

    st.write("")

    st.markdown(
        f"<div style='text-align: center; color: gray;'>—— {best_quote['author']}《{best_quote['source']}》 ({best_quote['year']})</div>",
        unsafe_allow_html=True
    )
