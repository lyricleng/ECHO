import streamlit as st
from openai import OpenAI
import numpy as np
from quotes import quotes

client = OpenAI()

# 页面设置
st.set_page_config(page_title="人类回声", layout="centered")

# 标题
st.markdown("<h1 style='text-align: center;'>ECHO</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>你在想什么</p>", unsafe_allow_html=True)

st.write("")

# 输入框（更简洁）
user_input = st.text_input("", placeholder="写下一句话...")

def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

@st.cache_resource
def compute_db_embeddings():
    return [get_embedding(q["text"]) for q in quotes]

db_embeddings = compute_db_embeddings()

if user_input:
    with st.spinner("……"):
        user_emb = get_embedding(user_input)

        scores = []
        for i, emb in enumerate(db_embeddings):
            score = cosine(user_emb, emb)
            scores.append((score, i))

        scores.sort(reverse=True)
        top_indices = [idx for _, idx in scores[:8]]
        candidates = [quotes[i] for i in top_indices]

        prompt = f"""
用户输入：
{user_input}

下面是一些人类曾经表达过的句子：
"""

        for i, c in enumerate(candidates):
            prompt += f"{i+1}. {c['text']} ——{c['author']}《{c['source']}》\n"

        prompt += """
请选出思想最接近的一句。
只返回编号。
"""

        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}]
        )

        choice = int(response.choices[0].message.content.strip()) - 1
        best_quote = candidates[choice]

    st.write("")
    st.write("")

    # ✨ 输出（更像作品）
    st.markdown(
        f"<div style='font-size: 24px; text-align: center;'>{best_quote['text']}</div>",
        unsafe_allow_html=True
    )

    st.write("")

    st.markdown(
        f"<div style='text-align: center; color: gray;'>—— {best_quote['author']}《{best_quote['source']}》 ({best_quote['year']})</div>",
        unsafe_allow_html=True
    )