import streamlit as st
from google import genai
from google.genai import types

st.title("いつでもしゅんさん")

# APIキー設定
try:
    api_key = st.secrets.get("GEMINI_API_KEY", "")
    if not api_key:
        st.error("APIキーが設定されていません")
        st.info("Streamlit CloudのSecretsにGEMINI_API_KEYを設定してください")
        st.stop()
    
    client = genai.Client(api_key=api_key)
except Exception as e:
    st.error(f"エラー: {e}")
    st.stop()

# セッション初期化
if "messages" not in st.session_state:
    st.session_state.messages = []

# メッセージ表示
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# 入力処理
if prompt := st.chat_input("メッセージを入力"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.write(prompt)
    
    with st.chat_message("assistant"):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-pro",
                contents=f"あなたは引き寄せの法則の専門家です。親しみやすく答えてください。\n\nユーザー: {prompt}",
                config=types.GenerateContentConfig(
                    temperature=0.7,
                    max_output_tokens=500,
                )
            )
            st.write(response.text)
            st.session_state.messages.append({"role": "assistant", "content": response.text})
        except Exception as e:
            st.error(f"エラー: {e}")