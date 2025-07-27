import streamlit as st
import google.generativeai as genai

st.title("いつでもしゅんさん")

# APIキー設定
try:
    api_key = st.secrets.get("GEMINI_API_KEY", "")
    if not api_key:
        st.error("APIキーが設定されていません")
        st.info("Streamlit CloudのSecretsにGEMINI_API_KEYを設定してください")
        st.stop()
    
    genai.configure(api_key=api_key)
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
            model = genai.GenerativeModel('gemini-pro')
            response = model.generate_content(
                f"あなたは引き寄せの法則の専門家です。親しみやすく答えてください。\n\nユーザー: {prompt}"
            )
            st.write(response.text)
            st.session_state.messages.append({"role": "assistant", "content": response.text})
        except Exception as e:
            st.error(f"エラー: {e}")