import streamlit as st
import os
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import uuid
from datetime import datetime
import streamlit.components.v1 as components

# --- 定数 ---
SIMILARITY_THRESHOLD = 0.7 # 類似度評価のしきい値を再度有効化

# --- 初期設定 ---
st.title("いつでもしゅんさん")

# StreamlitのsecretsからAPIキーを設定
try:
    api_key = st.secrets["OPENAI_API_KEY"]
    client = OpenAI(api_key=api_key)
except KeyError:
    st.error("OpenAI APIキーが.streamlit/secrets.tomlに設定されていません。")
    st.stop()

# --- FAISSインデックスのロード ---
FAISS_INDEX_PATH = "data/faiss_index"

@st.cache_resource
def load_faiss_index(path):
    """FAISSインデックスをロードする（キャッシュを利用）"""
    try:
        embeddings = OpenAIEmbeddings()
        return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        st.error(f"知識ベースの読み込みに失敗しました: {e}")
        return None

db = load_faiss_index(FAISS_INDEX_PATH)

# セッション状態の初期化
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "僕はしゅんさんのクローンです。しゅんさんが教えてくれた情報を元にあなたの質問に答えちゃうよ！引き寄せの法則・オーダーノートを学ぶ中で疑問や人生相談などあればなんなりチャットから教えてください！\n\n※あなたが質問したことはいかなることであっても、しゅんさんや他の人には見えないから、安心してね！",
        "id": str(uuid.uuid4()),
    }]
if "scroll_to_bottom" not in st.session_state:
    st.session_state.scroll_to_bottom = False

# --- チャット履歴の表示 ---
for msg in st.session_state.messages:
    avatar = "assets/avatar.png" if msg["role"] == "assistant" else None
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("参照元ファイル"):
                for i, (doc, score) in enumerate(msg.get("sources", [])):
                    st.markdown(f"**ファイル名:** `{doc.metadata.get('source', 'N/A')}`")
                    st.markdown(f"**選択理由:** ユーザーの質問と内容が類似しているため (類似度スコア: {score:.4f})")
                    st.text_area("参照箇所:", value=doc.page_content, height=100, disabled=True, key=f"source_content_{msg['id']}_{i}")

# --- ユーザー入力の処理 ---
if prompt := st.chat_input("メッセージを入力してください..."):
    # ユーザーのメッセージを履歴に追加・表示
    st.session_state.messages.append({"role": "user", "content": prompt, "id": str(uuid.uuid4())})
    with st.chat_message("user"):
        st.markdown(prompt)

    # アシスタントの応答を生成・ストリーミング表示
    with st.chat_message("assistant", avatar="assets/avatar.png"):
        message_placeholder = st.empty()
        full_response = ""
        
        # --- 知識ベースから関連情報を検索 ---
        context = ""
        source_docs = []
        is_relevant_info_found = False
        if db:
            try:
                docs_with_scores = db.similarity_search_with_score(prompt, k=5)
                # 最も関連性の高いドキュメントのスコアをしきい値と比較
                if docs_with_scores and docs_with_scores[0][1] <= SIMILARITY_THRESHOLD:
                    is_relevant_info_found = True
                    source_docs = docs_with_scores
                    context += "--- 関連情報 ---\n"
                    for doc, score in source_docs:
                        context += doc.page_content + "\n\n"
            except Exception as e:
                st.warning(f"知識ベースの検索中にエラーが発生しました: {e}")

        # --- システムプロンプトの準備 ---
        try:
            with open("docs/system_prompt.md", "r", encoding="utf-8") as f:
                system_prompt = f.read()
        except FileNotFoundError:
            st.warning("docs/system_prompt.md が見つかりません。")
            system_prompt = "You are a helpful assistant."

        final_system_prompt = system_prompt
        # 関連情報が見つかった場合のみ、その情報をプロンプトに追加
        if is_relevant_info_found:
            final_system_prompt += "\n\n" + context
        else:
            # 見つからない場合は「なし」という明確な信号を送る
            final_system_prompt += "\n\n--- 関連情報 ---\nなし"

        # --- API呼び出しとストリーミング ---
        try:
            messages_for_api = [
                {"role": "system", "content": final_system_prompt},
                *[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ]
            ]
            
            stream = client.chat.completions.create(
                model="gpt-4o",
                messages=messages_for_api,
                stream=True,
            )
            for chunk in stream:
                full_response += (chunk.choices[0].delta.content or "")
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)

        except Exception as e:
            st.error(f"AIとの通信中にエラーが発生しました: {e}")
            full_response = "申し訳ありません、応答を生成できませんでした。"

        # --- 参照元の表示 (プロコーチモードではない場合) ---
        if full_response and "プロコーチモード" not in full_response and source_docs:
            with st.expander("参照元ファイル"):
                for i, (doc, score) in enumerate(source_docs):
                    st.markdown(f"**ファイル名:** `{doc.metadata.get('source', 'N/A')}`")
                    st.markdown(f"**選択理由:** ユーザーの質問と内容が類似しているため (類似度スコア: {score:.4f})")
                    st.text_area("参照箇所:", value=doc.page_content, height=100, disabled=True, key=f"stream_source_{i}")

    # --- 完全な応答をセッション履歴に追加 ---
    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response,
        "sources": source_docs,
        "id": str(uuid.uuid4()),
    })
    
    # 次の再描画でスクロールを実行するようにフラグを立てる
    st.session_state.scroll_to_bottom = True
    st.rerun()

# --- 自動スクロールの実行 ---
if st.session_state.get("scroll_to_bottom"):
    components.html(
        "<script>window.scrollTo(0, document.body.scrollHeight);</script>",
        height=0
    )
    st.session_state.scroll_to_bottom = False 