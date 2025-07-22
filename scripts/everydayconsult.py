import streamlit as st
import os
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import uuid
from datetime import datetime
import streamlit.components.v1 as components

def generate_source_reasons(client, prompt, docs_with_scores):
    """
    各参照ドキュメントがユーザーのプロンプトに本当に関連しているかを判断し、
    関連している場合はその理由を、していない場合はその旨を返します。
    """
    if not docs_with_scores:
        return []

    content_list = []
    for i, (doc, score) in enumerate(docs_with_scores):
        content_list.append(f"<{i+1}>\n{doc.page_content}\n</{i+1}>")
    
    formatted_chunks = "\n\n".join(content_list)

    system_prompt = "あなたは、ユーザーの質問と複数のテキスト断片の関係性を分析する専門家です。各テキスト断片がユーザーの質問に本当に関連しているかを厳密に判断し、関連している場合はその核心的な理由を、関連していない場合はその旨を明確に示してください。"
    
    user_message = f"""以下のユーザーの質問と、それに関連する可能性のあるテキスト断片リストを読んでください。

# ユーザーの質問
{prompt}

# テキスト断片リスト
{formatted_chunks}

# 指示
各テキスト断片について、以下のどちらかの形式で回答してください。
1.  **本当に関連している場合:** `理由: [ここに具体的な理由を1文で記述]`
2.  **関連していない場合:** `理由: [IRRELEVANT]`

番号を付けてリスト形式で出力し、他の余計な言葉は含めないでください。

例:
1. 理由: 〇〇という課題に対する具体的な解決策が示されているため。
2. 理由: [IRRELEVANT]
3. 理由: 質問内のキーワード「△△」に関する詳細な背景を説明しているため。
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0,
            max_tokens=500,
        )
        reasons_text = response.choices[0].message.content
        reasons = []
        for line in reasons_text.strip().split('\n'):
            if '理由: ' in line:
                reason = line.split('理由: ', 1)[1].strip()
                reasons.append(reason)
        
        if len(reasons) == len(docs_with_scores):
            return reasons
        else:
            # パース失敗時は全て無関係として扱う
            return ["[IRRELEVANT]"] * len(docs_with_scores)
    except Exception:
        return ["[IRRELEVANT]"] * len(docs_with_scores)


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
    avatar_url = "assets/avatar.png" if msg["role"] == "assistant" else "user"
    with st.chat_message(msg["role"], avatar=avatar_url):
        st.markdown(msg["content"])
        # アシスタントのメッセージで、かつ参照元情報がある場合
        if msg["role"] == "assistant" and "sources" in msg and msg["sources"]:
            with st.expander("参照元ファイル"):
                for item in msg["sources"]:
                    doc = item["doc"]
                    score = item["score"]
                    reason = item["reason"]
                    
                    st.markdown(f"**ファイル名:** `{doc.metadata.get('source', 'N/A')}`")
                    st.markdown(f"**選択理由:** {reason} (類似度スコア: {score:.4f})")
                    
                    # コンテナを使って参照箇所をスクロール可能に
                    content_html = f"""
                        <div style="background-color: #262730; border-radius: 5px; padding: 10px; height: 150px; overflow-y: auto; border: 1px solid #333;">
                            <pre style="white-space: pre-wrap; word-wrap: break-word; color: #FAFAFA; font-family: 'Source Code Pro', monospace;">{doc.page_content}</pre>
                        </div>
                    """
                    components.html(content_html, height=170)
                    st.divider()

# --- ユーザーからの入力 ---
if prompt := st.chat_input("ここにメッセージを入力してください"):
    # ユーザーのメッセージを履歴に追加
    st.session_state.messages.append({"role": "user", "content": prompt, "id": str(uuid.uuid4())})
    
    # 画面にユーザーのメッセージを表示
    with st.chat_message("user"):
        st.markdown(prompt)

    # --- AIの応答生成 ---
    with st.chat_message("assistant", avatar="assets/avatar.png"):
        message_placeholder = st.empty()
        full_response = ""

        # --- 知識ベースから関連情報を検索 ---
        context = ""
        source_docs_with_reasons = []
        is_relevant_info_found = False
        if db:
            try:
                # ステップ1: 類似度スコアに基づき、候補を検索
                docs_with_scores = db.similarity_search_with_score(prompt, k=5)
                
                if docs_with_scores and docs_with_scores[0][1] <= SIMILARITY_THRESHOLD:
                    
                    # ステップ2: AIによる意味的な関連性チェック(関所)
                    reasons = generate_source_reasons(client, prompt, docs_with_scores)
                    
                    # ステップ3:本当に関連するドキュメントだけをフィルタリング
                    relevant_sources = []
                    for (doc, score), reason in zip(docs_with_scores, reasons):
                        if reason != "[IRRELEVANT]":
                            relevant_sources.append({
                                "doc": doc,
                                "score": score,
                                "reason": reason
                            })
                    
                    # ステップ4: 関連するものが1つでもあれば、コンテキストを構築
                    if relevant_sources:
                        is_relevant_info_found = True
                        source_docs_with_reasons = relevant_sources
                        context += "--- 関連情報 ---\n"
                        for item in source_docs_with_reasons:
                            context += item["doc"].page_content + "\n\n"

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

    # --- 完全な応答をセッション履歴に追加 ---
    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response,
        "sources": source_docs_with_reasons,
        "id": str(uuid.uuid4()),
    })

    # --- 自動スクロールと再実行 ---
    st.session_state.scroll_to_bottom = True
    st.rerun()

# --- 自動スクロールの実行 ---
if st.session_state.get("scroll_to_bottom"):
    components.html(
        "<script>window.scrollTo(0, document.body.scrollHeight);</script>",
        height=0
    )
    st.session_state.scroll_to_bottom = False 