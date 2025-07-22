import streamlit as st
import os
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import uuid
from datetime import datetime
import streamlit.components.v1 as components

# --- 定数 ---
SIMILARITY_THRESHOLD = 0.7 # 類似度のしきい値
REQUEST_LOG_FILE = "shun_requests.log"

# --- 初期設定 ---
st.title("いつでもしゅんさん")

# StreamlitのsecretsからAPIキーを設定
try:
    api_key = st.secrets["OPENAI_API_KEY"]
    client = OpenAI(api_key=api_key)
except KeyError:
    st.error("OpenAI APIキーが.streamlit/secrets.tomlに設定されていません。")
    st.stop()

# --- 定数 ---
KNOWLEDGE_BASE_DIR = "google_api/google_docs_source" # 検索対象を限定

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
        "sources": [],
        "id": str(uuid.uuid4()),
        "offer_request": False, # リクエスト提案中かどうかのフラグ
        "request_sent": False, # リクエスト送信済みかどうかのフラグ
        "coaching_mode": False, # 質問コーチング中かどうかのフラグ
    }]
if "advice_mode" not in st.session_state:
    st.session_state.advice_mode = False

# --- ヘルパー関数 ---
def log_request(query_to_log):
    """ユーザーの質問をログファイルに記録する"""
    with open(REQUEST_LOG_FILE, "a", encoding="utf-8") as f:
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "request_content": query_to_log,
        }
        f.write(str(log_entry) + "\n")

def rephrase_and_log_request(user_prompt):
    """ユーザーの質問をプライバシーに配慮してリフレーズし、ログに記録する"""
    try:
        rephrase_system_prompt = (
            "あなたは、ユーザーからの質問を、プライバシーに配慮しつつ、コーチである「しゅんさん」へのリクエストとして要約するアシスタントです。"
            "以下のルールに従って、質問を1文のリクエストに変換してください。\n"
            "・ユーザーの個人的な話や固有名詞は絶対に含めないでください。\n"
            "・あくまで一般的な相談や質問の形にしてください。\n"
            "・元の質問の意図や核となるトピックは維持してください。\n\n"
            "例1：『〇〇さん（姉）との関係でいつもイライラしてしまいます。どうすればいいですか？』→『家族関係（特に姉妹）の悩みについて、関係を改善するためのアドバイスを求めています。』\n"
            "例2：『副業で月5万円稼ぎたいのですが、何から始めればいいかわかりません。』→『副業での収益化（月5万円目標）に関する具体的な始め方についての質問があります。』"
        )
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": rephrase_system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            stream=False,
            temperature=0.2,
        )
        rephrased_query = response.choices[0].message.content or "リクエストの要約に失敗しました。"
        
        log_content = f"[自動検知] {rephrased_query}"
        log_request(log_content)
        st.toast("リクエストを自動記録しました📝")

    except Exception as e:
        log_request(f"[自動検知エラー] 質問の要約に失敗しました。Error: {e}")
        st.error(f"リクエストの自動ログ記録中にエラーが発生しました: {e}")


# --- チャット履歴の表示 ---
for idx, msg in enumerate(st.session_state.messages):
    avatar = "assets/avatar.png" if msg["role"] == "assistant" else None
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        # --- 参照元とフィードバック機能の表示 ---
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("参照元ファイル"):
                for i, source_item in enumerate(msg["sources"]):
                    if isinstance(source_item, tuple) and len(source_item) == 2:
                        source_doc, score = source_item
                        source_key = source_doc.metadata.get('source', 'N/A')
                        st.markdown(f"**ファイル名:** `{source_key}`")
                        st.markdown(f"**選択理由:** ユーザーの質問と内容が類似しているため (類似度スコア: {score:.4f})")
                        st.text_area("参照箇所:", value=source_doc.page_content, height=100, disabled=True, key=f"source_content_{msg['id']}_{i}")
        
        # --- リクエスト/アドバイスボタンの表示 ---
        if msg.get("offer_request") and not msg.get("request_sent"):
            col1, col2 = st.columns(2)
            with col1:
                if st.button("しゅんさんにリクエストを送る", key=f"request_button_{msg['id']}", use_container_width=True):
                    # コーチングモードを開始する
                    st.session_state.messages[idx]["request_sent"] = True # ボタンを消すために送信済みにする
                    st.session_state.coaching_mode = True
                    
                    # コーチング開始のメッセージを追加
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": "わかった！そしたら、どんな質問をしゅんさんに送ろうか？質問が浮かばない時は僕がサポートするから、遠慮なくいってね。",
                        "id": str(uuid.uuid4())
                    })
                    st.rerun()
            with col2:
                if st.button("クローンしゅんさんのアドバイスを聞く", key=f"advice_button_{msg['id']}", use_container_width=True):
                    st.session_state.messages[idx]["request_sent"] = True # ボタンを消す
                    st.session_state.advice_mode = True
                    st.rerun() # advice_modeをトリガーに再実行


# --- アドバイスモードの応答生成 ---
if st.session_state.get("advice_mode", False):
    st.session_state.advice_mode = False # 一度使ったらリセット
    with st.chat_message("assistant", avatar="assets/avatar.png"):
        message_placeholder = st.empty()
        full_response = ""

        # ナレッジベース全体をコンテキストとして形成
        context_docs = []
        if db:
            # FAISSインデックスからすべてのドキュメントIDを取得し、ドキュメントを取得する
            docstore = st.session_state.db.docstore
            all_doc_ids = list(docstore._dict.keys())
            retrieved_docs = [docstore.search(doc_id) for doc_id in all_doc_ids]
            # 不要なNoneをフィルタリング
            retrieved_docs = [doc for doc in retrieved_docs if doc is not None]

            # 取得したドキュメントを結合してコンテキストを作成
            context = "--- 参考ナレッジベース ---\n"
            for doc in retrieved_docs:
                if doc: # 取得できたドキュメントのみ追加
                    context += f"ファイル名: {doc.metadata.get('source', 'N/A')}\n"
                    context += doc.page_content + "\n---\n"
        
        # システムプロンプトを読み込む
        try:
            with open("docs/system_prompt.md", "r", encoding="utf-8") as f:
                system_prompt = f.read()
        except FileNotFoundError:
            st.warning("docs/system_prompt.md が見つかりません。")
            system_prompt = "You are a helpful assistant."

        # 検索したコンテキストをシステムプロンプトに結合
        final_system_prompt = system_prompt + "\n\n" + context

        # API呼び出し
        try:
            messages_for_api = [
                {"role": "system", "content": final_system_prompt},
                *[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages if m.get("role") in ["user", "assistant"]
                ]
            ]
            stream = client.chat.completions.create(
                model="gpt-4o",
                messages=messages_for_api, # type: ignore
                stream=True,
            )
            for chunk in stream:
                full_response += (chunk.choices[0].delta.content or "")
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)

        except Exception as e:
            st.error(f"AIとの通信中にエラーが発生しました: {e}")
            full_response = "申し訳ありません、応答を生成できませんでした。"

    # アドバイスの応答を履歴に追加
    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response,
        "sources": [],
        "id": str(uuid.uuid4()),
        "offer_request": False,
        "request_sent": True,
        "coaching_mode": False,
    })


# --- ユーザー入力の処理 ---
if prompt := st.chat_input("メッセージを入力してください..."):
    # コーチングモードの処理
    if st.session_state.get("coaching_mode", False):
        log_content = f"[ユーザー確認済] {prompt}"
        log_request(log_content)
        st.session_state.coaching_mode = False
        st.session_state.messages.append({"role": "user", "content": prompt, "id": str(uuid.uuid4())})
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"「{prompt}」だね。OK、この内容でしゅんさんに送っておくね！",
            "id": str(uuid.uuid4())
        })
        st.rerun()

    # アドバイスモードの処理
    if st.session_state.get("advice_mode", False):
        log_request(prompt)
        st.session_state.advice_mode = False
        st.session_state.messages.append({"role": "user", "content": prompt, "id": str(uuid.uuid4())})
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"「{prompt}」だね。OK、この内容でしゅんさんに送っておくね！",
            "id": str(uuid.uuid4())
        })
        st.rerun()

    # 通常の応答処理
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
        offer_shun_request = False

        if db:
            try:
                docs_with_scores = db.similarity_search_with_score(prompt, k=5)
                if docs_with_scores and docs_with_scores[0][1] <= SIMILARITY_THRESHOLD:
                    source_docs = docs_with_scores
                    context += "--- 関連情報 ---\n"
                    for doc, score in source_docs:
                        context += doc.page_content + "\n\n"
                else:
                    offer_shun_request = True
                    context = "--- 関連情報 ---\nなし"
                    # 回答できないと判断した時点で、リクエストを要約してログに記録
                    rephrase_and_log_request(prompt)
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
        if context:
            final_system_prompt += "\n\n" + context

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

        # --- 参照元の表示 ---
        if source_docs:
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
        "offer_request": offer_shun_request,
        "request_sent": False,
        "coaching_mode": False
    })
    
    # 会話の最後に自動でスクロールするためのJavaScriptハック
    components.html(
        "<script>window.scrollTo(0, document.body.scrollHeight);</script>",
        height=0
    ) 